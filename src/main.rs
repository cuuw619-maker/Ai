use ai::dataset::{dataset_metadata, validate, DatasetFormat};
use ai::inference::{GenerationConfig, InferenceEngine};
use ai::model::{AiNet, ModelConfig};
use ai::tokenizer::{Tokenizer, TokenizerTrainer, TokenizerTrainerConfig};
use ai::training::{Checkpoint, TrainingCommand, TrainingConfig, TrainingEvent, Trainer, TrainingWorker};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None | Some("help") | Some("--help") => print_help(),
        Some("model") => model_command(&args[1..]),
        Some("tokenizer") => tokenizer_command(&args[1..]),
        Some("dataset") => dataset_command(&args[1..]),
        Some("train") => train_command(&args[1..]),
        Some("checkpoint") => checkpoint_command(&args[1..]),
        Some("generate") => generate_command(&args[1..]),
        Some("benchmark") => benchmark_command(),
        Some(other) => Err(format!("unknown command: {other}")),
    }
}

fn print_help() -> Result<(), String> {
    println!(r#"Ai / AiNet v1
Own neural engine, own tokenizer, own model/checkpoint formats.

Commands:
  model create --name NAME --vocab N --embedding N --hidden N --layers N --sequence N --seed N --output PATH
  tokenizer train --dataset PATH --format txt|jsonl --vocab N --output PATH
  dataset validate --dataset PATH [--format txt|jsonl] [--tokenizer PATH]
  train --model PATH --tokenizer PATH --dataset PATH [--format txt|jsonl] [--epochs N] [--sequence N]
        [--accumulation N] [--lr F] [--weight-decay F] [--max-grad-norm F] [--checkpoint-steps N]
        [--memory-mb N] [--max-threads N] [--continuous] [--steps N]
  train pause
  train stop
  train resume --checkpoint PATH --tokenizer PATH --dataset PATH [--format txt|jsonl]
  train status
  checkpoint list [--run run-000001]
  generate --model PATH --tokenizer PATH --prompt TEXT [--tokens N] [--temperature F] [--top-k N] [--top-p F] [--greedy]
  benchmark
"#);
    Ok(())
}

fn model_command(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("create") => {
            let output = required_path(args, "--output", "data/models/model.aimodel");
            let config = ModelConfig {
                architecture: "AiNet-v1.1".into(),
                model_id: required(args, "--name")?.to_string(),
                vocab_size: parse_usize(args, "--vocab", 4096)?,
                embedding_dim: parse_usize(args, "--embedding", 192)?,
                hidden_dim: parse_usize(args, "--hidden", 192)?,
                layer_count: parse_usize(args, "--layers", 6)?,
                sequence_length: parse_usize(args, "--sequence", 128)?,
                seed: parse_u64(args, "--seed", 1)?,
            };
            let model = AiNet::new(config.clone())?;
            let parent = output.parent().unwrap_or_else(|| Path::new("."));
            fs::create_dir_all(parent).map_err(|e| format!("create model directory: {e}"))?;
            model.save(&output)?;
            println!("status=UNTRAINED");
            println!("architecture={}", config.architecture);
            println!("model_id={}", config.model_id);
            println!("parameters={}", model.parameter_count());
            println!("seed={}", config.seed);
            println!("model={}", output.display());
            Ok(())
        }
        _ => Err("usage: Ai.exe model create ...".into()),
    }
}

fn tokenizer_command(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("train") => {
            let dataset = required(args, "--dataset")?;
            let format = dataset_format(args, Path::new(dataset))?;
            let output = required_path(args, "--output", "data/tokenizer/tokenizer.aitok");
            let config = TokenizerTrainerConfig {
                target_vocab_size: parse_usize(args, "--vocab", 4096)?,
                min_frequency: parse_u64(args, "--min-frequency", 2)?,
                max_merges: parse_usize(args, "--max-merges", 3840)?,
                sample_bytes: parse_usize(args, "--sample-bytes", 8 * 1024 * 1024)?,
            };
            let started = Instant::now();
            let tokenizer = TokenizerTrainer::train(Path::new(dataset), format, &config)?;
            tokenizer.save(&output)?;
            println!("vocab={}", tokenizer.vocab_size());
            println!("merges={}", tokenizer.merges.len());
            println!("tokenizer_id={}", tokenizer.tokenizer_id());
            println!("elapsed_ms={}", started.elapsed().as_millis());
            println!("tokenizer={}", output.display());
            Ok(())
        }
        _ => Err("usage: Ai.exe tokenizer train ...".into()),
    }
}

fn dataset_command(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("validate") => {
            let path = required(args, "--dataset")?;
            let format = dataset_format(args, Path::new(path))?;
            let tokenizer = optional(args, "--tokenizer").map(|p| Tokenizer::load(Path::new(p))).transpose()?;
            let report = validate(Path::new(path), format.clone(), tokenizer.as_ref())?;
            let metadata = dataset_metadata(Path::new(path), format)?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({
                "dataset_id": metadata.dataset_id,
                "samples": report.samples,
                "errors": report.errors,
                "empty_samples": report.empty_samples,
                "bytes": report.bytes,
                "estimated_tokens": report.estimated_tokens,
                "content_hash": metadata.content_hash
            })).map_err(|e| e.to_string())?);
            Ok(())
        }
        _ => Err("usage: Ai.exe dataset validate ...".into()),
    }
}

fn train_command(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("pause") => write_control("pause"),
        Some("stop") => write_control("stop"),
        Some("status") => {
            let path = Path::new("data/training-status.json");
            if !path.exists() { return Err("no training status found".into()); }
            println!("{}", fs::read_to_string(path).map_err(|e| format!("read status: {e}"))?);
            Ok(())
        }
        Some("resume") => {
            let checkpoint = required_path(args, "--checkpoint", "data/checkpoints/checkpoint-latest.aicheckpoint");
            let tokenizer = Tokenizer::load(Path::new(required(args, "--tokenizer")?))?;
            let dataset = required(args, "--dataset")?;
            let format = dataset_format(args, Path::new(dataset))?;
            let trainer = Trainer::from_checkpoint(
                checkpoint,
                tokenizer,
                dataset,
                format,
                Path::new("data"),
            )?;
            remove_control();
            let mut worker = TrainingWorker::spawn(trainer, Some(PathBuf::from("data/training.command")));
            worker.send(TrainingCommand::Resume)?;
            drain_worker(&mut worker);
            worker.join();
            Ok(())
        }
        Some("train") | None => {
            let model = AiNet::load(Path::new(required(args, "--model")?))?;
            let tokenizer = Tokenizer::load(Path::new(required(args, "--tokenizer")?))?;
            let dataset = required(args, "--dataset")?;
            let format = dataset_format(args, Path::new(dataset))?;
            let mut config = TrainingConfig::low_end();
            config.epochs = parse_u64(args, "--epochs", config.epochs)?;
            config.sequence_length = parse_usize(args, "--sequence", config.sequence_length)?;
            config.gradient_accumulation = parse_usize(args, "--accumulation", config.gradient_accumulation)?;
            config.learning_rate = parse_f32(args, "--lr", config.learning_rate)?;
            config.weight_decay = parse_f32(args, "--weight-decay", config.weight_decay)?;
            config.max_grad_norm = parse_f32(args, "--max-grad-norm", config.max_grad_norm)?;
            config.checkpoint_interval_steps = parse_u64(args, "--checkpoint-steps", config.checkpoint_interval_steps)?;
            config.memory_budget_mb = parse_usize(args, "--memory-mb", config.memory_budget_mb)?;
            config.max_cpu_threads = parse_usize(args, "--max-threads", config.max_cpu_threads)?;
            config.continuous = has_flag(args, "--continuous");
            config.max_steps = optional(args, "--steps").map(str::parse).transpose().map_err(|e| format!("invalid --steps: {e}"))?;
            remove_control();
            let trainer = Trainer::new(model, tokenizer, dataset, format, config, Path::new("data"))?;
            println!("estimated_training_mb={}", trainer.model_memory_estimate_bytes() / 1024 / 1024);
            let mut worker = TrainingWorker::spawn(trainer, Some(PathBuf::from("data/training.command")));
            worker.send(TrainingCommand::Start)?;
            drain_worker(&mut worker);
            worker.join();
            Ok(())
        }
        _ => Err("usage: Ai.exe train ...".into()),
    }
}

fn drain_worker(worker: &mut TrainingWorker) {
    while let Ok(event) = worker.events.recv() {
        match event {
            TrainingEvent::Started(p) => println!("started run={} step={} epoch={}", p.run_id, p.step, p.epoch),
            TrainingEvent::Step(p) => println!(
                "step={} epoch={} loss={:.6} tokens_seen={} tokens_sec={:.3} grad_norm={:.6} clipped={}",
                p.step, p.epoch, p.loss, p.tokens_seen, p.tokens_per_second, p.gradient_norm, p.clipped
            ),
            TrainingEvent::CheckpointSaved(path) => println!("checkpoint={}", path.display()),
            TrainingEvent::Paused => { println!("state=PAUSED"); break; }
            TrainingEvent::Resumed => println!("state=RESUMING"),
            TrainingEvent::Completed => { println!("state=COMPLETED"); break; }
            TrainingEvent::Stopped => { println!("state=STOPPED"); break; }
            TrainingEvent::Failed(error) => { println!("state=FAILED error={error}"); break; }
        }
    }
}

fn checkpoint_command(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("list") => {
            let dir = if let Some(run) = optional(args, "--run") {
                PathBuf::from("data/runs").join(run).join("checkpoints")
            } else {
                PathBuf::from("data/runs")
            };
            if dir.ends_with("runs") {
                for entry in fs::read_dir(dir).map_err(|e| format!("read runs: {e}"))? {
                    let path = entry.map_err(|e| format!("read run: {e}"))?.path().join("checkpoints");
                    for checkpoint in Checkpoint::list(&path)? {
                        println!("{}", checkpoint.display());
                    }
                }
            } else {
                for checkpoint in Checkpoint::list(&dir)? {
                    println!("{}", checkpoint.display());
                }
            }
            Ok(())
        }
        _ => Err("usage: Ai.exe checkpoint list [--run RUN]".into()),
    }
}

fn generate_command(args: &[String]) -> Result<(), String> {
    let model_path = required_path(args, "--model", "data/models/model.aimodel");
    let tokenizer = Tokenizer::load(Path::new(required(args, "--tokenizer")?))?;
    let prompt = required(args, "--prompt")?;
    let mut engine = InferenceEngine::new(1);
    engine.load(&model_path)?;
    let config = GenerationConfig {
        temperature: parse_f32(args, "--temperature", 0.8)?,
        top_k: parse_usize(args, "--top-k", 40)?,
        top_p: parse_f32(args, "--top-p", 0.9)?,
        greedy: has_flag(args, "--greedy"),
    };
    let ids = engine.generate(prompt, &tokenizer, parse_usize(args, "--tokens", 32)?, &config)?;
    println!("{}", tokenizer.decode(&ids)?);
    Ok(())
}

fn benchmark_command() -> Result<(), String> {
    let tokenizer = Tokenizer::new_default();
    let model = AiNet::new(ModelConfig {
        architecture: "AiNet-v1.1".into(),
        model_id: "benchmark".into(),
        vocab_size: tokenizer.vocab_size(),
        embedding_dim: 32,
        hidden_dim: 32,
        layer_count: 2,
        sequence_length: 32,
        seed: 7,
    })?;
    let input = [1usize; 32];
    let target = [2usize; 32];
    let mut model = model;
    let started = Instant::now();
    for _ in 0..10 {
        model.train_step(&input, &target, None)?;
    }
    let elapsed = started.elapsed().as_secs_f64().max(1e-9);
    println!("training_tokens_per_second={:.3}", 320.0 / elapsed);
    let text = "Привет, hello, 123, 🙂";
    let started = Instant::now();
    let mut total = 0usize;
    for _ in 0..1000 { total += tokenizer.encode(text).len(); }
    let elapsed = started.elapsed().as_secs_f64().max(1e-9);
    println!("tokenizer_tokens_per_second={:.3}", total as f64 / elapsed);
    Ok(())
}

fn write_control(command: &str) -> Result<(), String> {
    fs::create_dir_all("data").map_err(|e| format!("create data directory: {e}"))?;
    fs::write("data/training.command", command).map_err(|e| format!("write training command: {e}"))?;
    println!("command={command}");
    Ok(())
}

fn remove_control() { let _ = fs::remove_file("data/training.command"); }

fn required<'a>(args: &'a [String], key: &str) -> Result<&'a str, String> {
    optional(args, key).ok_or_else(|| format!("missing {key}"))
}

fn optional<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.windows(2).find(|pair| pair[0] == key).map(|pair| pair[1].as_str())
}

fn required_path(args: &[String], key: &str, default: &str) -> PathBuf {
    optional(args, key).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

fn parse_usize(args: &[String], key: &str, default: usize) -> Result<usize, String> {
    optional(args, key).map(str::parse).transpose().map_err(|e| format!("invalid {key}: {e}")).map(|v| v.unwrap_or(default))
}

fn parse_u64(args: &[String], key: &str, default: u64) -> Result<u64, String> {
    optional(args, key).map(str::parse).transpose().map_err(|e| format!("invalid {key}: {e}")).map(|v| v.unwrap_or(default))
}

fn parse_f32(args: &[String], key: &str, default: f32) -> Result<f32, String> {
    optional(args, key).map(str::parse).transpose().map_err(|e| format!("invalid {key}: {e}")).map(|v| v.unwrap_or(default))
}

fn has_flag(args: &[String], key: &str) -> bool { args.iter().any(|arg| arg == key) }

fn dataset_format(args: &[String], path: &Path) -> Result<DatasetFormat, String> {
    optional(args, "--format").map(|value| match value.to_ascii_lowercase().as_str() {
        "txt" => Ok(DatasetFormat::Txt),
        "jsonl" => Ok(DatasetFormat::Jsonl),
        _ => Err(format!("unsupported dataset format: {value}")),
    }).unwrap_or_else(|| DatasetFormat::from_path(path))
}
