use ai::dataset::{dataset_metadata, DatasetFormat, DatasetReader, TrainingStream};
use ai::model::{AiNet, ModelConfig};
use ai::tokenizer::{default_special_tokens, Tokenizer, TokenizerTrainer, TokenizerTrainerConfig};
use ai::training::{TrainingCommand, TrainingConfig, TrainingEvent, Trainer, TrainingWorker};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(name: &str) -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!("ai-phase2-{name}-{}-{stamp}", std::process::id()))
}

fn tokenizer_config() -> TokenizerTrainerConfig {
    TokenizerTrainerConfig {
        target_vocab_size: 300,
        min_frequency: 2,
        max_merges: 38,
        sample_bytes: 64 * 1024,
    }
}

#[test]
fn tokenizer_round_trip_and_checksum_are_real() {
    let corpus = temp_path("tokenizer.txt");
    let tokenizer_path = temp_path("tokenizer.aitok");
    let corrupt_path = temp_path("tokenizer-corrupt.aitok");
    fs::write(&corpus, "hello hello\nПривет мир\nhello ai\n🙂 hello\n").unwrap();

    let tokenizer = TokenizerTrainer::train(&corpus, DatasetFormat::Txt, &tokenizer_config()).unwrap();
    for text in ["hello", "Привет мир", "hello Привет", "🙂 123", ""] {
        let encoded = tokenizer.encode(text);
        assert_eq!(tokenizer.decode(&encoded).unwrap(), text);
    }
    let id_before = tokenizer.tokenizer_id();
    tokenizer.save(&tokenizer_path).unwrap();
    let loaded = Tokenizer::load(&tokenizer_path).unwrap();

    assert_eq!(id_before, loaded.tokenizer_id());
    assert_eq!(tokenizer.encode("Українська + English 🙂"), loaded.encode("Українська + English 🙂"));
    assert_eq!(tokenizer.decode(&loaded.encode("Українська + English 🙂")).unwrap(), "Українська + English 🙂");

    let mut bytes = fs::read(&tokenizer_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(&corrupt_path, bytes).unwrap();
    assert!(Tokenizer::load(&corrupt_path).is_err());

    let _ = fs::remove_file(corpus);
    let _ = fs::remove_file(tokenizer_path);
    let _ = fs::remove_file(corrupt_path);
}

#[test]
fn dataset_cursor_resumes_exact_sequence() {
    let corpus = temp_path("cursor.txt");
    fs::write(&corpus, "abcdefghijklmno\npqrstuvwxyz0123456789\n").unwrap();
    let tokenizer = Tokenizer::new_default();
    let metadata = dataset_metadata(&corpus, DatasetFormat::Txt).unwrap();

    let reader = DatasetReader::open(&corpus, DatasetFormat::Txt).unwrap();
    let mut stream = TrainingStream::new(reader, tokenizer.clone(), 8, metadata.dataset_id.clone()).unwrap();
    let first = stream.next_sequence().unwrap().unwrap();
    let second = stream.next_sequence().unwrap().unwrap();

    let reader2 = DatasetReader::open(&corpus, DatasetFormat::Txt).unwrap();
    let mut resumed = TrainingStream::resume(
        reader2,
        tokenizer,
        8,
        metadata.dataset_id.clone(),
        &first.cursor_after,
    ).unwrap();
    let resumed_second = resumed.next_sequence().unwrap().unwrap();

    assert_eq!(second.input, resumed_second.input);
    assert_eq!(second.target, resumed_second.target);

    let _ = fs::remove_file(corpus);
}

#[test]
fn checkpoint_resume_matches_continuous_training() {
    let corpus = temp_path("train.txt");
    let data_root = temp_path("data-root");
    fs::create_dir_all(&data_root).unwrap();
    fs::write(
        &corpus,
        "abcdefghijklmno\npqrstuvwxyz0123456789\nABCDEFGHIJKLMNO\nPQRSTUVWXYZ012345\n",
    ).unwrap();

    let tokenizer = Tokenizer::from_parts(default_special_tokens(), Vec::new()).unwrap();
    let model_config = ModelConfig {
        architecture: "AiNet-v1.1".into(),
        model_id: "phase2-checkpoint".into(),
        vocab_size: tokenizer.vocab_size(),
        embedding_dim: 8,
        hidden_dim: 8,
        layer_count: 1,
        sequence_length: 8,
        seed: 987654,
    };
    let mut config = TrainingConfig::low_end();
    config.epochs = 3;
    config.sequence_length = 8;
    config.gradient_accumulation = 1;
    config.checkpoint_interval_steps = 1000;
    config.max_steps = Some(2);
    config.memory_budget_mb = 512;

    let mut continuous = Trainer::new(
        AiNet::new(model_config.clone()).unwrap(),
        tokenizer.clone(),
        &corpus,
        DatasetFormat::Txt,
        config.clone(),
        &data_root,
    ).unwrap();
    let (_command_tx, command_rx) = mpsc::channel();
    let (event_tx, _event_rx) = mpsc::channel();
    continuous.run(&command_rx, None, &event_tx);
    assert_eq!(continuous.state.step, 2);

    let paused_trainer = Trainer::new(
        AiNet::new(model_config).unwrap(),
        tokenizer.clone(),
        &corpus,
        DatasetFormat::Txt,
        config,
        &data_root,
    ).unwrap();
    let mut worker = TrainingWorker::spawn(paused_trainer, None);
    worker.send(TrainingCommand::Start).unwrap();

    let mut checkpoint = None;
    loop {
        match worker.events.recv().unwrap() {
            TrainingEvent::Step(progress) if progress.step == 1 => {
                worker.send(TrainingCommand::Pause).unwrap();
            }
            TrainingEvent::CheckpointSaved(path) => checkpoint = Some(path),
            TrainingEvent::Paused => break,
            TrainingEvent::Failed(error) => panic!("training failed: {error}"),
            _ => {}
        }
    }
    worker.join();
    let checkpoint = checkpoint.expect("pause must save a checkpoint");

    let mut resumed = Trainer::from_checkpoint(
        &checkpoint,
        tokenizer,
        &corpus,
        DatasetFormat::Txt,
        &data_root,
    ).unwrap();
    resumed.run(&command_rx, None, &event_tx);
    assert_eq!(resumed.state.step, 2);

    let continuous_params: Vec<Vec<f32>> = continuous.model.parameters().into_iter().map(|p| p.data.clone()).collect();
    let resumed_params: Vec<Vec<f32>> = resumed.model.parameters().into_iter().map(|p| p.data.clone()).collect();
    assert_eq!(continuous.model.parameter_names(), resumed.model.parameter_names());
    assert_eq!(continuous_params, resumed_params);
    assert_eq!(continuous.optimizer.step_count(), resumed.optimizer.step_count());

    let _ = fs::remove_file(corpus);
    let _ = fs::remove_dir_all(data_root);
}
