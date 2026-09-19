use super::checkpoint::Checkpoint;
use super::config::TrainingConfig;
use super::events::{TrainingCommand, TrainingEvent, TrainingProgress};
use super::state::{TrainingState, TrainingStatus};
use crate::dataset::{
    dataset_metadata, DatasetCursor, DatasetFormat, DatasetReader, TrainingStream,
};
use crate::model::AiNet;
use crate::optimizer::AdamW;
use crate::tokenizer::Tokenizer;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn next_run_id(data_root: &Path) -> Result<String, String> {
    let runs = data_root.join("runs");
    fs::create_dir_all(&runs).map_err(|e| format!("create runs directory: {e}"))?;
    let mut max_id = 0u64;
    for entry in fs::read_dir(&runs).map_err(|e| format!("read runs directory: {e}"))? {
        let name = entry
            .map_err(|e| format!("read run entry: {e}"))?
            .file_name()
            .to_string_lossy()
            .into_owned();
        if let Some(value) = name
            .strip_prefix("run-")
            .and_then(|v| v.parse::<u64>().ok())
        {
            max_id = max_id.max(value);
        }
    }
    Ok(format!("run-{max_id:06}", max_id = max_id + 1))
}

pub struct Trainer {
    pub model: AiNet,
    pub tokenizer: Tokenizer,
    pub optimizer: AdamW,
    pub dataset_path: PathBuf,
    pub dataset_format: DatasetFormat,
    pub dataset_id: String,
    pub tokenizer_id: String,
    pub config: TrainingConfig,
    pub state: TrainingState,
    pub run_dir: PathBuf,
    pub memory_state: Vec<Vec<f32>>,
    data_root: PathBuf,
}

impl Trainer {
    pub fn new(
        model: AiNet,
        tokenizer: Tokenizer,
        dataset_path: impl AsRef<Path>,
        dataset_format: DatasetFormat,
        config: TrainingConfig,
        data_root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        config.validate()?;
        if model.config.vocab_size != tokenizer.vocab_size() {
            return Err(format!(
                "model vocab {} != tokenizer vocab {}",
                model.config.vocab_size,
                tokenizer.vocab_size()
            ));
        }
        if config.sequence_length > model.config.sequence_length {
            return Err("training sequence length exceeds model sequence_length".into());
        }
        let dataset_path = dataset_path.as_ref().to_path_buf();
        let data_root = data_root.as_ref().to_path_buf();
        let meta = dataset_metadata(&dataset_path, dataset_format.clone())?;
        let run = next_run_id(&data_root)?;
        Self::build(
            model,
            tokenizer,
            dataset_path,
            dataset_format,
            config,
            data_root,
            run,
            meta.dataset_id,
            None,
            false,
        )
    }

    fn build(
        model: AiNet,
        tokenizer: Tokenizer,
        dataset_path: PathBuf,
        dataset_format: DatasetFormat,
        config: TrainingConfig,
        data_root: PathBuf,
        run: String,
        dataset_id: String,
        cursor: Option<DatasetCursor>,
        resuming: bool,
    ) -> Result<Self, String> {
        config.validate()?;
        let tokenizer_id = tokenizer.tokenizer_id();
        if model.config.vocab_size != tokenizer.vocab_size() {
            return Err(format!(
                "model vocab {} != tokenizer vocab {}",
                model.config.vocab_size,
                tokenizer.vocab_size()
            ));
        }
        let run_dir = data_root.join("runs").join(&run);
        fs::create_dir_all(run_dir.join("checkpoints"))
            .map_err(|e| format!("create run directory: {e}"))?;
        let reader = DatasetReader::open(&dataset_path, dataset_format.clone())?;
        let stream_cursor = cursor.clone();
        let _stream = if let Some(c) = &stream_cursor {
            TrainingStream::resume(
                reader,
                tokenizer.clone(),
                config.sequence_length,
                dataset_id.clone(),
                c,
            )?
        } else {
            TrainingStream::new(
                reader,
                tokenizer.clone(),
                config.sequence_length,
                dataset_id.clone(),
            )?
        };
        let initial_cursor = cursor.unwrap_or_else(|| DatasetCursor {
            dataset_id: dataset_id.clone(),
            file_path: dataset_path.to_string_lossy().into_owned(),
            file_offset: 0,
            sample_index: 0,
            token_position: 0,
        });
        let state = TrainingState {
            status: if resuming {
                TrainingStatus::Resuming
            } else {
                TrainingStatus::Idle
            },
            run_id: run.clone(),
            epoch: 0,
            step: 0,
            tokens_seen: 0,
            tokens_this_run: 0,
            tokens_this_epoch: 0,
            cursor: initial_cursor,
            random_state: model.config.seed ^ 0xA11CE5EED,
        };
        let estimate = model.estimated_training_bytes(config.sequence_length);
        let budget = config.memory_budget_mb.saturating_mul(1024 * 1024);
        if estimate > budget {
            return Err(format!(
                "estimated training RAM {} MB exceeds budget {} MB",
                estimate / 1024 / 1024,
                config.memory_budget_mb
            ));
        }
        let hidden_dim = model.config.hidden_dim;
        let layer_count = model.config.layer_count;

        Ok(Self {
            model,
            tokenizer,
            optimizer: AdamW::new(config.learning_rate, config.weight_decay),
            dataset_path,
            dataset_format,
            dataset_id,
            tokenizer_id,
            config,
            state,
            run_dir,
            memory_state: vec![vec![0.0; hidden_dim]; layer_count],
            data_root,
        })
    }

    pub fn from_checkpoint(
        checkpoint_path: impl AsRef<Path>,
        tokenizer: Tokenizer,
        dataset_path: impl AsRef<Path>,
        dataset_format: DatasetFormat,
        data_root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let (data, _source) = Checkpoint::load_latest_or_previous(checkpoint_path.as_ref())?;
        let current_meta = dataset_metadata(dataset_path.as_ref(), dataset_format.clone())?;
        if data.dataset_id != current_meta.dataset_id {
            return Err("checkpoint dataset_id does not match current dataset".into());
        }
        let tokenizer_id = tokenizer.tokenizer_id();
        if data.tokenizer_id != tokenizer_id {
            return Err("checkpoint tokenizer_id does not match current tokenizer".into());
        }
        let model = AiNet::from_aimodel_bytes(&data.model_bytes)?;
        if model.weights_checksum() != data.model_checksum {
            return Err("checkpoint model checksum mismatch".into());
        }
        let mut trainer = Self::build(
            model,
            tokenizer,
            dataset_path.as_ref().to_path_buf(),
            dataset_format,
            data.config.clone(),
            data_root.as_ref().to_path_buf(),
            data.run_id.clone(),
            data.dataset_id.clone(),
            Some(data.cursor.clone()),
            true,
        )?;
        trainer.state = data.state.clone();
        trainer.state.transition(TrainingStatus::Resuming)?;
        trainer.state.random_state = data.random_state;
        if data.memory_state.len() != trainer.model.config.layer_count
            || data
                .memory_state
                .iter()
                .any(|layer| layer.len() != trainer.model.config.hidden_dim)
        {
            return Err("checkpoint recurrent memory shape does not match model".into());
        }
        trainer.memory_state = data.memory_state;
        trainer
            .optimizer
            .load_state(data.optimizer, trainer.model.parameters())?;
        Ok(trainer)
    }

    pub fn run(
        &mut self,
        commands: &Receiver<TrainingCommand>,
        control_file: Option<&Path>,
        events: &Sender<TrainingEvent>,
    ) {
        if self.state.status == TrainingStatus::Resuming {
            let _ = events.send(TrainingEvent::Resumed);
        } else {
            let _ = self.state.transition(TrainingStatus::Starting);
        }
        if self.state.status == TrainingStatus::Starting {
            let _ = self.state.transition(TrainingStatus::Running);
        } else if self.state.status == TrainingStatus::Resuming {
            let _ = self.state.transition(TrainingStatus::Running);
        }
        if let Err(e) = self
            .write_run_metadata()
            .and_then(|_| self.persist_status())
        {
            let _ = events.send(TrainingEvent::Failed(e));
            return;
        }
        let _ = events.send(TrainingEvent::Started(self.progress(0.0, 0.0, 0.0, false)));

        let reader = match DatasetReader::open(&self.dataset_path, self.dataset_format.clone()) {
            Ok(v) => v,
            Err(e) => {
                let _ = events.send(TrainingEvent::Failed(e));
                return;
            }
        };
        let mut stream = match if self.state.step == 0
            && self.state.cursor.file_offset == 0
            && self.state.cursor.sample_index == 0
            && self.state.cursor.token_position == 0
        {
            TrainingStream::new(
                reader,
                self.tokenizer.clone(),
                self.config.sequence_length,
                self.dataset_id.clone(),
            )
        } else {
            TrainingStream::resume(
                reader,
                self.tokenizer.clone(),
                self.config.sequence_length,
                self.dataset_id.clone(),
                &self.state.cursor,
            )
        } {
            Ok(v) => v,
            Err(e) => {
                let _ = events.send(TrainingEvent::Failed(e));
                return;
            }
        };

        let mut accumulated_loss = 0.0f64;
        let mut accumulation_count = 0usize;
        let mut memory = self.memory_state.clone();
        let started = now_ms();

        loop {
            if self
                .config
                .max_steps
                .is_some_and(|limit| self.state.step >= limit)
            {
                break;
            }
            let (pause, stop) = poll_commands(commands, control_file);
            if stop {
                let _ = self.state.transition(TrainingStatus::Stopping);
            } else if pause {
                let _ = self.state.transition(TrainingStatus::Pausing);
            }

            if matches!(
                self.state.status,
                TrainingStatus::Pausing | TrainingStatus::Stopping
            ) && accumulation_count == 0
            {
                let target = if self.state.status == TrainingStatus::Pausing {
                    TrainingStatus::Paused
                } else {
                    TrainingStatus::Stopped
                };
                let _ = self.state.transition(TrainingStatus::Saving);
                if self.save_checkpoint(events, target, &memory).is_ok() {
                    let _ = self.state.transition(target);
                    let _ = self.persist_status();
                    let _ = events.send(if target == TrainingStatus::Paused {
                        TrainingEvent::Paused
                    } else {
                        TrainingEvent::Stopped
                    });
                }
                return;
            }

            let sequence = match stream.next_sequence() {
                Ok(Some(v)) => v,
                Ok(None) => {
                    if accumulation_count > 0 {
                        self.apply_accumulation(
                            &mut accumulation_count,
                            &mut accumulated_loss,
                            events,
                            started,
                        );
                    }
                    self.state.epoch += 1;
                    if !self.config.continuous && self.state.epoch >= self.config.epochs {
                        let _ = self.state.transition(TrainingStatus::Completed);
                        self.state.cursor = stream
                            .cursor()
                            .unwrap_or_else(|_| self.state.cursor.clone());
                        let _ = self.persist_status();
                        let _ = self.save_checkpoint(events, TrainingStatus::Completed, &memory);
                        let _ = events.send(TrainingEvent::Completed);
                        return;
                    }
                    self.state.tokens_this_epoch = 0;
                    for layer in &mut memory {
                        layer.fill(0.0);
                    }
                    let reset_reader = match DatasetReader::open(
                        &self.dataset_path,
                        self.dataset_format.clone(),
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = events.send(TrainingEvent::Failed(e));
                            return;
                        }
                    };
                    stream = match TrainingStream::new(
                        reset_reader,
                        self.tokenizer.clone(),
                        self.config.sequence_length,
                        self.dataset_id.clone(),
                    ) {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = events.send(TrainingEvent::Failed(e));
                            return;
                        }
                    };
                    self.state.cursor = match stream.cursor() {
                        Ok(c) => c,
                        Err(_) => self.state.cursor.clone(),
                    };
                    continue;
                }
                Err(e) => {
                    let _ = events.send(TrainingEvent::Failed(e));
                    return;
                }
            };

            let result = if accumulation_count == 0 {
                self.model
                    .train_step_with_state(&sequence.input, &sequence.target, Some(&memory))
            } else {
                self.model.accumulate_train_step_with_state(
                    &sequence.input,
                    &sequence.target,
                    Some(&memory),
                )
            };
            match result {
                Ok((loss, next_memory)) => {
                    memory = next_memory;
                    accumulated_loss += loss as f64;
                    accumulation_count += 1;
                    self.state.tokens_seen += sequence.input.len() as u64;
                    self.state.tokens_this_run += sequence.input.len() as u64;
                    self.state.tokens_this_epoch += sequence.input.len() as u64;
                    self.state.cursor = sequence.cursor_after;
                }
                Err(e) => {
                    let _ = events.send(TrainingEvent::Failed(e));
                    return;
                }
            }

            if accumulation_count >= self.config.gradient_accumulation
                || matches!(
                    self.state.status,
                    TrainingStatus::Pausing | TrainingStatus::Stopping
                )
            {
                self.apply_accumulation(
                    &mut accumulation_count,
                    &mut accumulated_loss,
                    events,
                    started,
                );
            }

            if self.state.status == TrainingStatus::Pausing
                || self.state.status == TrainingStatus::Stopping
            {
                let target = if self.state.status == TrainingStatus::Pausing {
                    TrainingStatus::Paused
                } else {
                    TrainingStatus::Stopped
                };
                let _ = self.state.transition(TrainingStatus::Saving);
                if self.save_checkpoint(events, target, &memory).is_ok() {
                    let _ = self.state.transition(target);
                    let _ = self.persist_status();
                    let _ = events.send(if target == TrainingStatus::Paused {
                        TrainingEvent::Paused
                    } else {
                        TrainingEvent::Stopped
                    });
                }
                return;
            }

            if self.state.step > 0 && self.state.step % self.config.checkpoint_interval_steps == 0 {
                let _ = self.state.transition(TrainingStatus::Saving);
                if self
                    .save_checkpoint(events, TrainingStatus::Running, &memory)
                    .is_err()
                {
                    self.state.status = TrainingStatus::Failed;
                    let _ = self.persist_status();
                    return;
                }
                let _ = self.state.transition(TrainingStatus::Running);
                let _ = self.persist_status();
            }
        }

        let _ = self.state.transition(TrainingStatus::Completed);
        let _ = self.persist_status();
        let _ = self.save_checkpoint(events, TrainingStatus::Completed, &memory);
        let _ = events.send(TrainingEvent::Completed);
    }

    fn apply_accumulation(
        &mut self,
        accumulation_count: &mut usize,
        accumulated_loss: &mut f64,
        events: &Sender<TrainingEvent>,
        started: u64,
    ) {
        if *accumulation_count == 0 {
            return;
        }
        let scale = 1.0 / *accumulation_count as f32;
        self.model.scale_gradients(scale);
        let (norm, clipped) = self.model.clip_grad_norm(self.config.max_grad_norm);
        let loss = (*accumulated_loss / *accumulation_count as f64) as f32;
        self.optimizer.step(self.model.parameters_mut());
        self.state.step += 1;
        let elapsed = ((now_ms().saturating_sub(started)).max(1) as f64) / 1000.0;
        let tps = self.state.tokens_this_run as f64 / elapsed;
        let progress = self.progress(loss, tps, norm, clipped);
        let _ = append_metrics(&self.run_dir.join("metrics.jsonl"), &progress);
        let _ = events.send(TrainingEvent::Step(progress));
        *accumulation_count = 0;
        *accumulated_loss = 0.0;
        let _ = self.persist_status();
    }

    fn progress(
        &mut self,
        loss: f32,
        tps: f64,
        gradient_norm: f32,
        clipped: bool,
    ) -> TrainingProgress {
        TrainingProgress {
            run_id: self.state.run_id.clone(),
            status: self.state.status,
            epoch: self.state.epoch,
            step: self.state.step,
            tokens_seen: self.state.tokens_seen,
            tokens_this_run: self.state.tokens_this_run,
            tokens_this_epoch: self.state.tokens_this_epoch,
            loss,
            tokens_per_second: tps,
            gradient_norm,
            clipped,
            timestamp_unix_ms: now_ms(),
        }
    }

    fn save_checkpoint(
        &mut self,
        events: &Sender<TrainingEvent>,
        status_for_checkpoint: TrainingStatus,
        memory: &[Vec<f32>],
    ) -> Result<(), String> {
        let mut state = self.state.clone();
        state.status = status_for_checkpoint;
        let optimizer_state = self.optimizer.export_state(self.model.parameters())?;
        let latest = self
            .run_dir
            .join("checkpoints/checkpoint-latest.aicheckpoint");
        Checkpoint::save_latest(
            &latest,
            &self.state.run_id,
            &self.model,
            &optimizer_state,
            &self.dataset_id,
            &self.tokenizer_id,
            &state,
            &self.config,
            self.state.random_state,
            memory,
            now_ms(),
        )?;
        let numbered = self.run_dir.join(format!(
            "checkpoints/checkpoint-{:09}.aicheckpoint",
            self.state.step
        ));
        let bytes = fs::read(&latest).map_err(|e| format!("read latest checkpoint: {e}"))?;
        fs::write(&numbered, bytes).map_err(|e| format!("write numbered checkpoint: {e}"))?;
        self.memory_state = memory.to_vec();
        let _ = events.send(TrainingEvent::CheckpointSaved(latest));
        Ok(())
    }

    fn write_run_metadata(&self) -> Result<(), String> {
        fs::create_dir_all(&self.run_dir).map_err(|e| format!("create run dir: {e}"))?;
        let config = serde_json::to_vec_pretty(&self.config).map_err(|e| e.to_string())?;
        fs::write(self.run_dir.join("config.json"), config)
            .map_err(|e| format!("write config: {e}"))?;
        let dataset = serde_json::json!({
            "dataset_id": self.dataset_id,
            "path": self.dataset_path,
            "format": self.dataset_format,
            "tokenizer_id": self.tokenizer_id,
            "model_id": self.model.config.model_id,
            "architecture": self.model.config.architecture,
            "parameter_count": self.model.parameter_count(),
            "estimated_training_bytes": self.model.estimated_training_bytes(self.config.sequence_length),
        });
        let bytes = serde_json::to_vec_pretty(&dataset).map_err(|e| e.to_string())?;
        fs::write(self.run_dir.join("metadata.json"), bytes)
            .map_err(|e| format!("write metadata: {e}"))?;
        Ok(())
    }

    fn persist_status(&self) -> Result<(), String> {
        let json = serde_json::to_vec_pretty(&self.state).map_err(|e| e.to_string())?;
        fs::write(self.run_dir.join("state.json"), &json)
            .map_err(|e| format!("write run state: {e}"))?;
        fs::create_dir_all(&self.data_root).map_err(|e| format!("create data root: {e}"))?;
        fs::write(self.data_root.join("training-status.json"), json)
            .map_err(|e| format!("write global training status: {e}"))?;
        Ok(())
    }
}

fn poll_commands(
    commands: &Receiver<TrainingCommand>,
    control_file: Option<&Path>,
) -> (bool, bool) {
    let mut pause = false;
    let mut stop = false;
    while let Ok(command) = commands.try_recv() {
        match command {
            TrainingCommand::Pause => pause = true,
            TrainingCommand::Stop => stop = true,
            TrainingCommand::Start | TrainingCommand::Resume => {}
        }
    }
    if let Some(path) = control_file {
        if let Ok(content) = fs::read_to_string(path) {
            match content.trim().to_ascii_lowercase().as_str() {
                "pause" => pause = true,
                "stop" => stop = true,
                _ => {}
            }
            if pause || stop {
                let _ = fs::remove_file(path);
            }
        }
    }
    (pause, stop)
}

fn append_metrics(path: &Path, p: &TrainingProgress) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create metrics dir: {e}"))?;
    }
    let line = serde_json::json!({
        "timestamp_unix_ms": p.timestamp_unix_ms,
        "run_id": p.run_id,
        "state": format!("{:?}", p.status),
        "step": p.step,
        "epoch": p.epoch,
        "loss": p.loss,
        "tokens_seen": p.tokens_seen,
        "tokens_per_second": p.tokens_per_second,
        "gradient_norm": p.gradient_norm,
        "clipped": p.clipped
    });
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("open metrics: {e}"))?;
    writeln!(file, "{line}").map_err(|e| format!("append metrics: {e}"))
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
