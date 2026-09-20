use super::config::{AppConfig, TrainingUiConfig};
use super::logging::{CrashContext, Logger, RuntimeGuard};
use super::resources::{ResourceMonitor, ResourceSnapshot};
use crate::architecture::AiNet;
use crate::dataset::{dataset_metadata, validate, DatasetFormat, ValidationReport};
use crate::inference::{GenerationConfig, InferenceEngine};
use crate::tokenizer::{Tokenizer, TokenizerTrainer, TokenizerTrainerConfig};
use crate::training::{Checkpoint, Trainer, TrainingCommand, TrainingConfig, TrainingEvent, TrainingProgress, TrainingStatus, TrainingWorker};
use crate::neural::Tensor;
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppPage {
    Home,
    Chat,
    Train,
    Model,
    Dataset,
    Memory,
    Evaluation,
    Logs,
    Settings,
    System,
}

impl AppPage {
    pub const ALL: [AppPage; 10] = [
        Self::Home, Self::Chat, Self::Train, Self::Model, Self::Dataset,
        Self::Memory, Self::Evaluation, Self::Logs, Self::Settings, Self::System,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Home => "HOME",
            Self::Chat => "CHAT",
            Self::Train => "TRAIN",
            Self::Model => "MODEL",
            Self::Dataset => "DATASET",
            Self::Memory => "MEMORY",
            Self::Evaluation => "EVALUATION",
            Self::Logs => "LOGS",
            Self::Settings => "SETTINGS",
            Self::System => "SYSTEM",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub config: Option<crate::model::ModelConfig>,
    pub parameter_count: usize,
    pub checksum: u64,
    pub file_size: u64,
    pub loaded_from_previous: bool,
}

impl ModelInfo {
    pub fn status(&self) -> &'static str {
        "LOADED"
    }
}

#[derive(Clone, Debug)]
pub struct TrainingSnapshot {
    pub state: TrainingStatus,
    pub run_id: String,
    pub epoch: u64,
    pub step: u64,
    pub tokens: u64,
    pub loss: Option<f32>,
    pub avg_loss: Option<f32>,
    pub best_loss: Option<f32>,
    pub tokens_per_second: f64,
    pub gradient_norm: f32,
    pub learning_rate: f32,
    pub elapsed_seconds: f64,
    pub started_at_ms: Option<u64>,
    pub checkpoint: Option<PathBuf>,
    pub initial_checksum: Option<u64>,
    pub final_checksum: Option<u64>,
}

impl Default for TrainingSnapshot {
    fn default() -> Self {
        Self {
            state: TrainingStatus::Idle,
            run_id: String::new(),
            epoch: 0,
            step: 0,
            tokens: 0,
            loss: None,
            avg_loss: None,
            best_loss: None,
            tokens_per_second: 0.0,
            gradient_norm: 0.0,
            learning_rate: 0.0,
            elapsed_seconds: 0.0,
            started_at_ms: None,
            checkpoint: None,
            initial_checksum: None,
            final_checksum: None,
        }
    }
}

impl TrainingSnapshot {
    pub fn label(&self) -> &'static str {
        match self.state {
            TrainingStatus::Idle => "IDLE",
            TrainingStatus::Starting => "STARTING",
            TrainingStatus::Running => "RUNNING",
            TrainingStatus::Pausing => "PAUSING",
            TrainingStatus::Paused => "PAUSED",
            TrainingStatus::Resuming => "RESUMING",
            TrainingStatus::Saving => "SAVING",
            TrainingStatus::Completed => "COMPLETED",
            TrainingStatus::Stopping => "STOPPING",
            TrainingStatus::Stopped => "STOPPED",
            TrainingStatus::Failed => "FAILED",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelStatsSnapshot {
    pub parameter_count: usize,
    pub checksum: u64,
    pub gradient_norm: f32,
    pub weight_norm: f32,
    pub updated_parameters: usize,
    pub average_update: f32,
    pub max_update: f32,
    pub layers: Vec<LayerStats>,
}

#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    pub layer: usize,
    pub weight_norm: f32,
    pub gradient_norm: f32,
    pub memory_norm: f32,
}

#[derive(Clone, Debug)]
pub struct DatasetInfo {
    pub path: PathBuf,
    pub format: DatasetFormat,
    pub metadata_id: Option<String>,
    pub report: Option<ValidationReport>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role: &'static str,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct SelfTestResult {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

pub struct AppCore {
    pub root: PathBuf,
    pub config: AppConfig,
    pub logger: Logger,
    pub crash_context: Arc<Mutex<CrashContext>>,
    pub runtime: RuntimeGuard,
    pub resources: ResourceSnapshot,
    pub resource_monitor: ResourceMonitor,
    pub page: AppPage,
    pub previous_crash: bool,
    pub safe_mode: bool,
    pub recovery_session: Option<TrainingRecovery>,
    pub model: Option<ModelInfo>,
    pub tokenizer_path: Option<PathBuf>,
    pub dataset: Option<DatasetInfo>,
    pub training: TrainingSnapshot,
    pub model_stats: ModelStatsSnapshot,
    pub loss_points: VecDeque<(u64, f32)>,
    pub events: VecDeque<String>,
    pub worker: Option<TrainingWorker>,
    pub last_error: Option<String>,
    pub self_test_rx: Option<Receiver<Vec<SelfTestResult>>>,
    pub self_test_result: Option<Vec<SelfTestResult>>,
    pub tokenizer_job_rx: Option<Receiver<Result<PathBuf, String>>>,
    pub chat_rx: Option<Receiver<ChatEvent>>,
    pub chat_stop: Option<Arc<AtomicBool>>,
    pub chat_messages: Vec<ChatMessage>,
    pub chat_input: String,
    pub chat_generated: String,
    pub chat_generating: bool,
    pub log_channel: String,
    pub wizard_step: usize,
    pub wizard_open: bool,
    last_resource_poll: Instant,
}

#[derive(Clone, Debug)]
pub struct TrainingRecovery {
    pub run_id: String,
    pub step: u64,
    pub epoch: u64,
    pub checkpoint: PathBuf,
    pub model_id: String,
    pub dataset_id: String,
}

enum ChatEvent {
    Token(Vec<u32>),
    Finished(Result<(), String>),
}

impl AppCore {
    pub fn bootstrap(
        root: impl AsRef<Path>,
        logger: Logger,
        crash_context: Arc<Mutex<CrashContext>>,
    ) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| format!("storage initialization: {e}"))?;
        let (runtime, previous_crash) = RuntimeGuard::acquire(&root)?;
        let (config, config_recovered) = AppConfig::load(&root)?;
        if config_recovered {
            logger.app("config.toml was invalid; preserved broken config and created defaults");
        }
        logger.app("application bootstrap started");

        let page = parse_page(&config.selected_page);
        let safe_mode = config.safe_mode;
        let resource_monitor = ResourceMonitor::start(Duration::from_millis(config.ui_update_ms.max(500)));
        let mut core = Self {
            root: root.clone(),
            config,
            logger,
            crash_context,
            runtime,
            resources: ResourceSnapshot::default(),
            resource_monitor,
            page,
            previous_crash,
            safe_mode,
            recovery_session: None,
            model: None,
            tokenizer_path: None,
            dataset: None,
            training: TrainingSnapshot::default(),
            model_stats: ModelStatsSnapshot::default(),
            loss_points: VecDeque::new(),
            events: VecDeque::new(),
            worker: None,
            last_error: None,
            self_test_rx: None,
            self_test_result: None,
            tokenizer_job_rx: None,
            chat_rx: None,
            chat_stop: None,
            chat_messages: Vec::new(),
            chat_input: String::new(),
            chat_generated: String::new(),
            chat_generating: false,
            log_channel: "APP LOG".into(),
            wizard_step: 0,
            wizard_open: false,
            last_resource_poll: Instant::now(),
        };
        if core.previous_crash {
            core.events.push_front("Previous crash detected.".into());
        }
        if core.config.first_start {
            core.wizard_open = true;
        }
        core.refresh_resources();
        if !core.previous_crash {
            core.refresh_model();
        }
        core.refresh_tokenizer();
        core.refresh_dataset();
        core.detect_training_recovery();
        core.update_crash_context();
        Ok(core)
    }

    pub fn command_page(&mut self, page: AppPage) {
        self.page = page;
        self.config.selected_page = page.name().to_string().to_ascii_lowercase();
    }

    pub fn is_trainer_running(&self) -> bool {
        matches!(
            self.training.state,
            TrainingStatus::Starting | TrainingStatus::Running | TrainingStatus::Pausing | TrainingStatus::Resuming | TrainingStatus::Saving
        )
    }

    pub fn refresh(&mut self) {
        self.refresh_resources();
        self.poll_worker();
        self.poll_tokenizer_job();
        self.poll_chat();
        self.update_crash_context();
        if self.last_resource_poll.elapsed() > Duration::from_secs(2) && self.model.is_none() {
            self.refresh_model();
        }
    }

    pub fn refresh_resources(&mut self) {
        while let Ok(snapshot) = self.resource_monitor.events.try_recv() {
            self.resources = snapshot;
            self.last_resource_poll = Instant::now();
        }
    }

    pub fn refresh_model(&mut self) {
        let Some(path_string) = self.config.model_path.clone() else {
            self.model = None;
            return;
        };
        let path = PathBuf::from(path_string);
        match load_model_info(&path) {
            Ok(info) => {
                self.model = Some(info);
                self.logger.app(format!("model loaded: {}", path.display()));
            }
            Err(error) => {
                let previous = previous_path(&path);
                match load_model_info(&previous) {
                    Ok(mut info) => {
                        info.loaded_from_previous = true;
                        self.model = Some(info);
                        self.last_error = Some(format!("Primary model invalid; using previous valid model: {error}"));
                        self.logger.app(self.last_error.as_deref().unwrap_or_default());
                    }
                    Err(_) => {
                        self.model = None;
                        self.last_error = Some(format!("Model corrupted or unavailable: {error}"));
                    }
                }
            }
        }
    }

    pub fn refresh_tokenizer(&mut self) {
        self.tokenizer_path = self
            .config
            .tokenizer_path
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.exists());
    }

    pub fn refresh_dataset(&mut self) {
        let Some(path) = self.config.datasets.first().map(PathBuf::from) else {
            self.dataset = None;
            return;
        };
        let format = DatasetFormat::from_path(&path);
        let info = DatasetInfo { path, format, metadata_id: None, report: None, error: None };
        self.dataset = Some(info);
    }

    pub fn pick_dataset(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Dataset", &["txt", "jsonl"])
            .pick_file()
        else { return };
        self.config.datasets = vec![path.display().to_string()];
        self.dataset = Some(DatasetInfo {
            format: DatasetFormat::from_path(&path),
            path,
            metadata_id: None,
            report: None,
            error: None,
        });
        self.save_config();
        self.log_event(format!("Dataset selected: {}", self.config.datasets[0]));
    }

    pub fn validate_dataset(&mut self) {
        let Some(dataset) = self.dataset.clone() else {
            self.last_error = Some("No dataset selected.".into());
            return;
        };
        let tokenizer = self.tokenizer_path.as_ref().and_then(|p| Tokenizer::load(p).ok());
        match validate(&dataset.path, dataset.format.clone(), tokenizer.as_ref()) {
            Ok(report) => {
                let metadata = dataset_metadata(&dataset.path, dataset.format.clone()).ok();
                if let Some(current) = self.dataset.as_mut() {
                    current.report = Some(report);
                    current.metadata_id = metadata.map(|m| m.dataset_id);
                    current.error = None;
                }
                self.log_event("Dataset validation passed.");
            }
            Err(error) => {
                if let Some(current) = self.dataset.as_mut() {
                    current.error = Some(error.clone());
                }
                self.last_error = Some(error);
            }
        }
    }

    pub fn train_tokenizer(&mut self) {
        let Some(dataset) = self.dataset.clone() else {
            self.last_error = Some("Select a dataset first.".into());
            return;
        };
        if self.tokenizer_job_rx.is_some() {
            return;
        }
        let output = self.root.join("tokenizer").join("tokenizer.aitok");
        let config = TokenizerTrainerConfig {
            target_vocab_size: 4096,
            min_frequency: 2,
            max_merges: 3840,
            sample_bytes: 8 * 1024 * 1024,
        };
        let log = self.logger.clone();
        let (tx, rx) = mpsc::channel();
        self.tokenizer_job_rx = Some(rx);
        thread::spawn(move || {
            let result = (|| -> Result<PathBuf, String> {
                let tokenizer = TokenizerTrainer::train(&dataset.path, dataset.format, &config)?;
                tokenizer.save(&output)?;
                Ok(output)
            })();
            if let Err(error) = &result {
                log.app(format!("tokenizer job failed: {error}"));
            }
            let _ = tx.send(result);
        });
        self.log_event("Tokenizer training started.");
    }

    pub fn create_model(&mut self, name: &str, vocab: usize, embedding: usize, hidden: usize, layers: usize, sequence: usize, seed: u64) {
        if name.trim().is_empty() {
            self.last_error = Some("Model name must not be empty.".into());
            return;
        }
        let config = crate::model::ModelConfig {
            architecture: "AiNet-v1.1".into(),
            model_id: name.trim().into(),
            vocab_size: vocab,
            embedding_dim: embedding,
            hidden_dim: hidden,
            layer_count: layers,
            sequence_length: sequence,
            seed,
        };
        match AiNet::new(config.clone()).and_then(|model| {
            let path = self.root.join("models").join(format!("{}.aimodel", sanitize_file_name(name)));
            fs::create_dir_all(path.parent().unwrap_or(&self.root)).map_err(|e| format!("create model directory: {e}"))?;
            model.save(&path)?;
            Ok(path)
        }) {
            Ok(path) => {
                self.config.model_path = Some(path.display().to_string());
                self.save_config();
                self.refresh_model();
                self.log_event(format!("Created random model {}.", name.trim()));
            }
            Err(error) => self.last_error = Some(error),
        }
    }

    pub fn load_model_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new().add_filter("Ai model", &["aimodel"]).pick_file() else { return };
        self.config.model_path = Some(path.display().to_string());
        self.save_config();
        self.refresh_model();
    }

    pub fn start_training(&mut self) {
        if self.worker.is_some() || self.is_trainer_running() {
            return;
        }
        let Some(model_path) = self.model.as_ref().map(|m| m.path.clone()) else {
            self.last_error = Some("Create or load a model first.".into());
            return;
        };
        let Some(tokenizer_path) = self.tokenizer_path.clone() else {
            self.last_error = Some("Train a tokenizer or select an .aitok first.".into());
            return;
        };
        let Some(dataset) = self.dataset.clone() else {
            self.last_error = Some("Select a dataset first.".into());
            return;
        };
        let tokenizer = match Tokenizer::load(&tokenizer_path) {
            Ok(v) => v,
            Err(e) => { self.last_error = Some(e); return; }
        };
        let model = match AiNet::load(&model_path) {
            Ok(v) => v,
            Err(e) => { self.last_error = Some(e); return; }
        };
        let config = self.training_config();
        if let Err(error) = config.validate() {
            self.last_error = Some(error);
            return;
        }
        if let Some(report) = &self.dataset.as_ref().and_then(|d| d.report.clone()) {
            if report.errors != 0 {
                self.last_error = Some("Dataset validation failed; training blocked.".into());
                return;
            }
        }
        let checksum = model.weights_checksum();
        let trainer = match Trainer::new(model, tokenizer, dataset.path.clone(), dataset.format.clone(), config, &self.root) {
            Ok(v) => v,
            Err(e) => { self.last_error = Some(e); return; }
        };
        let mut worker = TrainingWorker::spawn(trainer, Some(self.root.join("training.command")));
        if let Err(e) = worker.send(TrainingCommand::Start) {
            self.last_error = Some(e);
            return;
        }
        self.training = TrainingSnapshot {
            state: TrainingStatus::Starting,
            initial_checksum: Some(checksum),
            learning_rate: self.config.training.learning_rate,
            started_at_ms: Some(now_ms()),
            ..Default::default()
        };
        self.loss_points.clear();
        self.model_stats = ModelStatsSnapshot::default();
        self.worker = Some(worker);
        self.log_event("Training start requested.");
        self.logger.training("Training start requested");
    }

    pub fn pause_training(&mut self) {
        if let Some(worker) = &self.worker {
            let _ = worker.send(TrainingCommand::Pause);
            self.training.state = TrainingStatus::Pausing;
            self.log_event("Pause requested.");
            self.logger.training("Pause requested");
        }
    }

    pub fn stop_training(&mut self) {
        if let Some(worker) = &self.worker {
            let _ = worker.send(TrainingCommand::Stop);
            self.training.state = TrainingStatus::Stopping;
            self.log_event("Stop requested.");
            self.logger.training("Stop requested");
        }
    }

    pub fn resume_training(&mut self) {
        if self.worker.is_some() {
            return;
        }
        let Some(recovery) = self.recovery_session.clone().or_else(|| {
            self.training.checkpoint.clone().map(|checkpoint| TrainingRecovery {
                run_id: self.training.run_id.clone(),
                step: self.training.step,
                epoch: self.training.epoch,
                checkpoint,
                model_id: self.model.as_ref()?.config.as_ref()?.model_id.clone(),
                dataset_id: self.dataset.as_ref()?.metadata_id.clone().unwrap_or_default(),
            })
        }) else {
            self.last_error = Some("No resumable checkpoint found.".into());
            return;
        };
        let Some(tokenizer_path) = self.tokenizer_path.clone() else {
            self.last_error = Some("Tokenizer is required for resume.".into());
            return;
        };
        let Some(dataset) = self.dataset.clone() else {
            self.last_error = Some("Dataset is required for resume.".into());
            return;
        };
        let tokenizer = match Tokenizer::load(tokenizer_path) {
            Ok(v) => v,
            Err(e) => { self.last_error = Some(e); return; }
        };
        let trainer = match Trainer::from_checkpoint(&recovery.checkpoint, tokenizer, dataset.path.clone(), dataset.format.clone(), &self.root) {
            Ok(v) => v,
            Err(e) => { self.last_error = Some(e); return; }
        };
        let mut worker = TrainingWorker::spawn(trainer, Some(self.root.join("training.command")));
        if let Err(e) = worker.send(TrainingCommand::Resume) {
            self.last_error = Some(e);
            return;
        }
        self.training.state = TrainingStatus::Resuming;
        self.training.checkpoint = Some(recovery.checkpoint.clone());
        self.training.step = recovery.step;
        self.training.epoch = recovery.epoch;
        self.worker = Some(worker);
        self.log_event(format!("Resuming from checkpoint step {}.", recovery.step));
        self.logger.training(format!("Resuming from checkpoint {}", recovery.checkpoint.display()));
        self.recovery_session = None;
    }

    pub fn run_self_test(&mut self) {
        if self.self_test_rx.is_some() {
            return;
        }
        let (tx, rx) = mpsc::channel();
        self.self_test_rx = Some(rx);
        let root = self.root.clone();
        let logger = self.logger.clone();
        thread::spawn(move || {
            let results = run_self_tests(&root);
            logger.app(format!("self-test completed: {} checks", results.len()));
            let _ = tx.send(results);
        });
    }

    pub fn export_training_report(&mut self) {
        let report = serde_json::json!({
            "application": "Ai",
            "version": env!("CARGO_PKG_VERSION"),
            "architecture": "AiNet v1.1",
            "model": self.model.as_ref().map(|m| serde_json::json!({
                "path": m.path,
                "parameter_count": m.parameter_count,
                "checksum": format!("{:016x}", m.checksum),
                "file_size": m.file_size,
                "model_id": m.config.as_ref().map(|c| c.model_id.clone()),
                "architecture": m.config.as_ref().map(|c| c.architecture.clone()),
            }));
            "dataset": self.dataset.as_ref().map(|d| serde_json::json!({
                "path": d.path,
                "format": format!("{:?}", d.format),
                "dataset_id": d.metadata_id,
                "samples": d.report.as_ref().map(|r| r.samples),
                "estimated_tokens": d.report.as_ref().map(|r| r.estimated_tokens),
            })),
            "training": {
                "run_id": self.training.run_id,
                "state": self.training.label(),
                "epoch": self.training.epoch,
                "step": self.training.step,
                "tokens": self.training.tokens,
                "loss": self.training.loss,
                "avg_loss": self.training.avg_loss,
                "best_loss": self.training.best_loss,
                "initial_checksum": self.training.initial_checksum,
                "final_checksum": self.training.final_checksum,
                "checkpoint": self.training.checkpoint,
            }
        });
        match serde_json::to_string_pretty(&report).and_then(|v| {
            let path = self.root.join("reports").join(format!("training-{}.json", now_ms()));
            fs::create_dir_all(path.parent().unwrap()).map_err(serde_json::Error::io)?;
            fs::write(&path, v).map_err(serde_json::Error::io)?;
            Ok(path)
        }) {
            Ok(path) => self.log_event(format!("Training report exported: {}", path.display())),
            Err(error) => self.last_error = Some(error.to_string()),
        }
    }

    pub fn generate_chat(&mut self, generation: GenerationConfig, max_tokens: usize, seed: u64) {
        if self.chat_generating {
            return;
        }
        let Some(model_path) = self.model.as_ref().map(|m| m.path.clone()) else {
            self.last_error = Some("Chat unavailable: model is not loaded.".into());
            return;
        };
        let Some(tokenizer_path) = self.tokenizer_path.clone() else {
            self.last_error = Some("Chat unavailable: tokenizer is not loaded.".into());
            return;
        };
        let prompt = self.chat_input.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        self.chat_messages.push(ChatMessage { role: "User", text: prompt.clone() });
        self.chat_input.clear();
        self.chat_generated.clear();
        self.chat_generating = true;
        let stop = Arc::new(AtomicBool::new(false));
        self.chat_stop = Some(Arc::clone(&stop));
        let (tx, rx) = mpsc::channel();
        self.chat_rx = Some(rx);
        let log = self.logger.clone();
        thread::spawn(move || {
            let result = (|| -> Result<(), String> {
                let tokenizer = Tokenizer::load(&tokenizer_path)?;
                let mut engine = InferenceEngine::new(seed);
                engine.load(&model_path)?;
                engine.generate_stream(&tokenizer.encode(&prompt), tokenizer.eos_id(), max_tokens, &generation, |token| {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    let _ = tx.send(ChatEvent::Token(vec![token]));
                })?;
                tx.send(ChatEvent::Finished(Ok(()))).map_err(|e| e.to_string())?;
                Ok(())
            })();
            if let Err(error) = result {
                let _ = tx.send(ChatEvent::Finished(Err(error)));
            }
            log.inference("generation worker finished");
        });
        self.logger.inference("generation started");
    }

    pub fn stop_chat(&mut self) {
        if let Some(stop) = &self.chat_stop {
            stop.store(true, Ordering::Relaxed);
        }
    }

    pub fn save_config(&mut self) {
        self.config.selected_page = self.page.name().to_ascii_lowercase();
        if let Err(error) = self.config.save(&self.root) {
            self.last_error = Some(error);
        }
    }

    fn training_config(&self) -> TrainingConfig {
        TrainingConfig {
            epochs: self.config.training.epochs,
            sequence_length: self.config.training.sequence_length,
            micro_batch: 1,
            gradient_accumulation: self.config.training.gradient_accumulation,
            learning_rate: self.config.training.learning_rate,
            weight_decay: self.config.training.weight_decay,
            max_grad_norm: self.config.training.max_grad_norm,
            checkpoint_interval_steps: self.config.training.checkpoint_interval_steps,
            max_cpu_threads: self.config.training.max_cpu_threads,
            memory_budget_mb: self.config.training.memory_budget_mb,
            continuous: self.config.training.continuous,
            max_steps: None,
        }
    }

    fn poll_worker(&mut self) {
        let Some(worker) = &self.worker else { return };
        let mut events = Vec::new();
        while let Ok(event) = worker.events.try_recv() {
            events.push(event);
        }
        for event in events {
            match event {
                TrainingEvent::Started(progress) => self.apply_progress(progress),
                TrainingEvent::Step(progress) => self.apply_progress(progress),
                TrainingEvent::CheckpointSaved(path) => {
                    self.training.checkpoint = Some(path.clone());
                    self.logger.training(format!("Checkpoint saved: {}", path.display()));
                    self.log_event(format!("Checkpoint saved: {}", path.display()));
                }
                TrainingEvent::Paused => {
                    self.training.state = TrainingStatus::Paused;
                    self.log_event("Training paused; checkpoint is ready.");
                    self.logger.training("Training paused");
                }
                TrainingEvent::Resumed => {
                    self.training.state = TrainingStatus::Resuming;
                    self.log_event("Training resuming from checkpoint.");
                }
                TrainingEvent::Completed => {
                    self.training.state = TrainingStatus::Completed;
                    self.training.final_checksum = self.model.as_ref().map(|m| load_model_info(&m.path).ok().map(|i| i.checksum)).flatten();
                    self.log_event("Training completed.");
                    self.logger.training("Training completed");
                    self.worker = None;
                }
                TrainingEvent::Stopped => {
                    self.training.state = TrainingStatus::Stopped;
                    self.log_event("Training stopped with a checkpoint.");
                    self.worker = None;
                }
                TrainingEvent::Failed(error) => {
                    self.training.state = TrainingStatus::Failed;
                    self.last_error = Some(error.clone());
                    self.log_event(format!("Training failed: {error}"));
                    self.logger.training(format!("Training failed: {error}"));
                    self.worker = None;
                }
                }
        }
        if self.worker.as_ref().is_some_and(|w| w.is_finished()) && self.is_trainer_running() {
            self.training.state = TrainingStatus::Failed;
            self.last_error = Some("Trainer thread exited unexpectedly.".into());
            self.log_event("Trainer thread exited unexpectedly.");
            self.worker = None;
        }
    }

    fn apply_progress(&mut self, progress: TrainingProgress) {
        self.training.state = progress.status;
        self.training.run_id = progress.run_id.clone();
        self.training.epoch = progress.epoch;
        self.training.step = progress.step;
        self.training.tokens = progress.tokens_seen;
        self.training.loss = Some(progress.loss);
        self.training.tokens_per_second = progress.tokens_per_second;
        self.training.gradient_norm = progress.gradient_norm;
        self.training.learning_rate = self.config.training.learning_rate;
        let elapsed = self.training.started_at_ms.unwrap_or(progress.timestamp_unix_ms);
        self.training.elapsed_seconds = progress.timestamp_unix_ms.saturating_sub(elapsed) as f64 / 1000.0;
        let loss = progress.loss;
        self.training.avg_loss = Some(match self.training.avg_loss {
            None => loss,
            Some(previous) => previous * 0.98 + loss * 0.02,
        });
        self.training.best_loss = Some(self.training.best_loss.map_or(loss, |best| best.min(loss)));
        if loss.is_finite() {
            self.loss_points.push_back((progress.step, loss));
            while self.loss_points.len() > 1200 {
                self.loss_points.pop_front();
            }
        }
        self.log_event(format!("Step {} • loss {:.5}", progress.step, progress.loss));
    }

    fn poll_tokenizer_job(&mut self) {
        let Some(rx) = &self.tokenizer_job_rx else { return };
        if let Ok(result) = rx.try_recv() {
            self.tokenizer_job_rx = None;
            match result {
                Ok(path) => {
                    self.config.tokenizer_path = Some(path.display().to_string());
                    self.save_config();
                    self.refresh_tokenizer();
                    self.log_event(format!("Tokenizer ready: {}", path.display()));
                }
                Err(error) => self.last_error = Some(error),
            }
        }
    }

    fn poll_chat(&mut self) {
        let Some(rx) = &self.chat_rx else { return };
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        for event in events {
            match event {
                ChatEvent::Token(ids) => {
                    if let Some(tokenizer_path) = &self.tokenizer_path {
                        if let Ok(tokenizer) = Tokenizer::load(tokenizer_path) {
                            self.chat_generated = tokenizer.decode(&ids).unwrap_or_default();
                        }
                    }
                }
                ChatEvent::Finished(result) => {
                    self.chat_generating = false;
                    if let Err(error) = result {
                        self.last_error = Some(error);
                    } else {
                        self.chat_messages.push(ChatMessage { role: "Ai", text: self.chat_generated.clone() });
                    }
                    self.chat_stop = None;
                    self.chat_rx = None;
                }
            }
        }
    }

    fn detect_training_recovery(&mut self) {
        let status_path = self.root.join("training-status.json");
        let Ok(content) = fs::read_to_string(status_path) else { return };
        let Ok(state) = serde_json::from_str::<crate::training::TrainingState>(&content) else { return };
        if matches!(state.status, TrainingStatus::Running | TrainingStatus::Pausing | TrainingStatus::Saving | TrainingStatus::Resuming) {
            let candidate = self.root.join("runs").join(&state.run_id).join("checkpoints/checkpoint-latest.aicheckpoint");
            if candidate.exists() {
                let dataset_id = state.cursor.dataset_id.clone();
                self.recovery_session = Some(TrainingRecovery {
                    run_id: state.run_id.clone(),
                    step: state.step,
                    epoch: state.epoch,
                    checkpoint: candidate,
                    model_id: self.model.as_ref().and_then(|m| m.config.as_ref().map(|c| c.model_id.clone())).unwrap_or_default(),
                    dataset_id,
                });
                self.events.push_front(format!("Previous training session found at step {}.", state.step));
            }
        }
    }

    fn update_crash_context(&mut self) {
        if let Ok(mut ctx) = self.crash_context.lock() {
            ctx.application_state = format!("{:?}", self.page.name());
            ctx.training_state = self.training.label().into();
            ctx.model_id = self.model.as_ref().and_then(|m| m.config.as_ref().map(|c| c.model_id.clone())).unwrap_or_default();
            ctx.dataset_id = self.dataset.as_ref().and_then(|d| d.metadata_id.clone()).unwrap_or_default();
            ctx.last_training_step = self.training.step;
            ctx.last_checkpoint = self.training.checkpoint.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
            ctx.cpu = self.resources.cpu_name.clone();
            ctx.ram_used_mb = self.resources.ram_used_bytes / 1024 / 1024;
            ctx.ram_available_mb = self.resources.ram_available_bytes / 1024 / 1024;
        }
    }

    pub fn mark_clean_shutdown(&mut self) {
        self.save_config();
        self.runtime.mark_clean();
        self.resource_monitor.stop();
    }

    pub fn log_event(&mut self, event: impl Into<String>) {
        let value = event.into();
        self.events.push_back(value);
        while self.events.len() > 200 { self.events.pop_front(); }
    }

    pub fn format_mb(bytes: u64) -> String {
        format!("{:.1} MB", bytes as f64 / 1024.0 / 1024.0)
    }

    pub fn low_end_profile(&self) -> bool {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        self.resources.ram_available_bytes < 8 * 1024 * 1024 * 1024 || cores <= 4
    }
}

impl Drop for AppCore {
    fn drop(&mut self) {
        if let Some(worker) = &self.worker {
            let _ = worker.send(TrainingCommand::Stop);
        }
        if let Some(mut worker) = self.worker.take() {
            worker.join();
        }
        self.resource_monitor.stop();
        self.runtime.mark_clean();
    }
}

fn load_model_info(path: &Path) -> Result<ModelInfo, String> {
    let model = AiNet::load(path)?;
    let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(ModelInfo {
        path: path.to_path_buf(),
        parameter_count: model.parameter_count(),
        checksum: model.weights_checksum(),
        config: Some(model.config.clone()),
        file_size,
        loaded_from_previous: false,
    })
}

fn previous_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.prev", path.display()))
}

fn parse_page(value: &str) -> AppPage {
    let normalized = value.to_ascii_lowercase();
    AppPage::ALL.into_iter().find(|p| p.name().to_ascii_lowercase() == normalized).unwrap_or(AppPage::Home)
}

fn sanitize_file_name(value: &str) -> String {
    value.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

fn run_self_tests(root: &Path) -> Vec<SelfTestResult> {
    let mut results = Vec::new();
    let tensor = || -> Result<(), String> {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let c = a.matmul(&b)?;
        if c.data != vec![19.0, 22.0, 43.0, 50.0] { return Err("matmul mismatch".into()); }
        let p = Tensor::new(vec![1.0, 2.0, 3.0], &[3])?.softmax()?;
        if (p.sum() - 1.0).abs() > 1e-5 { return Err("softmax normalization mismatch".into()); }
        Ok(())
    };
    push_test(&mut results, "Tensor test", tensor());

    let model_test = || -> Result<(), String> {
        let tokenizer = Tokenizer::new_default();
        let model = AiNet::new(crate::model::ModelConfig {
            architecture: "AiNet-v1.1".into(), model_id: "self-test".into(),
            vocab_size: tokenizer.vocab_size(), embedding_dim: 8, hidden_dim: 8, layer_count: 2, sequence_length: 8, seed: 7,
        })?;
        let ids = tokenizer.encode("hello");
        let (loss, _) = model.sequence_loss(&ids[..4.min(ids.len())], &ids[..4.min(ids.len())], None)?;
        if !loss.is_finite() { return Err("non-finite sequence loss".into()); }
        Ok(())
    };
    push_test(&mut results, "AiNet/AiCell forward", model_test());

    let gradient = || -> Result<(), String> {
        let tokenizer = Tokenizer::new_default();
        let mut model = AiNet::new(crate::model::ModelConfig {
            architecture: "AiNet-v1.1".into(), model_id: "gradient".into(),
            vocab_size: tokenizer.vocab_size(), embedding_dim: 8, hidden_dim: 8, layer_count: 1, sequence_length: 4, seed: 11,
        })?;
        let ids = tokenizer.encode("abcd");
        let before = model.weights_checksum();
        let loss = model.train_step(&ids[..4], &ids[..4], None)?;
        if !loss.is_finite() { return Err("train loss not finite".into()); }
        let norm = model.global_gradient_norm();
        if !norm.is_finite() || norm <= 0.0 { return Err("gradient norm invalid".into()); }
        let mut opt = crate::optimizer::AdamW::new(0.001, 0.01);
        opt.step(model.parameters_mut());
        if model.weights_checksum() == before { return Err("weights did not change".into()); }
        Ok(())
    };
    push_test(&mut results, "Gradient/training mutation", gradient());

    let serialization = || -> Result<(), String> {
        let tokenizer = Tokenizer::new_default();
        let model = AiNet::new(crate::model::ModelConfig {
            architecture: "AiNet-v1.1".into(), model_id: "serialization".into(),
            vocab_size: tokenizer.vocab_size(), embedding_dim: 4, hidden_dim: 4, layer_count: 1, sequence_length: 4, seed: 9,
        })?;
        let bytes = model.to_aimodel_bytes()?;
        let restored = AiNet::from_aimodel_bytes(&bytes)?;
        if restored.weights_checksum() != model.weights_checksum() { return Err("model checksum changed after round trip".into()); }
        Ok(())
    };
    push_test(&mut results, ".aimodel serialization", serialization());

    let tokenizer_test = || -> Result<(), String> {
        let tokenizer = Tokenizer::new_default();
        let text = "Привіт, hello 🙂";
        let ids = tokenizer.encode(text);
        if tokenizer.decode(&ids)? != text { return Err("UTF-8 tokenizer round trip failed".into()); }
        Ok(())
    };
    push_test(&mut results, "Tokenizer", tokenizer_test());

    let dataset_test = || -> Result<(), String> {
        let dir = root.join("self-test");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let path = dir.join("tiny.txt");
        fs::write(&path, "hello world\nsecond sample\n").map_err(|e| e.to_string())?;
        let report = validate(&path, DatasetFormat::Txt, None)?;
        if report.errors != 0 || report.samples != 2 { return Err(format!("unexpected dataset report: {:?}", report)); }
        Ok(())
    };
    push_test(&mut results, "Dataset validation", dataset_test());

    let checkpoint = || -> Result<(), String> {
        let dir = root.join("self-test-checkpoint");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let dataset = dir.join("tiny.txt");
        fs::write(&dataset, "abcdefghi\n").map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::new_default();
        let model = AiNet::new(crate::model::ModelConfig {
            architecture: "AiNet-v1.1".into(), model_id: "checkpoint-test".into(),
            vocab_size: tokenizer.vocab_size(), embedding_dim: 4, hidden_dim: 4, layer_count: 1, sequence_length: 4, seed: 12,
        })?;
        let mut cfg = TrainingConfig::low_end();
        cfg.sequence_length = 4;
        cfg.gradient_accumulation = 1;
        cfg.checkpoint_interval_steps = 1;
        cfg.memory_budget_mb = 512;
        cfg.max_steps = Some(1);
        let trainer = Trainer::new(model, tokenizer, dataset, DatasetFormat::Txt, cfg, &dir)?;
        let mut worker = TrainingWorker::spawn(trainer, None);
        worker.send(TrainingCommand::Start)?;
        loop {
            match worker.events.recv().map_err(|e| e.to_string())? {
                TrainingEvent::CheckpointSaved(path) if path.exists() => { worker.join(); return Ok(()); }
                TrainingEvent::Completed => { worker.join(); return Err("completed without checkpoint event".into()); }
                TrainingEvent::Failed(e) => { worker.join(); return Err(e); }
                _ => {}
            }
        }
    };
    push_test(&mut results, "Checkpoint", checkpoint());

    results
}

fn push_test(results: &mut Vec<SelfTestResult>, name: &str, result: Result<(), String>) {
    match result {
        Ok(()) => results.push(SelfTestResult { name: name.into(), ok: true, detail: "PASS".into() }),
        Err(error) => results.push(SelfTestResult { name: name.into(), ok: false, detail: error }),
    }
}
