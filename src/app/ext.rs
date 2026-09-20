use super::core::{AppCore, DatasetInfo, TrainingSnapshot};
use crate::training::TrainingStatus;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

impl AppCore {
    pub fn apply_low_end_defaults_if_needed(&mut self) {
        if !self.config.performance_profile.eq_ignore_ascii_case("AUTO") {
            return;
        }

        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let available = system.available_memory();
        let low_end = logical <= 4 || available < 8 * 1024 * 1024 * 1024;
        if !low_end {
            self.config.performance_profile = "STANDARD".into();
            return;
        }

        self.config.performance_profile = "LOW-END".into();
        self.config.training.max_cpu_threads = self.config.training.max_cpu_threads.min(2).max(1);
        self.config.training.gradient_accumulation =
            self.config.training.gradient_accumulation.min(4).max(1);
        self.config.training.memory_budget_mb =
            self.config.training.memory_budget_mb.min(2048).max(512);

        if let Some(model) = &self.model {
            if let Some(config) = &model.config {
                self.config.training.sequence_length = self
                    .config
                    .training
                    .sequence_length
                    .min(64)
                    .min(config.sequence_length);
            }
        } else {
            self.config.training.sequence_length =
                self.config.training.sequence_length.min(64).max(1);
        }

        self.log_event("Automatic LOW-END performance profile enabled.");
        self.logger.app(format!(
            "LOW-END profile: logical_threads={} available_ram_mb={}",
            logical,
            available / 1024 / 1024
        ));
        self.save_config();
    }

    pub fn model_status_label(&self) -> &'static str {
        if self.model.is_none() {
            return "NOT LOADED";
        }
        match self.training.state {
            TrainingStatus::Running
            | TrainingStatus::Starting
            | TrainingStatus::Pausing
            | TrainingStatus::Resuming
            | TrainingStatus::Saving => "TRAINING",
            TrainingStatus::Paused | TrainingStatus::Stopping | TrainingStatus::Stopped => {
                "PAUSED"
            }
            TrainingStatus::Completed => "TRAINED",
            TrainingStatus::Failed => "FAILED",
            TrainingStatus::Idle => "UNTRAINED",
        }
    }

    pub fn training_eta_seconds(&self) -> Option<f64> {
        if !self.is_trainer_running() || self.training.tokens_per_second <= 0.0 {
            return None;
        }
        let report = self.dataset.as_ref()?.report.as_ref()?;
        let estimated_epoch_tokens = report.estimated_tokens? as f64;
        let total = estimated_epoch_tokens * self.config.training.epochs as f64;
        let done = self.training.tokens as f64;
        Some(((total - done).max(0.0) / self.training.tokens_per_second).max(0.0))
    }

    pub fn discard_recovery_session(&mut self) {
        self.recovery_session = None;
        let _ = fs::remove_file(self.root.join("training-status.json"));
        self.training = TrainingSnapshot::default();
        self.log_event("Previous training session discarded; checkpoints were kept.");
        self.logger.training("Previous training session discarded");
    }

    pub fn remove_dataset(&mut self) {
        self.config.datasets.clear();
        self.dataset = None;
        self.save_config();
        self.log_event("Dataset selection removed.");
    }

    pub fn clear_rotated_logs(&mut self) -> Result<(), String> {
        for stem in ["app", "training", "inference", "crash"] {
            for suffix in ["1", "2"] {
                let path = self.logger.directory().join(format!("{stem}.log.{suffix}"));
                if path.exists() {
                    fs::remove_file(&path)
                        .map_err(|e| format!("remove {}: {e}", path.display()))?;
                }
            }
        }
        self.log_event("Rotated logs cleared; active logs were preserved.");
        Ok(())
    }

    pub fn open_logs_folder(&self) {
        let _ = std::process::Command::new("explorer")
            .arg(self.logger.directory())
            .spawn();
    }

    pub fn open_run_folder(&self) {
        if self.training.run_id.is_empty() {
            return;
        }
        let path = self.root.join("runs").join(&self.training.run_id);
        let _ = std::process::Command::new("explorer").arg(path).spawn();
    }

    pub fn save_current_model_dialog(&mut self) {
        let Some(source) = self.model.as_ref().map(|m| m.path.clone()) else {
            self.last_error = Some("No model is loaded.".into());
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Ai model", ["aimodel"])
            .set_file_name(
                source
                    .file_name()
                    .and_then(|v| v.to_str())
                    .unwrap_or("model.aimodel"),
            )
            .save_file()
        else {
            return;
        };
        match crate::architecture::AiNet::load(&source).and_then(|model| model.save(&path)) {
            Ok(()) => self.log_event(format!("Model saved: {}", path.display())),
            Err(error) => self.last_error = Some(error),
        }
    }

    pub fn export_model_info(&mut self) {
        let Some(model) = self.model.clone() else {
            self.last_error = Some("No model is loaded.".into());
            return;
        };
        let path = self
            .root
            .join("reports")
            .join(format!("model-{}.json", now_ms()));
        let payload = serde_json::json!({
            "application": "Ai",
            "version": env!("CARGO_PKG_VERSION"),
            "engine": "AiNet v1.1",
            "container": "AIMDLv02-compatible",
            "status": self.model_status_label(),
            "model": {
                "path": model.path,
                "model_id": model.config.as_ref().map(|c| c.model_id.clone()),
                "architecture": model.config.as_ref().map(|c| c.architecture.clone()),
                "parameter_count": model.parameter_count,
                "checksum": format!("{:016x}", model.checksum),
                "file_size": model.file_size,
                "embedding": model.config.as_ref().map(|c| c.embedding_dim),
                "hidden": model.config.as_ref().map(|c| c.hidden_dim),
                "layers": model.config.as_ref().map(|c| c.layer_count),
                "vocabulary": model.config.as_ref().map(|c| c.vocab_size),
                "sequence": model.config.as_ref().map(|c| c.sequence_length),
                "seed": model.config.as_ref().map(|c| c.seed)
            }
        });
        let result = (|| -> Result<(), String> {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("create reports directory: {e}"))?;
            }
            let bytes = serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?;
            fs::write(&path, bytes).map_err(|e| format!("write model report: {e}"))
        })();
        match result {
            Ok(()) => self.log_event(format!("Model info exported: {}", path.display())),
            Err(error) => self.last_error = Some(error),
        }
    }

    pub fn write_training_session_summary(&mut self) {
        if self.training.run_id.is_empty() {
            return;
        }
        let run_dir = self.root.join("runs").join(&self.training.run_id);
        if let Err(error) = fs::create_dir_all(&run_dir) {
            self.last_error = Some(format!("create session directory: {error}"));
            return;
        }

        let checkpoints = fs::read_dir(run_dir.join("checkpoints"))
            .ok()
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter(|e| e.path().extension().and_then(|v| v.to_str()) == Some("aicheckpoint"))
                    .count()
            })
            .unwrap_or(0);

        let payload = serde_json::json!({
            "application": "Ai",
            "version": env!("CARGO_PKG_VERSION"),
            "run_id": self.training.run_id,
            "started_at_unix_ms": self.training.started_at_ms,
            "ended_at_unix_ms": now_ms(),
            "state": self.training.label(),
            "model": self.model.as_ref().map(|m| serde_json::json!({
                "path": m.path,
                "model_id": m.config.as_ref().map(|c| c.model_id.clone()),
                "parameter_count": m.parameter_count,
                "initial_checksum": self.training.initial_checksum,
                "final_checksum": self.training.final_checksum.or(Some(m.checksum))
            })),
            "dataset": self.dataset.as_ref().map(|d: &DatasetInfo| serde_json::json!({
                "path": d.path,
                "dataset_id": d.metadata_id,
                "samples": d.report.as_ref().map(|r| r.samples),
                "estimated_tokens": d.report.as_ref().and_then(|r| r.estimated_tokens)
            })),
            "training": {
                "epoch": self.training.epoch,
                "step": self.training.step,
                "tokens": self.training.tokens,
                "loss": self.training.loss,
                "avg_loss": self.training.avg_loss,
                "best_loss": self.training.best_loss,
                "duration_seconds": self.training.elapsed_seconds,
                "checkpoint_count": checkpoints,
                "checkpoint": self.training.checkpoint
            },
            "config": self.config.training
        });
        let path = run_dir.join("session.json");
        match serde_json::to_vec_pretty(&payload)
            .map_err(|e| e.to_string())
            .and_then(|bytes| fs::write(&path, bytes).map_err(|e| e.to_string()))
        {
            Ok(()) => self.logger.training(format!("Session summary written: {}", path.display())),
            Err(error) => self.last_error = Some(format!("write session summary: {error}")),
        }
    }

    pub fn update_training_session_marker(&mut self, last_finalized_run: &mut String) {
        if !matches!(
            self.training.state,
            TrainingStatus::Completed | TrainingStatus::Stopped | TrainingStatus::Failed
        ) || self.training.run_id.is_empty()
        {
            return;
        }
        if *last_finalized_run == self.training.run_id {
            return;
        }
        self.write_training_session_summary();
        *last_finalized_run = self.training.run_id.clone();
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
