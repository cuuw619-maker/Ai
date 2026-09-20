use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingUiConfig {
    pub epochs: u64,
    pub sequence_length: usize,
    pub gradient_accumulation: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub checkpoint_interval_steps: u64,
    pub memory_budget_mb: usize,
    pub max_cpu_threads: usize,
    pub continuous: bool,
    pub autosave: bool,
}

impl Default for TrainingUiConfig {
    fn default() -> Self {
        Self {
            epochs: 1,
            sequence_length: 128,
            gradient_accumulation: 8,
            learning_rate: 0.001,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            checkpoint_interval_steps: 100,
            memory_budget_mb: 3072,
            max_cpu_threads: 2,
            continuous: false,
            autosave: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppConfig {
    pub version: u32,
    pub first_start: bool,
    pub selected_page: String,
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub datasets: Vec<String>,
    pub ui_update_ms: u64,
    pub auto_recovery: bool,
    pub safe_mode: bool,
    pub window_width: f32,
    pub window_height: f32,
    pub window_x: Option<f32>,
    pub window_y: Option<f32>,
    pub performance_profile: String,
    pub training: TrainingUiConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            version: 1,
            first_start: true,
            selected_page: "Home".into(),
            model_path: None,
            tokenizer_path: None,
            datasets: Vec::new(),
            ui_update_ms: 750,
            auto_recovery: true,
            safe_mode: false,
            window_width: 1280.0,
            window_height: 820.0,
            window_x: None,
            window_y: None,
            performance_profile: "AUTO".into(),
            training: TrainingUiConfig::default(),
        }
    }
}

impl AppConfig {
    pub fn load(root: &Path) -> Result<(Self, bool), String> {
        let path = root.join("config.toml");
        if !path.exists() {
            let value = Self::default();
            value.save(root)?;
            return Ok((value, false));
        }
        let content = match fs::read_to_string(&path) {
            Ok(v) => v,
            Err(e) => {
                preserve_broken(&path)?;
                let value = Self::default();
                value.save(root)?;
                return Err(format!("config read failed; safe defaults created: {e}"));
            }
        };
        match toml::from_str::<Self>(&content) {
            Ok(value) => Ok((value, false)),
            Err(e) => {
                preserve_broken(&path)?;
                let value = Self::default();
                value.save(root)?;
                Ok((value, true))
            }
        }
    }

    pub fn save(&self, root: &Path) -> Result<(), String> {
        fs::create_dir_all(root).map_err(|e| format!("create data root: {e}"))?;
        let content = toml::to_string_pretty(self).map_err(|e| format!("serialize config: {e}"))?;
        let temp = root.join("config.toml.tmp");
        fs::write(&temp, content).map_err(|e| format!("write config temp: {e}"))?;
        fs::rename(&temp, root.join("config.toml")).map_err(|e| format!("replace config: {e}"))
    }
}

fn preserve_broken(path: &Path) -> Result<(), String> {
    let mut target = PathBuf::from(path);
    target.set_extension("toml.broken");
    if target.exists() {
        for index in 1..100u32 {
            let candidate = path.with_file_name(format!("config.toml.broken.{index}"));
            if !candidate.exists() {
                target = candidate;
                break;
            }
        }
    }
    fs::rename(path, target).map_err(|e| format!("preserve broken config: {e}"))
}
