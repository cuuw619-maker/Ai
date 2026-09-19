#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub epochs: u64,
    pub sequence_length: usize,
    pub micro_batch: usize,
    pub gradient_accumulation: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub max_grad_norm: f32,
    pub checkpoint_interval_steps: u64,
    pub max_cpu_threads: usize,
    pub memory_budget_mb: usize,
    pub continuous: bool,
    pub max_steps: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::low_end()
    }
}

impl TrainingConfig {
    pub fn low_end() -> Self {
        Self {
            epochs: 1,
            sequence_length: 128,
            micro_batch: 1,
            gradient_accumulation: 8,
            learning_rate: 0.001,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
            checkpoint_interval_steps: 100,
            max_cpu_threads: 2,
            memory_budget_mb: 3072,
            continuous: false,
            max_steps: None,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.epochs == 0 && !self.continuous {
            return Err("epochs must be non-zero unless continuous training is enabled".into());
        }
        if self.sequence_length == 0 || self.micro_batch != 1 || self.gradient_accumulation == 0 {
            return Err(
                "sequence_length, micro_batch=1 and gradient_accumulation must be valid".into(),
            );
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("learning_rate must be finite and positive".into());
        }
        if !self.weight_decay.is_finite() || self.weight_decay < 0.0 {
            return Err("weight_decay cannot be negative or non-finite".into());
        }
        if !self.max_grad_norm.is_finite() || self.max_grad_norm < 0.0 {
            return Err("max_grad_norm cannot be negative or non-finite".into());
        }
        if self.checkpoint_interval_steps == 0
            || self.max_cpu_threads == 0
            || self.memory_budget_mb < 256
        {
            return Err("checkpoint interval, CPU threads and memory budget are invalid".into());
        }
        Ok(())
    }
}
