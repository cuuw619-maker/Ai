use super::state::TrainingStatus;

#[derive(Clone, Debug)]
pub enum TrainingCommand {
    Start,
    Pause,
    Resume,
    Stop,
}

#[derive(Clone, Debug)]
pub struct TrainingProgress {
    pub run_id: String,
    pub status: TrainingStatus,
    pub epoch: u64,
    pub step: u64,
    pub tokens_seen: u64,
    pub tokens_this_run: u64,
    pub tokens_this_epoch: u64,
    pub loss: f32,
    pub tokens_per_second: f64,
    pub gradient_norm: f32,
    pub clipped: bool,
    pub timestamp_unix_ms: u64,
}

#[derive(Clone, Debug)]
pub enum TrainingEvent {
    Started(TrainingProgress),
    Step(TrainingProgress),
    CheckpointSaved(std::path::PathBuf),
    Paused,
    Resumed,
    Completed,
    Stopped,
    Failed(String),
}
