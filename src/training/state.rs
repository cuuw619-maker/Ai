use crate::dataset::DatasetCursor;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrainingStatus {
    Idle,
    Starting,
    Running,
    Pausing,
    Paused,
    Resuming,
    Saving,
    Completed,
    Stopping,
    Stopped,
    Failed,
}

impl TrainingStatus {
    pub fn can_transition_to(self, next: Self) -> bool {
        use TrainingStatus::*;
        matches!(
            (self, next),
            (Idle, Starting)
                | (Starting, Running)
                | (Starting, Failed)
                | (Running, Pausing)
                | (Running, Stopping)
                | (Running, Saving)
                | (Running, Completed)
                | (Running, Failed)
                | (Saving, Running)
                | (Saving, Paused)
                | (Saving, Stopped)
                | (Saving, Completed)
                | (Saving, Failed)
                | (Pausing, Saving)
                | (Stopping, Saving)
                | (Paused, Resuming)
                | (Stopped, Resuming)
                | (Completed, Resuming)
                | (Resuming, Running)
                | (Resuming, Failed)
        )
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrainingState {
    pub status: TrainingStatus,
    pub run_id: String,
    pub epoch: u64,
    pub step: u64,
    pub tokens_seen: u64,
    pub tokens_this_run: u64,
    pub tokens_this_epoch: u64,
    pub cursor: DatasetCursor,
    pub random_state: u64,
}

impl TrainingState {
    pub fn transition(&mut self, next: TrainingStatus) -> Result<(), String> {
        if !self.status.can_transition_to(next) {
            return Err(format!("invalid training state transition {:?} -> {:?}", self.status, next));
        }
        self.status = next;
        Ok(())
    }
}
