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
            return Err(format!(
                "invalid training state transition {:?} -> {:?}",
                self.status, next
            ));
        }
        self.status = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{TrainingState, TrainingStatus};
    use crate::dataset::DatasetCursor;

    fn state(status: TrainingStatus) -> TrainingState {
        TrainingState {
            status,
            run_id: "run-000001".into(),
            epoch: 0,
            step: 0,
            tokens_seen: 0,
            tokens_this_run: 0,
            tokens_this_epoch: 0,
            cursor: DatasetCursor {
                dataset_id: "dataset".into(),
                file_path: "data.txt".into(),
                file_offset: 0,
                sample_index: 0,
                token_position: 0,
            },
            random_state: 1,
        }
    }

    #[test]
    fn completed_cannot_resume_without_new_run() {
        assert!(!TrainingStatus::Completed.can_transition_to(TrainingStatus::Resuming));
    }

    #[test]
    fn stopped_can_resume_explicitly() {
        assert!(TrainingStatus::Stopped.can_transition_to(TrainingStatus::Resuming));
    }

    #[test]
    fn running_can_pause_and_stop() {
        assert!(TrainingStatus::Running.can_transition_to(TrainingStatus::Pausing));
        assert!(TrainingStatus::Running.can_transition_to(TrainingStatus::Stopping));
    }

    #[test]
    fn invalid_transition_is_rejected() {
        let mut state = state(TrainingStatus::Completed);
        assert!(state.transition(TrainingStatus::Running).is_err());
    }
}
