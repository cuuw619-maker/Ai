use super::{Trainer, TrainingCommand, TrainingEvent};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

pub struct TrainingWorker {
    pub commands: Sender<TrainingCommand>,
    pub events: Receiver<TrainingEvent>,
    join: Option<JoinHandle<()>>,
}

impl TrainingWorker {
    pub fn spawn(mut trainer: Trainer, control_file: Option<PathBuf>) -> Self {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let join = thread::spawn(move || match command_rx.recv() {
            Ok(TrainingCommand::Start) | Ok(TrainingCommand::Resume) => {
                trainer.run(&command_rx, control_file.as_deref(), &event_tx);
            }
            _ => {}
        });
        Self {
            commands: command_tx,
            events: event_rx,
            join: Some(join),
        }
    }

    pub fn send(&self, command: TrainingCommand) -> Result<(), String> {
        self.commands
            .send(command)
            .map_err(|e| format!("training command send: {e}"))
    }

    pub fn is_finished(&self) -> bool {
        self.join.as_ref().is_none_or(std::thread::JoinHandle::is_finished)
    }

    pub fn join(&mut self) {
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}
