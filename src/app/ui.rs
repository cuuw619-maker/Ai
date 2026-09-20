use super::core::{AppCore, AppPage, ModelInfo};
use crate::inference::GenerationConfig;
use eframe::egui::{self, Align, Color32, FontId, Layout, RichText, TextStyle, Ui, Vec2};

pub struct AiApplication {
    pub core: AppCore,
    new_model_name: String,
    new_model_vocab: String,
    new_model_embedding: String,
    new_model_hidden: String,
    new_model_layers: String,
    new_model_sequence: String,
    new_model_seed: String,
    temperature: String,
    top_k: String,
    top_p: String,
    max_tokens: String,
    generation_seed: String,
}

impl AiApplication {
    pub fn new(core: AppCore) -> Self {
        Self {
            core,
            new_model_name: "my-model".into(),
            new_model_vocab: "4096".into(),
            new_model_embedding: "64".into(),
            new_model_hidden: "64".into(),
            new_model_layers: "2".into(),
            new_model_sequence: "64".into(),
            new_model_seed: "1".into(),
            temperature: "0.8".into(),
            top_k: "40".into(),
            top_p: "0.9".into(),
            max_tokens: "64".into(),
            generation_seed: "1".into(),
        }
    }

    fn header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading(RichText::new("Ai").strong());
            ui.label(RichText::new("Own Neural Engine").color(Color32::from_rgb(120, 180, 255)));
            ui.label(RichText::new("LOCAL / FROM SCRATCH").small());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let status = self.core.training.label();
                status_pill(ui, status);
                ui.label(format!("AiNet v1.1  •  {}", env!("CARGO_PKG_VERSION")));
            });
        });
        ui.separator();
    }

    fn sidebar(&mut self, ui: &mut Ui) {
        ui.set_width(190.0);
        ui.heading("NEURAL LAB");
        ui.add_space(8.0);
        for page in AppPage::ALL {
            let selected = self.core.page == page;
            if ui.selectable_label(selected, page.name()).clicked() {
                self.core.command_page(page);
            }
        }
        ui.add_space(12.0);
        ui.separator();
        ui.label(RichText::new("ENGINE").small());
        ui.label("AiNet v1.1");
        ui.label("CPU • Local");
        ui.label(format!("RAM {}", AppCore::format_mb(self.core.resources.process_ram_bytes)));
        ui.label(format!("TRAIN {}", self.core.training.label()));
        ui.add_space(12.0);
        if self.core.previous_crash {
            ui.colored_label(Color32::YELLOW, "Previous crash detected");
        }
        if self.core.safe_mode {
            ui.colored_label(Color32::YELLOW, "SAFE MODE");
        }
        ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
            if ui.button("SELF TEST").clicked() {
                self.core.run_self_test();
            }
            if ui.button("OPEN LOGS").clicked() {
                self.core.command_page(AppPage::Logs);
            }
        });
    }

    fn home(&mut self, ui: &mut Ui) {
        ui.heading("Home");
        ui.label("Project state is derived from the native neural/training services.");
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            self.metric_card(ui, "MODEL", self.model_status(), model_detail(self.core.model.as_ref()));
            self.metric_card(ui, "TRAINING", self.core.training.label(), format!("step {} • epoch {}", self.core.training.step, self.core.training.epoch));
            self.metric_card(ui, "SYSTEM", format!("{:.1}% CPU", self.core.resources.cpu_percent), format!("RAM {} available", AppCore::format_mb(self.core.resources.ram_available_bytes)));
            self.metric_card(ui, "DATASET", dataset_status(self.core.dataset.as_ref()), dataset_detail(self.core.dataset.as_ref()));
        });
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            if ui.add_sized([130.0, 36.0], egui::Button::new("CHAT")).clicked() {
                self.core.command_page(AppPage::Chat);
            }
            if ui.add_sized([130.0, 36.0], egui::Button::new("TRAIN")).clicked() {
                self.core.command_page(AppPage::Train);
            }
            let can_pause = matches!(self.core.training.state, crate::training::TrainingStatus::Running);
            if ui.add_enabled(can_pause, egui::Button::new("PAUSE")).clicked() {
                self.core.pause_training();
            }
            let can_resume = matches!(self.core.training.state, crate::training::TrainingStatus::Paused | crate::training::TrainingStatus::Stopped)
                && self.core.worker.is_none()
                && (self.core.recovery_session.is_some() || self.core.training.checkpoint.is_some());
            if ui.add_enabled(can_resume, egui::Button::new("RESUME")).clicked() {
                self.core.resume_training();
            }
        });
        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("LAST CHECKPOINT").small());
            ui.label(self.core.training.checkpoint.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "None".into()));
        });
        if let Some(error) = self.core.last_error.clone() {
            error_card(ui, &error);
        }
    }

    fn metric_card(&self, ui: &mut Ui, title: &str, value: String, detail: String) {
        card_with_size(ui, [250.0, 105.0], |ui| {
            ui.label(RichText::new(title).small());
            ui.label(RichText::new(value).size(22.0).strong());
            ui.label(RichText::new(detail).small());
        });
    }

    fn model_status(&self) -> String {
        self.core.model.as_ref().map(|m| m.status().into()).unwrap_or_else(|| "NOT LOADED".into())
    }

    fn train(&mut self, ui: &mut Ui) {
        ui.heading("Train");
        ui.horizontal(|ui| {
            status_pill(ui, self.core.training.label());
            ui.label(format!("run {}", if self.core.training.run_id.is_empty() { "—" } else { &self.core.training.run_id }));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if self.core.is_trainer_running() {
                    if ui.button("STOP").clicked() { self.core.stop_training(); }
                    if matches!(self.core.training.state, crate::training::TrainingStatus::Running) && ui.button("PAUSE").clicked() {
                        self.core.pause_training();
                    }
                } else if matches!(self.core.training.state, crate::training::TrainingStatus::Paused | crate::training::TrainingStatus::Stopped) && self.core.training.checkpoint.is_some() {
                    if ui.button("RESUME").clicked() { self.core.resume_training(); }
                } else if self.core.worker.is_none() && ui.button("START TRAINING").clicked() {
                    self.core.start_training();
                }
            });
        });
        ui.add_space(8.0);
        ui.columns(2, |columns| {
            card(&mut columns[0], |ui| {
                ui.label(RichText::new("TRAINING METRICS").strong());
                row_value(ui, "State", self.core.training.label());
                row_value(ui, "Epoch", &self.core.training.epoch.to_string());
                row_value(ui, "Step", &self.core.training.step.to_string());
                row_value(ui, "Tokens", &self.core.training.tokens.to_string());
                row_value(ui, "Current loss", loss_string(self.core.training.loss));
                row_value(ui, "Average loss", loss_string(self.core.training.avg_loss));
                row_value(ui, "Best loss", loss_string(self.core.training.best_loss));
                row_value(ui, "Tokens/sec", &format!("{:.2}", self.core.training.tokens_per_second));
                row_value(ui, "Gradient norm", &format!("{:.6}", self.core.training.gradient_norm));
                row_value(ui, "Learning rate", &format!("{:.6}", self.core.training.learning_rate));
                row_value(ui, "Elapsed", &format!("{:.1}s", self.core.training.elapsed_seconds));
            });
            card(&mut columns[1], |ui| {
                ui.label(RichText::new("TRAINING CONFIG").strong());
                row_value(ui, "Model", self.core.model.as_ref().and_then(|m| m.config.as_ref()).map(|c| c.model_id.as_str()).unwrap_or("—"));
                row_value(ui, "Dataset", self.core.dataset.as_ref().map(|d| d.path.file_name().and_then(|v| v.to_str()).unwrap_or("dataset")).unwrap_or("—"));
                row_value(ui, "Sequence", &self.core.config.training.sequence_length.to_string());
                row_value(ui, "Accumulation", &self.core.config.training.gradient_accumulation.to_string());
                row_value(ui, "Threads", &self.core.config.training.max_cpu_threads.to_string());
                row_value(ui, "Memory budget", &format!("{} MB", self.core.config.training.memory_budget_mb));
                row_value(ui, "Checkpoint", &format!("every {} steps", self.core.config.training.checkpoint_interval_steps));
            });
        });
        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("LOSS vs STEP").strong());
            draw_loss_graph(ui, &self.core.loss_points);
        });
        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("TRAINING ACTIVITY").strong());
            egui::ScrollArea::vertical().max_height(180.0).stick_to_bottom(true).show(ui, |ui| {
                for event in self.core.events.iter().rev().take(30) {
                    ui.label(event);
                }
            });
        });
    }

    fn model(&mut self, ui: &mut Ui) {
        ui.heading("Model");
        if let Some(model) = self.core.model.clone() {
            card(ui, |ui| show_model_info(ui, &model));
        } else {
            card(ui, |ui| {
                ui.colored_label(Color32::YELLOW, "Model: NOT LOADED");
                ui.label("Create a random model or load an existing .aimodel.");
            });
        }
        ui.horizontal(|ui| {
            if ui.button("LOAD MODEL").clicked() { self.core.load_model_dialog(); }
            if ui.button("CREATE MODEL").clicked() { self.core.command_page(AppPage::Model); }
            if ui.button("EXPORT INFO").clicked() {
                self.core.export_training_report();
            }
        });
        ui.separator();
        ui.label(RichText::new("CREATE NEW AiNet v1.1").strong());
        egui::Grid::new("model-form").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
            text_field(ui, "Name", &mut self.new_model_name);
            text_field(ui, "Vocabulary", &mut self.new_model_vocab);
            text_field(ui, "Embedding", &mut self.new_model_embedding);
            text_field(ui, "Hidden", &mut self.new_model_hidden);
            text_field(ui, "Layers", &mut self.new_model_layers);
            text_field(ui, "Sequence", &mut self.new_model_sequence);
            text_field(ui, "Seed", &mut self.new_model_seed);
        });
        let vocab = self.new_model_vocab.parse::<usize>().unwrap_or(0);
        let embedding = self.new_model_embedding.parse::<usize>().unwrap_or(0);
        let hidden = self.new_model_hidden.parse::<usize>().unwrap_or(0);
        let layers = self.new_model_layers.parse::<usize>().unwrap_or(0);
        let sequence = self.new_model_sequence.parse::<usize>().unwrap_or(0);
        let params = estimate_parameters(vocab, embedding, hidden, layers);
        let memory = if params > 0 { params.saturating_mul(4) } else { 0 };
        ui.label(format!("Estimated parameters: {}", params));
        ui.label(format!("Estimated weights memory: {}", AppCore::format_mb(memory as u64)));
        if ui.button("CREATE RANDOM MODEL").clicked() {
            let seed = self.new_model_seed.parse::<u64>().unwrap_or(1);
            self.core.create_model(&self.new_model_name, vocab, embedding, hidden, layers, sequence, seed);
        }
    }

    fn dataset(&mut self, ui: &mut Ui) {
        ui.heading("Dataset");
        ui.horizontal(|ui| {
            if ui.button("ADD DATASET").clicked() { self.core.pick_dataset(); }
            if ui.button("VALIDATE").clicked() { self.core.validate_dataset(); }
            if ui.add_enabled(self.core.dataset.is_some(), egui::Button::new("BUILD TOKENIZER")).clicked() {
                self.core.train_tokenizer();
            }
        });
        if let Some(dataset) = self.core.dataset.as_ref() {
            card(ui, |ui| {
                row_value(ui, "File", &dataset.path.display().to_string());
                row_value(ui, "Format", &format!("{:?}", dataset.format));
                row_value(ui, "Dataset ID", dataset.metadata_id.as_deref().unwrap_or("not validated"));
                if let Some(report) = &dataset.report {
                    row_value(ui, "Samples", &report.samples.to_string());
                    row_value(ui, "Errors", &report.errors.to_string());
                    row_value(ui, "Estimated tokens", &report.estimated_tokens.map(|v| v.to_string()).unwrap_or_else(|| "—".into()));
                    if report.errors == 0 {
                        ui.colored_label(Color32::from_rgb(90, 220, 120), "VALID");
                    } else {
                        ui.colored_label(Color32::YELLOW, "VALIDATION ERRORS");
                    }
                }
                if let Some(error) = &dataset.error {
                    error_card(ui, error);
                }
            });
        } else {
            card(ui, |ui| ui.label("No dataset selected."));
        }
        card(ui, |ui| {
            ui.label(RichText::new("TOKENIZER").strong());
            row_value(ui, "Status", if self.core.tokenizer_path.is_some() { "READY" } else { "NOT READY" });
            let vocab = self.core.tokenizer_path.as_ref().and_then(|p| crate::tokenizer::Tokenizer::load(p).ok()).map(|t| t.vocab_size());
            row_value(ui, "Vocabulary", &vocab.map(|v| v.to_string()).unwrap_or_else(|| "—".into()));
            row_value(ui, "Special tokens", if vocab.is_some() { "7" } else { "—" });
        });
    }

    fn memory(&mut self, ui: &mut Ui) {
        ui.heading("Memory");
        let process = self.core.resources.process_ram_bytes;
        let available = self.core.resources.ram_available_bytes;
        card(ui, |ui| {
            row_value(ui, "Process RAM", &AppCore::format_mb(process));
            row_value(ui, "Available RAM", &AppCore::format_mb(available));
            if let Some(model) = &self.core.model {
                if let Some(config) = &model.config {
                    let weight_bytes = model.parameter_count.saturating_mul(4);
                    let bptt_bytes = self.core.config.training.sequence_length
                        .saturating_mul(config.layer_count)
                        .saturating_mul(config.hidden_dim)
                        .saturating_mul(7)
                        .saturating_mul(4);
                    let estimated = weight_bytes
                        .saturating_mul(4)
                        .saturating_add(bptt_bytes.saturating_mul(2))
                        .saturating_add(4 * 1024 * 1024);
                    row_value(ui, "Parameter storage", &AppCore::format_mb(weight_bytes as u64));
                    row_value(ui, "BPTT estimate", &AppCore::format_mb(bptt_bytes as u64));
                    row_value(ui, "Estimated training", &AppCore::format_mb(estimated as u64));
                    row_value(ui, "Sequence", &config.sequence_length.to_string());
                }
            }
        });
        if self.core.low_end_profile() {
            card(ui, |ui| {
                ui.colored_label(Color32::YELLOW, "Performance profile: LOW-END");
                ui.label("The current profile uses small CPU/thread and memory defaults.");
            });
        }
    }

    fn evaluation(&mut self, ui: &mut Ui) {
        ui.heading("Evaluation");
        card(ui, |ui| {
            row_value(ui, "Hook", "TinyEvaluator at epoch boundaries");
            row_value(ui, "Weights mutated", "No — evaluation uses forward-only sequence_loss");
            ui.label("Evaluation metrics are appended to the run metrics.jsonl.");
        });
    }

    fn logs(&mut self, ui: &mut Ui) {
        ui.heading("Logs");
        ui.horizontal(|ui| {
            for name in ["APP LOG", "TRAINING LOG", "INFERENCE LOG", "CRASH LOG"] {
                if ui.selectable_label(self.core.log_channel == name, name).clicked() {
                    self.core.log_channel = name.into();
                }
            }
            if ui.button("OPEN FOLDER").clicked() {
                let _ = std::process::Command::new("explorer").arg(self.core.logger.directory()).spawn();
            }
        });
        let file = match self.core.log_channel.as_str() {
            "TRAINING LOG" => "training.log",
            "INFERENCE LOG" => "inference.log",
            "CRASH LOG" => "crash.log",
            _ => "app.log",
        };
        let mut content = self.core.logger.recent(file, 250);
        egui::TextEdit::multiline(&mut content).font(TextStyle::Monospace).desired_rows(24).interactive(false).show(ui);
        if !content.is_empty() && ui.button("COPY").clicked() {
            ui.ctx().copy_text(content.clone());
        }
    }

    fn settings(&mut self, ui: &mut Ui) {
        ui.heading("Settings");
        egui::Grid::new("settings").num_columns(2).spacing([12.0, 10.0]).show(ui, |ui| {
            ui.label("CPU threads");
            ui.add(egui::DragValue::new(&mut self.core.config.training.max_cpu_threads).range(1..=64));
            ui.end_row();
            ui.label("RAM budget MB");
            ui.add(egui::DragValue::new(&mut self.core.config.training.memory_budget_mb).range(256..=65536));
            ui.end_row();
            ui.label("UI update ms");
            ui.add(egui::DragValue::new(&mut self.core.config.ui_update_ms).range(500..=5000));
            ui.end_row();
            ui.label("Autosave");
            ui.checkbox(&mut self.core.config.training.autosave, "enabled");
            ui.end_row();
            ui.label("Auto recovery");
            ui.checkbox(&mut self.core.config.auto_recovery, "enabled");
            ui.end_row();
        });
        if ui.button("SAVE CONFIG").clicked() {
            self.core.save_config();
        }
    }

    fn system(&mut self, ui: &mut Ui) {
        ui.heading("System");
        card(ui, |ui| {
            row_value(ui, "CPU", &format!("{:.1}%", self.core.resources.cpu_percent));
            row_value(ui, "Process CPU", &format!("{:.1}%", self.core.resources.process_cpu_percent));
            row_value(ui, "RAM used", &AppCore::format_mb(self.core.resources.ram_used_bytes));
            row_value(ui, "RAM available", &AppCore::format_mb(self.core.resources.ram_available_bytes));
            row_value(ui, "Process RAM", &AppCore::format_mb(self.core.resources.process_ram_bytes));
            row_value(ui, "Disk", &format!("{} / {}", AppCore::format_mb(self.core.resources.disk_used_bytes), AppCore::format_mb(self.core.resources.disk_total_bytes)));
            row_value(ui, "CPU model", &self.core.resources.cpu_name);
            row_value(ui, "Threads", &self.core.resources.thread_count.map(|v| v.to_string()).unwrap_or_else(|| "unavailable".into()));
            row_value(ui, "Profile", if self.core.low_end_profile() { "LOW-END" } else { "STANDARD" });
        });
        ui.separator();
        ui.heading("Diagnostics");
        subsystem(ui, "Application", true, "UI is running");
        subsystem(ui, "Neural engine", self.core.model.is_some(), if self.core.model.is_some() { "Model available" } else { "No model loaded" });
        subsystem(ui, "Tokenizer", self.core.tokenizer_path.is_some(), if self.core.tokenizer_path.is_some() { "Tokenizer available" } else { "No tokenizer" });
        subsystem(ui, "Dataset", self.core.dataset.is_some(), if self.core.dataset.is_some() { "Dataset selected" } else { "No dataset" });
        subsystem(ui, "Training", self.core.worker.is_some() || !matches!(self.core.training.state, crate::training::TrainingStatus::Idle), true_status(&self.core.training));
        subsystem(ui, "Storage", self.core.root.exists(), "data directory");
        subsystem(ui, "Checkpoint", self.core.training.checkpoint.is_some() || self.core.recovery_session.is_some(), if self.core.training.checkpoint.is_some() { "Checkpoint available" } else { "None" });
        ui.add_space(8.0);
        if ui.button("RUN SELF TEST").clicked() { self.core.run_self_test(); }
    }

    fn chat(&mut self, ui: &mut Ui) {
        ui.heading("Chat");
        if self.core.model.is_none() || self.core.tokenizer_path.is_none() {
            ui.colored_label(Color32::YELLOW, "Model is untrained or unavailable. Chat can still be opened, but inference needs a model + tokenizer.");
        }
        card(ui, |ui| {
            egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                for msg in &self.core.chat_messages {
                    let heading = if msg.role == "User" { "YOU" } else { "AI" };
                    ui.label(RichText::new(heading).small().strong());
                    ui.label(&msg.text);
                    ui.separator();
                }
                if self.core.chat_generating {
                    ui.label(RichText::new("AI is typing...").italics());
                    ui.label(&self.core.chat_generated);
                }
            });
            ui.horizontal(|ui| {
                let response = ui.add_sized([ui.available_width() - 80.0, 44.0], egui::TextEdit::multiline(&mut self.core.chat_input).hint_text("Ask the local AiNet model..."));
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !self.core.chat_generating {
                    self.send_chat();
                }
                if ui.add_enabled(!self.core.chat_generating, egui::Button::new("SEND")).clicked() {
                    self.send_chat();
                }
                if ui.add_enabled(self.core.chat_generating, egui::Button::new("STOP")).clicked() {
                    self.core.stop_chat();
                }
            });
        });
        ui.separator();
        egui::CollapsingHeader::new("Generation controls").default_open(false).show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Temperature"); ui.text_edit_singleline(&mut self.temperature);
                ui.label("Top-k"); ui.text_edit_singleline(&mut self.top_k);
                ui.label("Top-p"); ui.text_edit_singleline(&mut self.top_p);
            });
            ui.horizontal(|ui| {
                ui.label("Max tokens"); ui.text_edit_singleline(&mut self.max_tokens);
                ui.label("Seed"); ui.text_edit_singleline(&mut self.generation_seed);
                if ui.button("CLEAR").clicked() {
                    self.core.chat_messages.clear();
                    self.core.chat_generated.clear();
                }
            });
        });
    }

    fn send_chat(&mut self) {
        let config = GenerationConfig {
            temperature: self.temperature.parse().unwrap_or(0.8),
            top_k: self.top_k.parse().unwrap_or(40),
            top_p: self.top_p.parse().unwrap_or(0.9),
            greedy: false,
        };
        let max_tokens = self.max_tokens.parse::<usize>().unwrap_or(64);
        let seed = self.generation_seed.parse::<u64>().unwrap_or(1);
        self.core.generate_chat(config, max_tokens, seed);
    }

    fn render_page(&mut self, ui: &mut Ui) {
        match self.core.page {
            AppPage::Home => self.home(ui),
            AppPage::Chat => self.chat(ui),
            AppPage::Train => self.train(ui),
            AppPage::Model => self.model(ui),
            AppPage::Dataset => self.dataset(ui),
            AppPage::Memory => self.memory(ui),
            AppPage::Evaluation => self.evaluation(ui),
            AppPage::Logs => self.logs(ui),
            AppPage::Settings => self.settings(ui),
            AppPage::System => self.system(ui),
        }
    }

    fn recovery_overlay(&mut self, ctx: &egui::Context) {
        if self.core.previous_crash {
            egui::Window::new("Previous crash detected")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("The previous Ai process did not exit cleanly.");
                    ui.label("Choose the startup profile.");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("START NORMALLY").clicked() {
                            self.core.safe_mode = false;
                            self.core.previous_crash = false;
                            self.core.refresh_model();
                            self.core.refresh_tokenizer();
                            self.core.refresh_dataset();
                            self.core.log_event("Normal startup selected after previous crash.");
                        }
                        if ui.button("SAFE MODE").clicked() {
                            self.core.safe_mode = true;
                            self.core.previous_crash = false;
                            self.core.worker = None;
                            self.core.model = None;
                            self.core.log_event("Safe mode enabled.");
                        }
                    });
                    ui.label("Safe Mode does not auto-load the last model and does not start training.");
                    if ui.button("OPEN DIAGNOSTICS").clicked() {
                        self.core.command_page(AppPage::System);
                    }
                    if ui.button("VIEW CRASH LOG").clicked() {
                        self.core.command_page(AppPage::Logs);
                    }
                });
        }
    }

    fn wizard(&mut self, ctx: &egui::Context) {
        if !self.core.wizard_open { return; }
        egui::Window::new("Welcome to Ai")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(RichText::new("Own Neural Engine").size(20.0).strong());
                ui.label("A local neural laboratory built around the project-owned AiNet v1.1.");
                ui.add_space(10.0);
                match self.core.wizard_step {
                    0 => {
                        ui.label("1 / 5  System check");
                        ui.label(format!("CPU: {}", self.core.resources.cpu_name));
                        ui.label(format!("RAM available: {}", AppCore::format_mb(self.core.resources.ram_available_bytes)));
                    }
                    1 => ui.label("2 / 5  Storage setup — data folder is ready."),
                    2 => {
                        ui.label("3 / 5  Create model");
                        if ui.button("OPEN MODEL").clicked() { self.core.command_page(AppPage::Model); }
                    }
                    3 => {
                        ui.label("4 / 5  Add dataset");
                        if ui.button("OPEN DATASET").clicked() { self.core.command_page(AppPage::Dataset); }
                    }
                    _ => ui.label("5 / 5  Start training from the TRAIN screen when model and dataset are ready."),
                }
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if self.core.wizard_step > 0 && ui.button("BACK").clicked() { self.core.wizard_step -= 1; }
                    if self.core.wizard_step < 4 {
                        if ui.button("NEXT").clicked() { self.core.wizard_step += 1; }
                    } else if ui.button("FINISH").clicked() {
                        self.core.wizard_open = false;
                        self.core.config.first_start = false;
                        self.core.save_config();
                    }
                });
            });
    }
}

impl eframe::App for AiApplication {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.core.refresh();
        ctx.set_visuals(egui::Visuals::dark());
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                self.sidebar(ui);
                ui.separator();
                ui.vertical(|ui| {
                    self.header(ui);
                    egui::ScrollArea::vertical().show(ui, |ui| self.render_page(ui));
                });
            });
        });
        self.recovery_overlay(ctx);
        self.wizard(ctx);
        if self.core.tokenizer_job_rx.is_some() {
            egui::Area::new("tokenizer-job".into()).anchor(egui::Align2::CENTER_BOTTOM, [0.0, -20.0]).show(ctx, |ui| {
                card(ui, |ui| ui.label("Tokenizer job is running in background..."));
            });
        }
        if let Some(results) = &self.core.self_test_result {
            egui::Window::new("Self Test").resizable(true).show(ctx, |ui| {
                for result in results {
                    let mark = if result.ok { "PASS" } else { "FAIL" };
                    ui.label(format!("[{mark}] {} — {}", result.name, result.detail));
                }
                if ui.button("CLOSE").clicked() {
                    self.core.self_test_result = None;
                }
            });
        }
        if let Some(error) = self.core.last_error.clone() {
            egui::TopBottomPanel::bottom("error").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(Color32::RED, "ERROR");
                    ui.label(error);
                    if ui.button("DISMISS").clicked() {
                        self.core.last_error = None;
                    }
                });
            });
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
    }
}

fn card(ui: &mut Ui, contents: impl FnOnce(&mut Ui)) {
    egui::Frame::group(ui.style()).inner_margin(12.0).show(ui, contents);
}

fn card_with_size(ui: &mut Ui, size: [f32; 2], contents: impl FnOnce(&mut Ui)) {
    egui::Frame::group(ui.style()).inner_margin(12.0).show(ui, |ui| {
        ui.set_min_size(Vec2::new(size[0], size[1]));
        contents(ui);
    });
}

fn status_pill(ui: &mut Ui, value: &str) {
    let response = ui.add(egui::Label::new(RichText::new(value).monospace().strong()).sense(egui::Sense::hover()));
    if response.hovered() {
        response.on_hover_text("Live state from the application services.");
    }
}

fn row_value(ui: &mut Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(key).small());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| ui.label(value));
    });
}

fn text_field(ui: &mut Ui, label: &str, value: &mut String) {
    ui.label(label);
    ui.text_edit_singleline(value);
    ui.end_row();
}

fn error_card(ui: &mut Ui, error: &str) {
    card(ui, |ui| {
        ui.colored_label(Color32::RED, "ERROR");
        ui.label(error);
    });
}

fn subsystem(ui: &mut Ui, name: &str, ok: bool, detail: &str) {
    ui.horizontal(|ui| {
        let label = if ok { "OK" } else { "ERROR" };
        let color = if ok { Color32::from_rgb(90, 220, 120) } else { Color32::RED };
        ui.colored_label(color, label);
        ui.label(RichText::new(name).strong());
        ui.label(detail);
    });
}

fn true_status(training: &super::core::TrainingSnapshot) -> &str {
    if training.state == crate::training::TrainingStatus::Failed { "Training failed" } else { training.label() }
}

fn model_detail(model: Option<&ModelInfo>) -> String {
    model.and_then(|m| m.config.as_ref()).map(|c| format!("{} • {} params", c.architecture, m.parameter_count)).unwrap_or_else(|| "No model".into())
}

fn dataset_status(dataset: Option<&super::core::DatasetInfo>) -> String {
    dataset.map(|_| "SELECTED".into()).unwrap_or_else(|| "NONE".into())
}

fn dataset_detail(dataset: Option<&super::core::DatasetInfo>) -> String {
    dataset.and_then(|d| d.path.file_name()).and_then(|v| v.to_str()).unwrap_or("No dataset").into()
}

fn loss_string(value: Option<f32>) -> String {
    value.map(|v| format!("{v:.6}")).unwrap_or_else(|| "—".into())
}

fn estimate_parameters(vocab: usize, embedding: usize, hidden: usize, layers: usize) -> usize {
    let embedding_params = vocab.saturating_mul(embedding);
    let projection = hidden.saturating_mul(embedding).saturating_add(hidden);
    let cell = layers.saturating_mul(hidden.saturating_mul(hidden).saturating_mul(7).saturating_add(hidden.saturating_mul(4)));
    let output = vocab.saturating_mul(hidden).saturating_add(vocab);
    embedding_params.saturating_add(projection).saturating_add(cell).saturating_add(output)
}

fn show_model_info(ui: &mut Ui, model: &ModelInfo) {
    row_value(ui, "Status", model.status());
    row_value(ui, "Model", &model.path.display().to_string());
    if let Some(config) = &model.config {
        row_value(ui, "Model ID", &config.model_id);
        row_value(ui, "Architecture", &config.architecture);
        row_value(ui, "Parameters", &model.parameter_count.to_string());
        row_value(ui, "Embedding", &config.embedding_dim.to_string());
        row_value(ui, "Hidden", &config.hidden_dim.to_string());
        row_value(ui, "Layers", &config.layer_count.to_string());
        row_value(ui, "Vocabulary", &config.vocab_size.to_string());
        row_value(ui, "Sequence", &config.sequence_length.to_string());
        row_value(ui, "Seed", &config.seed.to_string());
    }
    row_value(ui, "File size", &AppCore::format_mb(model.file_size));
    row_value(ui, "Checksum", &format!("{:016x}", model.checksum));
    if model.loaded_from_previous {
        ui.colored_label(Color32::YELLOW, "Loaded from previous valid model (.prev).");
    }
}

fn draw_loss_graph(ui: &mut Ui, points: &std::collections::VecDeque<(u64, f32)>) {
    let desired = Vec2::new(ui.available_width(), 260.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(rect, 6.0, egui::Stroke::new(1.0, ui.visuals().widgets.noninteractive.bg_stroke.color), egui::StrokeKind::Outside);
    if points.len() < 2 {
        painter.text(rect.center(), egui::Align2::CENTER_CENTER, "Waiting for real training metrics…", FontId::proportional(16.0), ui.visuals().text_color());
        return;
    }
    let min_loss = points.iter().map(|(_, v)| *v).fold(f32::INFINITY, f32::min);
    let max_loss = points.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let range = (max_loss - min_loss).max(1e-6);
    let min_step = points.front().map(|p| p.0).unwrap_or(0);
    let max_step = points.back().map(|p| p.0).unwrap_or(min_step + 1).max(min_step + 1);
    let inner = rect.shrink(18.0);
    let mut last: Option<egui::Pos2> = None;
    for (step, loss) in points {
        let x = inner.left() + ((*step - min_step) as f32 / (max_step - min_step) as f32) * inner.width();
        let y = inner.bottom() - ((*loss - min_loss) / range) * inner.height();
        let pos = egui::pos2(x, y);
        if let Some(previous) = last {
            painter.line_segment([previous, pos], egui::Stroke::new(2.0, Color32::from_rgb(120, 180, 255)));
        }
        last = Some(pos);
    }
    painter.text(inner.left_top(), egui::Align2::LEFT_TOP, format!("max {:.4}", max_loss), FontId::monospace(11.0), ui.visuals().weak_text_color());
    painter.text(inner.left_bottom(), egui::Align2::LEFT_BOTTOM, format!("min {:.4}", min_loss), FontId::monospace(11.0), ui.visuals().weak_text_color());
}
