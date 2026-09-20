use super::core::{AppCore, AppPage, DatasetInfo, ModelInfo};
use crate::inference::GenerationConfig;
use crate::training::TrainingStatus;
use crate::web_learning::WebStatus;
use crate::dataset::DatasetFormat;
use crate::dataset_discovery::{DatasetCandidate, DatasetDiscovery, DatasetEvent, DatasetSearchFilters, DatasetSourceKind, DatasetState};
use super::localization;
use eframe::egui::{
    self, Align, Align2, Color32, FontId, Layout, RichText, Stroke, StrokeKind, TextStyle, Ui, Vec2,
};
use std::collections::VecDeque;
use std::time::Duration;

pub struct AiApplication {
    pub core: AppCore,
    web_source_url: String,
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
    deterministic: bool,
    tokenizer_vocab: Option<(String, usize)>,
    start_training_dialog: bool,
    confirm_clear_logs: bool,
    selected_log: String,
    last_finalized_run: String,
    dataset_discovery: Option<DatasetDiscovery>,
    dd_results: Vec<DatasetCandidate>,
    dd_query: String,
    dd_language: String,
    dd_topic: String,
    dd_kind: String,
    dd_size: String,
    dd_min_quality: f32,
    dd_searching_source: Option<String>,
    dd_search_active: bool,
    dd_search_message: String,
    dd_progress: Option<(String, u64, Option<u64>)>,
    dd_preview: Option<(String, Vec<String>)>,
    dd_pending_approval: Option<(String, u64, u64)>,
}

impl AiApplication {
    pub fn new(mut core: AppCore) -> Self {
        core.apply_low_end_defaults_if_needed();
        if core.safe_mode {
            core.model = None;
            core.tokenizer_path = None;
            core.worker = None;
            core.wizard_open = false;
        }
        if core.previous_crash {
            core.wizard_open = false;
        }

        let selected_log = "APP LOG".to_string();
        let default_vocab = core
            .tokenizer_path
            .as_ref()
            .and_then(|path| crate::tokenizer::Tokenizer::load(path).ok())
            .map(|tokenizer| tokenizer.vocab_size())
            .unwrap_or(263);
        Self {
            core,
            web_source_url: String::new(),
            new_model_name: "my-model".into(),
            new_model_vocab: default_vocab.to_string(),
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
            deterministic: false,
            tokenizer_vocab: None,
            start_training_dialog: false,
            confirm_clear_logs: false,
            selected_log,
            last_finalized_run: String::new(),
            dataset_discovery: if core.safe_mode { None } else { DatasetDiscovery::spawn(&core.root).ok() },
            dd_results: Vec::new(),
            dd_query: "general text".into(),
            dd_language: "English".into(),
            dd_topic: "General text".into(),
            dd_kind: "Text".into(),
            dd_size: "Small".into(),
            dd_min_quality: 0.60,
            dd_searching_source: None,
            dd_search_active: false,
            dd_search_message: String::new(),
            dd_progress: None,
            dd_preview: None,
            dd_pending_approval: None,
        }
    }

    fn nav(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("navigation")
            .resizable(false)
            .default_width(196.0)
            .min_width(196.0)
            .show(ctx, |ui| {
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    ui.heading(RichText::new("Ai").size(28.0).strong());
                    ui.label(RichText::new("LAB").small());
                });
                ui.label(
                    RichText::new("Own Neural Engine").color(Color32::from_rgb(120, 180, 255)),
                );
                ui.add_space(14.0);

                for page in AppPage::ALL {
                    let selected = self.core.page == page;
                    let button =
                        egui::Button::new(RichText::new(localization::page_label(page, &self.core.config.ui_language)).strong().size(if selected {
                            13.0
                        } else {
                            12.0
                        }))
                        .selected(selected);
                    if ui.add_sized([172.0, 34.0], button).clicked() {
                        self.core.command_page(page);
                        self.core.save_config();
                    }
                }

                ui.add_space(18.0);
                ui.separator();
                ui.label(RichText::new("ENGINE").small());
                ui.label("AiNet v1.1");
                ui.label("LOCAL / FROM SCRATCH");
                ui.add_space(6.0);
                status_line(ui, "Model", self.core.model_status_label());
                status_line(ui, "Train", self.core.training.label());
                status_line(
                    ui,
                    "CPU",
                    &format!("{:.1}%", self.core.resources.cpu_percent),
                );
                status_line(
                    ui,
                    "RAM",
                    &AppCore::format_mb(self.core.resources.process_ram_bytes),
                );
                ui.add_space(10.0);

                let profile = if self.core.low_end_profile() {
                    "LOW-END"
                } else {
                    "STANDARD"
                };
                ui.label(RichText::new(format!("PROFILE  {profile}")).small());

                ui.with_layout(Layout::bottom_up(Align::LEFT), |ui| {
                    if ui.button("SELF TEST").clicked() {
                        self.core.run_self_test();
                    }
                    if self.core.previous_crash {
                        ui.colored_label(Color32::YELLOW, "Previous crash detected");
                    }
                    if self.core.safe_mode {
                        ui.colored_label(Color32::YELLOW, "SAFE MODE");
                    }
                });
            });
    }

    fn header(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(localization::page_title(self.core.page, &self.core.config.ui_language))
                    .size(24.0)
                    .strong(),
            );
            ui.label(
                RichText::new("LOCAL / FROM SCRATCH")
                    .small()
                    .color(Color32::from_rgb(120, 180, 255)),
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                status_pill(ui, self.core.training.label());
                ui.label(format!("AiNet v1.1  •  {}", env!("CARGO_PKG_VERSION")));
            });
        });
        ui.separator();
    }

    fn home(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("Own Neural Engine").size(34.0).strong());
        ui.label("A native Windows laboratory for the project-owned AiNet v1.1.");
        ui.add_space(12.0);

        ui.horizontal_wrapped(|ui| {
            metric_card(
                ui,
                "MODEL",
                self.core.model_status_label(),
                model_summary(self.core.model.as_ref()),
            );
            metric_card(
                ui,
                "TRAINING",
                self.core.training.label(),
                format!(
                    "step {} • epoch {} • loss {}",
                    self.core.training.step,
                    self.core.training.epoch,
                    loss_string(self.core.training.loss)
                ),
            );
            metric_card(
                ui,
                "SYSTEM",
                format!("{:.1}% CPU", self.core.resources.cpu_percent),
                format!(
                    "RAM {} available",
                    AppCore::format_mb(self.core.resources.ram_available_bytes)
                ),
            );
            metric_card(
                ui,
                "DATASET",
                dataset_status(self.core.dataset.as_ref()),
                dataset_summary(self.core.dataset.as_ref()),
            );
        });

        ui.add_space(12.0);
        card(ui, |ui| {
            ui.label(RichText::new("CONTROL CENTER").strong());
            ui.horizontal_wrapped(|ui| {
                if ui.button("CHAT").clicked() {
                    self.core.command_page(AppPage::Chat);
                }
                if ui.button("TRAIN").clicked() {
                    self.core.command_page(AppPage::Train);
                }
                let can_pause = matches!(self.core.training.state, TrainingStatus::Running);
                if ui
                    .add_enabled(can_pause, egui::Button::new("PAUSE"))
                    .clicked()
                {
                    self.core.pause_training();
                }
                let can_resume = matches!(
                    self.core.training.state,
                    TrainingStatus::Paused | TrainingStatus::Stopped
                ) && self.core.worker.is_none()
                    && (self.core.recovery_session.is_some()
                        || self.core.training.checkpoint.is_some());
                if ui
                    .add_enabled(can_resume, egui::Button::new("RESUME"))
                    .clicked()
                {
                    self.core.resume_training();
                }
                if self.core.training.state == TrainingStatus::Failed
                    && self.core.training.checkpoint.is_some()
                    && ui.button("LOAD LAST CHECKPOINT").clicked()
                {
                    self.core.resume_training();
                }
            });
        });

        if let Some(recovery) = self.core.recovery_session.clone() {
            card(ui, |ui| {
                ui.colored_label(Color32::YELLOW, "PREVIOUS TRAINING SESSION FOUND");
                ui.label(format!(
                    "{} • step {} • epoch {}",
                    recovery.run_id, recovery.step, recovery.epoch
                ));
                ui.label(format!("Checkpoint: {}", recovery.checkpoint.display()));
                ui.horizontal(|ui| {
                    if ui.button("RESUME").clicked() {
                        self.core.resume_training();
                    }
                    if ui.button("START NEW").clicked() {
                        self.core.discard_recovery_session();
                    }
                    if ui.button("DISCARD SESSION").clicked() {
                        self.core.discard_recovery_session();
                    }
                });
            });
        }

        if let Some(error) = self.core.last_error.clone() {
            error_card(ui, &error);
        }

        card(ui, |ui| {
            ui.label(RichText::new("PROJECT STATUS").strong());
            status_line(ui, "Model", self.core.model_status_label());
            status_line(
                ui,
                "Checkpoint",
                self.core
                    .training
                    .checkpoint
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "None".into())
                    .as_str(),
            );
            status_line(
                ui,
                "Weights checksum",
                &format!("{:016x}", self.core.model_stats.checksum),
            );
            status_line(
                ui,
                "Tokenizer",
                if self.core.tokenizer_path.is_some() {
                    "READY"
                } else {
                    "NOT READY"
                },
            );
        });
    }

    fn train(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            status_pill(ui, self.core.training.label());
            ui.label(format!("run {}", empty_dash(&self.core.training.run_id)));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if self.core.is_trainer_running() {
                    if ui.button("STOP").clicked() {
                        self.core.stop_training();
                    }
                    if self.core.training.state == TrainingStatus::Running
                        && ui.button("PAUSE").clicked()
                    {
                        self.core.pause_training();
                    }
                } else if matches!(
                    self.core.training.state,
                    TrainingStatus::Paused | TrainingStatus::Stopped
                ) && self.core.training.checkpoint.is_some()
                {
                    if ui.button("RESUME").clicked() {
                        self.core.resume_training();
                    }
                } else if self.core.worker.is_none() && ui.button("START TRAINING").clicked() {
                    self.start_training_dialog = true;
                }
            });
        });

        ui.add_space(8.0);
        ui.columns(3, |columns| {
            metric_card(
                &mut columns[0],
                "CURRENT LOSS",
                loss_string(self.core.training.loss),
                format!("best {}", loss_string(self.core.training.best_loss)),
            );
            metric_card(
                &mut columns[1],
                "TOKENS / SEC",
                format!("{:.2}", self.core.training.tokens_per_second),
                format!("tokens {}", self.core.training.tokens),
            );
            metric_card(
                &mut columns[2],
                "GRADIENT",
                format!("{:.6}", self.core.training.gradient_norm),
                format!("lr {:.6}", self.core.training.learning_rate),
            );
        });

        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("TRAINING SNAPSHOT").strong());
            egui::Grid::new("train-snapshot")
                .num_columns(2)
                .spacing([18.0, 6.0])
                .show(ui, |ui| {
                    row_value(ui, "State", self.core.training.label());
                    row_value(ui, "Epoch", &self.core.training.epoch.to_string());
                    row_value(ui, "Step", &self.core.training.step.to_string());
                    row_value(ui, "Tokens", &self.core.training.tokens.to_string());
                    row_value(
                        ui,
                        "Average loss",
                        &loss_string(self.core.training.avg_loss),
                    );
                    row_value(ui, "Best loss", &loss_string(self.core.training.best_loss));
                    row_value(
                        ui,
                        "Tokens/sec",
                        &format!("{:.2}", self.core.training.tokens_per_second),
                    );
                    row_value(ui, "Steps/sec", &steps_per_second(&self.core.training));
                    row_value(
                        ui,
                        "Gradient norm",
                        &format!("{:.6}", self.core.training.gradient_norm),
                    );
                    row_value(
                        ui,
                        "Learning rate",
                        &format!("{:.6}", self.core.training.learning_rate),
                    );
                    row_value(
                        ui,
                        "Elapsed",
                        &format!("{:.1}s", self.core.training.elapsed_seconds),
                    );
                    row_value(
                        ui,
                        "ETA (est.)",
                        &self
                            .core
                            .training_eta_seconds()
                            .map(format_eta)
                            .unwrap_or_else(|| "—".into()),
                    );
                    row_value(
                        ui,
                        "Checkpoint",
                        &self
                            .core
                            .training
                            .checkpoint
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "—".into()),
                    );
                    ui.end_row();
                });
        });

        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("LOSS vs STEP").strong());
            draw_loss_graph(ui, &self.core.loss_points);
        });

        ui.add_space(8.0);
        ui.columns(2, |columns| {
            card(&mut columns[0], |ui| {
                ui.label(RichText::new("LIVE WEIGHT CHANGE").strong());
                row_value(
                    ui,
                    "Parameters",
                    &self.core.model_stats.parameter_count.to_string(),
                );
                row_value(
                    ui,
                    "Updated this step",
                    &self.core.model_stats.updated_parameters.to_string(),
                );
                row_value(
                    ui,
                    "Avg update",
                    &format!("{:.8}", self.core.model_stats.average_update),
                );
                row_value(
                    ui,
                    "Max update",
                    &format!("{:.8}", self.core.model_stats.max_update),
                );
                row_value(
                    ui,
                    "Gradient magnitude",
                    &format!("{:.6}", self.core.model_stats.gradient_norm),
                );
                row_value(
                    ui,
                    "Weight norm",
                    &format!("{:.6}", self.core.model_stats.weight_norm),
                );
                row_value(
                    ui,
                    "Checksum",
                    &format!("{:016x}", self.core.model_stats.checksum),
                );
            });
            card(&mut columns[1], |ui| {
                ui.label(RichText::new("BEFORE / AFTER").strong());
                row_value(
                    ui,
                    "Initial checksum",
                    &self
                        .core
                        .training
                        .initial_checksum
                        .map(|v| format!("{v:016x}"))
                        .unwrap_or_else(|| "—".into()),
                );
                row_value(
                    ui,
                    "Final checksum",
                    &self
                        .core
                        .training
                        .final_checksum
                        .map(|v| format!("{v:016x}"))
                        .unwrap_or_else(|| "—".into()),
                );
                row_value(
                    ui,
                    "Changed",
                    &match (
                        self.core.training.initial_checksum,
                        self.core.training.final_checksum,
                    ) {
                        (Some(before), Some(after)) => (before != after).to_string(),
                        _ => "—".into(),
                    },
                );
                row_value(
                    ui,
                    "Duration",
                    &format!("{:.1}s", self.core.training.elapsed_seconds),
                );
            });
        });

        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("NETWORK / NEURAL ACTIVITY").strong());
            if self.core.model_stats.layers.is_empty() {
                ui.label("Waiting for a real ModelTrainingSnapshot from AiNet.");
            } else {
                draw_neural_activity(ui, &self.core.model_stats.layers);
            }
        });

        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("ROUTING ACTIVITY").strong());
            let total_channels: usize = self.core.model_stats.layers.iter().map(|l| l.active_channels + l.skipped_channels).sum();
            let active_channels: usize = self.core.model_stats.layers.iter().map(|l| l.active_channels).sum();
            let skipped_channels: usize = self.core.model_stats.layers.iter().map(|l| l.skipped_channels).sum();
            let entropy = if self.core.model_stats.layers.is_empty() {
                0.0
            } else {
                self.core.model_stats.layers.iter().map(|l| l.routing_entropy).sum::<f32>() / self.core.model_stats.layers.len() as f32
            };
            row_value(ui, "Total channels", &total_channels.to_string());
            row_value(ui, "Active channels", &active_channels.to_string());
            row_value(ui, "Active ratio", &format!("{:.1}%", if total_channels == 0 { 0.0 } else { active_channels as f32 * 100.0 / total_channels as f32 }));
            row_value(ui, "Skipped channels", &skipped_channels.to_string());
            row_value(ui, "Average routing entropy", &format!("{:.4}", entropy));
            for layer in &self.core.model_stats.layers {
                row_value(ui, &format!("AiCell {} blocks", layer.layer), &format!("{} active / {} skipped", layer.active_blocks, layer.skipped_blocks));
            }
        });

        ui.add_space(8.0);
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("TRAINING TIMELINE").strong());
                if ui.button("EXPORT TRAINING REPORT").clicked() {
                    self.core.export_training_report();
                }
                if ui.button("OPEN RUN").clicked() {
                    self.core.open_run_folder();
                }
            });
            egui::ScrollArea::vertical()
                .max_height(190.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for event in self.core.events.iter().rev().take(50) {
                        ui.label(event);
                    }
                });
        });
    }

    fn web_learning(&mut self, ui: &mut Ui) {
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Source URL");
                ui.text_edit_singleline(&mut self.web_source_url);
                if ui.button("ADD SOURCE").clicked() {
                    let url = self.web_source_url.trim().to_string();
                    if !url.is_empty() {
                        self.core.add_web_source(&url);
                        self.web_source_url.clear();
                    }
                }
            });
        });
        ui.add_space(8.0);
        card(ui, |ui| {
            ui.horizontal(|ui| {
                status_pill(ui, match self.core.web_status {
                    WebStatus::Running => "RUNNING",
                    WebStatus::Starting => "STARTING",
                    WebStatus::Pausing => "PAUSING",
                    WebStatus::Paused => "PAUSED",
                    WebStatus::Stopping => "STOPPING",
                    WebStatus::Offline => "OFFLINE",
                    WebStatus::Error => "ERROR",
                    WebStatus::Stopped => "STOPPED",
                });
                ui.label("Public text ingestion → Corpus → Tokenizer → Trainer");
            });
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                if ui.button("START").clicked() {
                    self.core.start_web_learning();
                }
                if ui.button("PAUSE WEB").clicked() {
                    self.core.pause_web_learning();
                }
                if ui.button("PAUSE TRAINING").clicked() {
                    self.core.pause_training();
                }
                if ui.button("PAUSE ALL").clicked() {
                    self.core.pause_all();
                }
                if ui.button("RESUME ALL").clicked() {
                    self.core.resume_all();
                }
                if ui.button("STOP").clicked() {
                    self.core.stop_web_learning();
                }
                if ui.button("STOP ALL").clicked() {
                    self.core.stop_all();
                }
                if ui.button("SCAN NOW").clicked() {
                    self.core.scan_web_now();
                }
            });
            ui.checkbox(&mut self.core.config.web.autonomous, "AUTONOMOUS LEARNING");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Data languages:");
                for language in ["English", "Russian", "Ukrainian", "All"] {
                    let enabled = self.core.config.web.languages.iter().any(|v| v.eq_ignore_ascii_case(language));
                    if ui.selectable_label(enabled, language).clicked() {
                        if language == "All" {
                            self.core.config.web.languages = vec!["All".into()];
                        } else {
                            self.core.config.web.languages.retain(|v| !v.eq_ignore_ascii_case("All"));
                            if enabled && self.core.config.web.languages.len() > 1 {
                                self.core.config.web.languages.retain(|v| !v.eq_ignore_ascii_case(language));
                            } else if !enabled {
                                self.core.config.web.languages.push(language.into());
                            }
                            if self.core.config.web.languages.is_empty() {
                                self.core.config.web.languages.push("English".into());
                            }
                        }
                        self.core.save_config();
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Freshness:");
                for (label, hours) in [("1h", 1u64), ("6h", 6), ("1d", 24), ("1w", 168), ("All", 0)] {
                    if ui.selectable_label(self.core.config.web.freshness_hours == hours, label).clicked() {
                        self.core.config.web.freshness_hours = hours;
                        self.core.save_config();
                    }
                }
            });
        });

        ui.add_space(10.0);
        ui.horizontal_wrapped(|ui| {
            metric_card(ui, "Sources", self.core.web_stats.sources.to_string(), format!("active {}", self.core.web_stats.active_sources));
            metric_card(ui, "Pages scanned", self.core.web_stats.pages_scanned.to_string(), format!("accepted {}", self.core.web_stats.pages_accepted));
            metric_card(ui, "Pages rejected", self.core.web_stats.pages_rejected.to_string(), format!("duplicates {}", self.core.web_stats.duplicates_skipped));
            metric_card(ui, "Articles", self.core.web_stats.articles_collected.to_string(), "quality-filtered".to_string());
            metric_card(ui, "Tokens queued", self.core.web_stats.tokens_queued.to_string(), format!("queue items {}", self.core.web_stats.training_queue));
        });

        ui.add_space(10.0);
        card(ui, |ui| {
            ui.label(RichText::new("SOURCE ACTIVITY").strong());
            egui::Grid::new("web-sources").striped(true).num_columns(6).show(ui, |ui| {
                ui.label("Source"); ui.label("Domain"); ui.label("Priority"); ui.label("Status"); ui.label("Last scan"); ui.label("Errors"); ui.end_row();
                for source in &self.core.web_sources {
                    ui.label(&source.id); ui.label(&source.domain); ui.label(format!("{:?}", source.priority));
                    let status = if source.last_success.is_some() { "READY" } else if source.error_count > 0 { "ERROR" } else { "PENDING" };
                    ui.label(status); ui.label(source.last_scan.map(|v| v.to_string()).unwrap_or_else(|| "—".into())); ui.label(source.error_count.to_string()); ui.end_row();
                }
            });
            row_value(ui, "Current source", self.core.web_stats.current_source.as_deref().unwrap_or("—"));
            row_value(ui, "Current URL", self.core.web_stats.current_url.as_deref().unwrap_or("—"));
            row_value(ui, "Last update", &self.core.web_stats.last_update.map(|v| v.to_string()).unwrap_or_else(|| "—".into()));
            row_value(ui, "Registry", &self.core.root.join("web_learning/sources.json").display().to_string());
            row_value(ui, "Corpus", &self.core.root.join("web_corpus/web.aicorpus").display().to_string());
            row_value(ui, "SQLite metadata", &self.core.root.join("web_learning/web.sqlite").display().to_string());
        });

        if let Some(error) = self.core.last_error.as_deref() {
            error_card(ui, error);
        }
    }


    fn model(&mut self, ui: &mut Ui) {
        if let Some(model) = self.core.model.clone() {
            card(ui, |ui| {
                show_model_info(ui, &model, self.core.model_status_label());
            });
        } else {
            card(ui, |ui| {
                ui.colored_label(Color32::YELLOW, "MODEL: NOT LOADED");
                ui.label("No model is currently loaded. Creation uses random weights from a deterministic seed.");
            });
        }

        ui.horizontal_wrapped(|ui| {
            if ui.button("LOAD MODEL").clicked() && !self.core.safe_mode {
                self.core.load_model_dialog();
            }
            if ui.button("SAVE MODEL").clicked() {
                self.core.save_current_model_dialog();
            }
            if ui.button("EXPORT INFO").clicked() {
                self.core.export_model_info();
            }
        });

        ui.add_space(12.0);
        card(ui, |ui| {
            ui.label(RichText::new("CREATE NEW AiNet v1.1").strong());
            egui::Grid::new("new-model-form")
                .num_columns(2)
                .spacing([12.0, 8.0])
                .show(ui, |ui| {
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
            let parameters = estimate_parameters(vocab, embedding, hidden, layers);
            let memory = parameters.saturating_mul(4);
            let training = estimate_training_memory(parameters, hidden, layers, sequence.max(1));
            ui.add_space(6.0);
            row_value(ui, "Estimated parameters", &parameters.to_string());
            row_value(
                ui,
                "Estimated model RAM",
                &AppCore::format_mb(memory as u64),
            );
            row_value(
                ui,
                "Estimated training RAM",
                &AppCore::format_mb(training as u64),
            );
            if ui.button("CREATE RANDOM MODEL").clicked() {
                let seed = self.new_model_seed.parse::<u64>().unwrap_or(1);
                self.core.create_model(
                    &self.new_model_name,
                    vocab,
                    embedding,
                    hidden,
                    layers,
                    sequence,
                    seed,
                );
            }
        });
    }

    fn dataset(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui.button("ADD DATASET").clicked() && !self.core.safe_mode {
                self.core.pick_dataset();
            }
            if ui.button("VALIDATE").clicked() {
                self.core.validate_dataset();
            }
            if ui
                .add_enabled(
                    self.core.dataset.is_some() && !self.core.safe_mode,
                    egui::Button::new("BUILD TOKENIZER"),
                )
                .clicked()
            {
                self.core.train_tokenizer();
            }
            if self.core.dataset.is_some() && ui.button("REMOVE").clicked() {
                self.core.remove_dataset();
            }
        });

        let dataset = self.core.dataset.clone();
        if let Some(dataset) = dataset {
            card(ui, |ui| {
                row_value(ui, "File", &dataset.path.display().to_string());
                row_value(ui, "Format", &format!("{:?}", dataset.format));
                row_value(
                    ui,
                    "Encoding",
                    "UTF-8 input; tokenizer operates on raw bytes",
                );
                row_value(
                    ui,
                    "Dataset ID",
                    dataset.metadata_id.as_deref().unwrap_or("not validated"),
                );
                if let Some(report) = &dataset.report {
                    row_value(ui, "Samples", &report.samples.to_string());
                    row_value(ui, "Errors", &report.errors.to_string());
                    row_value(
                        ui,
                        "Estimated tokens",
                        &report
                            .estimated_tokens
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "—".into()),
                    );
                    if report.errors == 0 {
                        ui.colored_label(Color32::from_rgb(90, 220, 120), "VALID");
                    } else {
                        ui.colored_label(Color32::YELLOW, "VALIDATION ERRORS");
                    }
                } else {
                    ui.colored_label(Color32::YELLOW, "VALIDATION REQUIRED BEFORE TRAINING");
                }
                if let Some(error) = &dataset.error {
                    error_card(ui, error);
                }
            });
        } else {
            card(ui, |ui| {
                ui.label("No dataset selected.");
            });
        }

        self.update_tokenizer_cache();
        card(ui, |ui| {
            ui.label(RichText::new("TOKENIZER").strong());
            row_value(
                ui,
                "Status",
                if self.core.tokenizer_path.is_some() {
                    "READY"
                } else {
                    "NOT READY"
                },
            );
            row_value(
                ui,
                "Vocabulary",
                &self
                    .tokenizer_vocab
                    .as_ref()
                    .map(|(_, size)| size.to_string())
                    .unwrap_or_else(|| "—".into()),
            );
            row_value(
                ui,
                "Special tokens",
                if self.tokenizer_vocab.is_some() {
                    "7"
                } else {
                    "—"
                },
            );
            if let Some(path) = &self.core.tokenizer_path {
                row_value(ui, "File", &path.display().to_string());
            }
        });
    }

    fn memory(&mut self, ui: &mut Ui) {
        let mut model_bytes = 0u64;
        let mut gradient_bytes = 0u64;
        let mut optimizer_bytes = 0u64;
        let mut bptt_bytes = 0u64;
        let mut total_estimate = 0u64;

        if let Some(model) = &self.core.model {
            let parameter_bytes = (model.parameter_count as u64).saturating_mul(4);
            model_bytes = parameter_bytes;
            gradient_bytes = parameter_bytes;
            optimizer_bytes = parameter_bytes.saturating_mul(2);
            if let Some(config) = &model.config {
                bptt_bytes = (self.core.config.training.sequence_length as u64)
                    .saturating_mul(config.layer_count as u64)
                    .saturating_mul(config.hidden_dim as u64)
                    .saturating_mul(7)
                    .saturating_mul(4);
            }
            total_estimate = model_bytes
                .saturating_add(gradient_bytes)
                .saturating_add(optimizer_bytes)
                .saturating_add(bptt_bytes)
                .saturating_add(bptt_bytes)
                .saturating_add(4 * 1024 * 1024);
        }

        card(ui, |ui| {
            row_value(ui, "Model RAM", &AppCore::format_mb(model_bytes));
            row_value(ui, "Gradient RAM", &AppCore::format_mb(gradient_bytes));
            row_value(ui, "Optimizer RAM (Adam m+v)", &AppCore::format_mb(optimizer_bytes));
            row_value(ui, "BPTT RAM", &AppCore::format_mb(bptt_bytes));
            row_value(ui, "Web cache RAM", "0 MB (no retained in-memory page cache)");
            row_value(ui, "Estimated training RAM", &AppCore::format_mb(total_estimate));
            row_value(ui, "Process RAM", &AppCore::format_mb(self.core.resources.process_ram_bytes));
            row_value(ui, "Available RAM", &AppCore::format_mb(self.core.resources.ram_available_bytes));
            if self.core.model.is_none() {
                ui.label("No model loaded; model-specific memory is 0.");
            }
        });

        if let Some(model) = &self.core.model {
            if let Some(config) = &model.config {
                let memory_budget = (self.core.config.training.memory_budget_mb as u64).saturating_mul(1024 * 1024);
                if total_estimate > memory_budget {
                    card(ui, |ui| {
                        ui.colored_label(Color32::YELLOW, "MEMORY BUDGET WARNING");
                        ui.label(format!(
                            "Estimated training RAM {} exceeds configured budget {}.",
                            AppCore::format_mb(total_estimate),
                            AppCore::format_mb(memory_budget)
                        ));
                    });
                }
                ui.add_space(8.0);
                card(ui, |ui| {
                    ui.label(RichText::new("MODEL SIZE").strong());
                    row_value(ui, "Parameters", &model.parameter_count.to_string());
                    row_value(ui, "Weight RAM", &AppCore::format_mb(model_bytes));
                    row_value(ui, "Layers", &config.layer_count.to_string());
                    row_value(ui, "Hidden", &config.hidden_dim.to_string());
                    row_value(ui, "Sequence", &self.core.config.training.sequence_length.to_string());
                    row_value(ui, "Estimated total", &AppCore::format_mb(total_estimate));
                });
            }
        }

        if self.core.low_end_profile() {
            card(ui, |ui| {
                ui.colored_label(Color32::YELLOW, "Performance profile: LOW-END");
                ui.label("Automatic limits: <=2 training threads, reduced BPTT sequence, bounded memory budget.");
            });
        }
    }

    fn evaluation(&mut self, ui: &mut Ui) {
        card(ui, |ui| {
            row_value(ui, "Hook", "TinyEvaluator at epoch boundaries");
            row_value(ui, "Mutates weights", "No");
            row_value(ui, "Metrics", "Written to run metrics.jsonl");
        });
    }

    fn logs(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for name in [
                "APP LOG",
                "TRAINING LOG",
                "INFERENCE LOG",
                "WEB LOG",
                "ROUTER LOG",
                "CRASH LOG",
            ] {
                if ui
                    .selectable_label(self.selected_log == name, name)
                    .clicked()
                {
                    self.selected_log = name.to_string();
                }
            }
            if ui.button("OPEN FOLDER").clicked() {
                self.core.open_logs_folder();
            }
            if ui.button("COPY").clicked() {
                let content = self.current_log_contents();
                ui.ctx().copy_text(content);
            }
            if ui.button("CLEAR ROTATED LOGS").clicked() {
                self.confirm_clear_logs = true;
            }
        });

        if self.confirm_clear_logs {
            card(ui, |ui| {
                ui.colored_label(
                    Color32::YELLOW,
                    "Only rotated logs (.1 / .2) will be removed.",
                );
                ui.horizontal(|ui| {
                    if ui.button("CONFIRM CLEAR").clicked() {
                        match self.core.clear_rotated_log_files() {
                            Ok(()) => self.confirm_clear_logs = false,
                            Err(error) => self.core.last_error = Some(error),
                        }
                    }
                    if ui.button("CANCEL").clicked() {
                        self.confirm_clear_logs = false;
                    }
                });
            });
        }

        let mut content = self.current_log_contents();
        egui::TextEdit::multiline(&mut content)
            .font(TextStyle::Monospace)
            .desired_rows(25)
            .interactive(false)
            .show(ui);
    }

    fn settings(&mut self, ui: &mut Ui) {
        card(ui, |ui| {
            ui.label(RichText::new("UI Language").strong());
            let mut language = self.core.config.ui_language.clone();
            egui::ComboBox::from_id_salt("ui-language")
                .selected_text(if localization::is_russian(&language) { "Русский" } else { "English" })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut language, "English".into(), "English");
                    ui.selectable_value(&mut language, "Russian".into(), "Русский");
                });
            if language != self.core.config.ui_language {
                self.core.config.ui_language = language;
                self.core.save_config();
            }

            ui.label(RichText::new("AI/Data Languages").strong());
            let options = ["English", "Russian", "Ukrainian"];
            for option in options {
                let mut enabled = self.core.config.ai_languages.iter().any(|v| v.eq_ignore_ascii_case(option));
                if ui.checkbox(&mut enabled, option).changed() {
                    if enabled {
                        if !self.core.config.ai_languages.iter().any(|v| v.eq_ignore_ascii_case(option)) {
                            self.core.config.ai_languages.push(option.into());
                        }
                    } else {
                        self.core.config.ai_languages.retain(|v| !v.eq_ignore_ascii_case(option));
                    }
                    if self.core.config.ai_languages.is_empty() {
                        self.core.config.ai_languages.push("English".into());
                    }
                    self.core.save_config();
                }
            }

            egui::Grid::new("settings")
                .num_columns(2)
                .spacing([16.0, 10.0])
                .show(ui, |ui| {
                    ui.label("CPU threads");
                    ui.add(
                        egui::DragValue::new(&mut self.core.config.training.max_cpu_threads)
                            .range(1..=64),
                    );
                    ui.end_row();
                    ui.label("RAM budget MB");
                    ui.add(
                        egui::DragValue::new(&mut self.core.config.training.memory_budget_mb)
                            .range(256..=65536),
                    );
                    ui.end_row();
                    ui.label("Gradient accumulation");
                    ui.add(
                        egui::DragValue::new(&mut self.core.config.training.gradient_accumulation)
                            .range(1..=128),
                    );
                    ui.end_row();
                    ui.label("Sequence");
                    ui.add(
                        egui::DragValue::new(&mut self.core.config.training.sequence_length)
                            .range(1..=4096),
                    );
                    ui.end_row();
                    ui.label("Learning rate");
                    ui.add(
                        egui::DragValue::new(&mut self.core.config.training.learning_rate)
                            .speed(0.0001)
                            .range(0.000001..=1.0),
                    );
                    ui.end_row();
                    ui.label("Epochs");
                    ui.add(
                        egui::DragValue::new(&mut self.core.config.training.epochs)
                            .range(1..=1_000_000),
                    );
                    ui.end_row();
                    ui.label("Checkpoint every");
                    ui.add(
                        egui::DragValue::new(
                            &mut self.core.config.training.checkpoint_interval_steps,
                        )
                        .range(1..=1_000_000),
                    );
                    ui.end_row();
                    ui.label("UI update ms");
                    ui.add(
                        egui::DragValue::new(&mut self.core.config.ui_update_ms).range(500..=5000),
                    );
                    ui.end_row();
                    ui.label("Autosave");
                    ui.checkbox(&mut self.core.config.training.autosave, "enabled");
                    ui.end_row();
                    ui.label("Auto recovery");
                    ui.checkbox(&mut self.core.config.auto_recovery, "enabled");
                    ui.end_row();
                    ui.label("Performance profile");
                    ui.label(&self.core.config.performance_profile);
                    ui.end_row();
                });
        });

        ui.horizontal(|ui| {
            if ui.button("SAVE CONFIG").clicked() {
                self.core.save_config();
            }
            if ui.button("SAFE MODE (SESSION)").clicked() {
                self.core.safe_mode = true;
                self.core.worker = None;
                self.core.model = None;
                self.core.tokenizer_path = None;
                self.core.log_event("Safe mode enabled from Settings.");
            }
            if ui.button("NORMAL MODE").clicked() {
                self.core.safe_mode = false;
                self.core.refresh_model();
                self.core.refresh_tokenizer();
                self.core.refresh_dataset();
                self.core.log_event("Normal mode enabled.");
            }
        });
        ui.label("Safe Mode never starts training and does not auto-load a model.");
    }

    fn system(&mut self, ui: &mut Ui) {
        card(ui, |ui| {
            row_value(
                ui,
                "CPU",
                &format!("{:.1}%", self.core.resources.cpu_percent),
            );
            row_value(
                ui,
                "Process CPU",
                &format!("{:.1}%", self.core.resources.process_cpu_percent),
            );
            row_value(
                ui,
                "RAM used",
                &AppCore::format_mb(self.core.resources.ram_used_bytes),
            );
            row_value(
                ui,
                "RAM available",
                &AppCore::format_mb(self.core.resources.ram_available_bytes),
            );
            row_value(
                ui,
                "Process RAM",
                &AppCore::format_mb(self.core.resources.process_ram_bytes),
            );
            row_value(
                ui,
                "Disk",
                &format!(
                    "{} / {}",
                    AppCore::format_mb(self.core.resources.disk_used_bytes),
                    AppCore::format_mb(self.core.resources.disk_total_bytes)
                ),
            );
            row_value(ui, "CPU model", &self.core.resources.cpu_name);
            row_value(
                ui,
                "Process threads",
                &self
                    .core
                    .resources
                    .thread_count
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unavailable".into()),
            );
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0);
            row_value(ui, "Logical CPU threads", &logical.to_string());
            row_value(ui, "CPU temperature", "unavailable");
            row_value(
                ui,
                "Performance profile",
                if self.core.low_end_profile() {
                    "LOW-END"
                } else {
                    "STANDARD"
                },
            );
        });

        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new("DIAGNOSTICS").strong());
            subsystem(ui, "Application", true, "native UI is running");
            subsystem(
                ui,
                "Neural engine",
                self.core.model.is_some(),
                if self.core.model.is_some() {
                    "AiNet model loaded"
                } else {
                    "model not loaded"
                },
            );
            subsystem(
                ui,
                "Tokenizer",
                self.core.tokenizer_path.is_some(),
                if self.core.tokenizer_path.is_some() {
                    "ready"
                } else {
                    "not ready"
                },
            );
            let dataset_ok = self
                .core
                .dataset
                .as_ref()
                .and_then(|d| d.report.as_ref())
                .is_some_and(|report| report.errors == 0);
            subsystem(
                ui,
                "Dataset",
                dataset_ok,
                if self.core.dataset.is_none() {
                    "no dataset selected"
                } else if dataset_ok {
                    "validated"
                } else {
                    "validation required"
                },
            );
            subsystem(
                ui,
                "Training",
                !matches!(self.core.training.state, TrainingStatus::Failed),
                self.core.training.label(),
            );
            subsystem(ui, "Storage", self.core.root.exists(), "data directory");
            subsystem(
                ui,
                "Checkpoint",
                self.core.training.checkpoint.is_some() || self.core.recovery_session.is_some(),
                if self.core.training.checkpoint.is_some() {
                    "checkpoint available"
                } else {
                    "none"
                },
            );
            subsystem(
                ui,
                "CPU",
                self.core.resources.cpu_percent.is_finite(),
                "live system sample",
            );
            subsystem(
                ui,
                "RAM",
                self.core.resources.ram_available_bytes > 0,
                "live system sample",
            );
        });

        if ui.button("RUN SELF TEST").clicked() {
            self.core.run_self_test();
        }

        if let Some(error) = &self.core.last_error {
            error_card(ui, error);
        }
    }

    fn chat(&mut self, ui: &mut Ui) {
        if self.core.model.is_none() || self.core.tokenizer_path.is_none() {
            card(ui, |ui| {
                ui.colored_label(Color32::YELLOW, "Inference: OFFLINE");
                ui.label("Load a model and tokenizer to run the local AiNet inference engine.");
            });
        } else {
            card(ui, |ui| {
                status_line(
                    ui,
                    "Inference",
                    if self.core.chat_generating {
                        "GENERATING"
                    } else {
                        "READY"
                    },
                );
                status_line(ui, "Model", self.core.model_status_label());
            });
        }

        card(ui, |ui| {
            egui::ScrollArea::vertical()
                .max_height(420.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for message in &self.core.chat_messages {
                        ui.label(
                            RichText::new(message.role.to_ascii_uppercase())
                                .small()
                                .strong(),
                        );
                        ui.label(&message.text);
                        ui.separator();
                    }
                    if self.core.chat_generating {
                        ui.label(RichText::new("AI is typing…").italics());
                        ui.label(&self.core.chat_generated);
                    }
                });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let width = (ui.available_width() - 92.0).max(160.0);
                let response = ui.add_sized(
                    [width, 56.0],
                    egui::TextEdit::multiline(&mut self.core.chat_input)
                        .hint_text("Talk to the local AiNet model…"),
                );
                if response.lost_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter))
                    && !self.core.chat_generating
                {
                    self.send_chat();
                }
                if ui
                    .add_enabled(!self.core.chat_generating, egui::Button::new("SEND"))
                    .clicked()
                {
                    self.send_chat();
                }
                if ui
                    .add_enabled(self.core.chat_generating, egui::Button::new("STOP"))
                    .clicked()
                {
                    self.core.stop_chat();
                }
            });
        });

        egui::CollapsingHeader::new("Generation controls")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Temperature");
                    ui.text_edit_singleline(&mut self.temperature);
                    ui.label("Top-k");
                    ui.text_edit_singleline(&mut self.top_k);
                    ui.label("Top-p");
                    ui.text_edit_singleline(&mut self.top_p);
                    ui.checkbox(&mut self.deterministic, "Deterministic");
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Max tokens");
                    ui.text_edit_singleline(&mut self.max_tokens);
                    ui.label("Seed");
                    ui.text_edit_singleline(&mut self.generation_seed);
                    if ui.button("CLEAR CHAT").clicked() {
                        self.core.chat_messages.clear();
                        self.core.chat_generated.clear();
                    }
                    if ui.button("NEW CHAT").clicked() {
                        self.core.chat_messages.clear();
                        self.core.chat_generated.clear();
                    }
                });
            });
    }

    fn send_chat(&mut self) {
        let config = GenerationConfig {
            temperature: self.temperature.parse::<f32>().unwrap_or(0.8).clamp(0.01, 10.0),
            top_k: self.top_k.parse().unwrap_or(40).max(1),
            top_p: self.top_p.parse::<f32>().unwrap_or(0.9).clamp(0.01, 1.0),
            greedy: self.deterministic,
        };
        let max_tokens = self
            .max_tokens
            .parse::<usize>()
            .unwrap_or(64)
            .clamp(1, 4096);
        let seed = self.generation_seed.parse::<u64>().unwrap_or(1);
        self.core.generate_chat(config, max_tokens, seed);
    }

    fn start_training_dialog(&mut self, ctx: &egui::Context) {
        if !self.start_training_dialog {
            return;
        }
        let mut open = true;
        egui::Window::new("Start Training")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                if self.core.model.is_none()
                    || self.core.dataset.is_none()
                    || self.core.tokenizer_path.is_none()
                {
                    ui.colored_label(
                        Color32::YELLOW,
                        "Model, tokenizer and dataset are required.",
                    );
                }
                if self
                    .core
                    .dataset
                    .as_ref()
                    .and_then(|d| d.report.as_ref())
                    .is_none()
                {
                    ui.colored_label(
                        Color32::YELLOW,
                        "Dataset validation is required before training.",
                    );
                }
                let model = self.core.model.clone();
                if let Some(model) = model {
                    let parameter_memory = model.parameter_count.saturating_mul(4);
                    let training_memory = model
                        .config
                        .as_ref()
                        .map(|config| {
                            estimate_training_memory(
                                model.parameter_count,
                                config.hidden_dim,
                                config.layer_count,
                                self.core.config.training.sequence_length,
                            )
                        })
                        .unwrap_or(0);
                    card(ui, |ui| {
                        row_value(ui, "Model", &model.path.display().to_string());
                        row_value(ui, "Parameters", &model.parameter_count.to_string());
                        row_value(
                            ui,
                            "Model RAM",
                            &AppCore::format_mb(parameter_memory as u64),
                        );
                        row_value(
                            ui,
                            "Training RAM",
                            &AppCore::format_mb(training_memory as u64),
                        );
                        row_value(
                            ui,
                            "Available RAM",
                            &AppCore::format_mb(self.core.resources.ram_available_bytes),
                        );
                        row_value(
                            ui,
                            "Sequence",
                            &self.core.config.training.sequence_length.to_string(),
                        );
                        row_value(
                            ui,
                            "Threads",
                            &self.core.config.training.max_cpu_threads.to_string(),
                        );
                        row_value(ui, "Batch", "1 micro-batch");
                        row_value(
                            ui,
                            "Learning rate",
                            &self.core.config.training.learning_rate.to_string(),
                        );
                        row_value(ui, "Epochs", &self.core.config.training.epochs.to_string());
                        row_value(
                            ui,
                            "Checkpoint",
                            &self
                                .core
                                .config
                                .training
                                .checkpoint_interval_steps
                                .to_string(),
                        );
                    });
                    let danger = training_memory as u64
                        > self.core.resources.ram_available_bytes.saturating_mul(8) / 10;
                    if danger {
                        ui.colored_label(
                            Color32::YELLOW,
                            "Warning: estimated training memory is close to available RAM.",
                        );
                        ui.label("LOW MEMORY MODE is active for LOW-END profiles.");
                    }
                }
                ui.horizontal(|ui| {
                    if ui.button("CANCEL").clicked() {
                        self.start_training_dialog = false;
                    }
                    let dataset_valid = self
                        .core
                        .dataset
                        .as_ref()
                        .and_then(|d| d.report.as_ref())
                        .is_some_and(|r| r.errors == 0);
                    let can_start = !self.core.safe_mode
                        && self.core.model.is_some()
                        && self.core.dataset.is_some()
                        && self.core.tokenizer_path.is_some()
                        && dataset_valid
                        && self.core.worker.is_none();
                    if ui
                        .add_enabled(can_start, egui::Button::new("START"))
                        .clicked()
                    {
                        self.start_training_dialog = false;
                        self.core.start_training();
                    }
                });
            });
        if !open {
            self.start_training_dialog = false;
        }
    }

    fn update_tokenizer_cache(&mut self) {
        let key = self
            .core
            .tokenizer_path
            .as_ref()
            .map(|p| p.display().to_string());
        let current = self.tokenizer_vocab.as_ref().map(|(path, _)| path.clone());
        if key == current {
            return;
        }
        self.tokenizer_vocab = key.and_then(|path| {
            crate::tokenizer::Tokenizer::load(&path)
                .ok()
                .map(|tokenizer| (path, tokenizer.vocab_size()))
        });
    }

    fn current_log_contents(&self) -> String {
        let file = match self.selected_log.as_str() {
            "TRAINING LOG" => "training.log",
            "INFERENCE LOG" => "inference.log",
            "WEB LOG" => "web.log",
            "ROUTER LOG" => "router.log",
            "CRASH LOG" => "crash.log",
            _ => "app.log",
        };
        self.core.logger.recent(file, 350)
    }

    fn recovery_overlay(&mut self, ctx: &egui::Context) {
        if self.core.previous_crash {
            egui::Window::new("Previous crash detected")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label("The previous process did not exit cleanly.");
                    ui.add_space(6.0);
                    ui.label("Choose how to recover without entering a crash loop.");
                    ui.horizontal(|ui| {
                        if ui.button("START NORMALLY").clicked() {
                            self.core.safe_mode = false;
                            self.core.previous_crash = false;
                            self.core.refresh_model();
                            self.core.refresh_tokenizer();
                            self.core.refresh_dataset();
                            self.core
                                .log_event("Normal startup selected after previous crash.");
                        }
                        if ui.button("SAFE MODE").clicked() {
                            self.core.safe_mode = true;
                            self.core.previous_crash = false;
                            self.core.worker = None;
                            self.core.model = None;
                            self.core.tokenizer_path = None;
                            self.core
                                .log_event("Safe Mode selected after previous crash.");
                        }
                    });
                    ui.label(
                        "Safe Mode disables model auto-load, training and heavy background jobs.",
                    );
                    if ui.button("OPEN DIAGNOSTICS").clicked() {
                        self.core.command_page(AppPage::System);
                    }
                    if ui.button("VIEW CRASH LOG").clicked() {
                        self.core.command_page(AppPage::Logs);
                    }
                });
        }

        if !self.core.previous_crash && self.core.recovery_session.is_some() {
            egui::Window::new("Training recovery")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    if let Some(recovery) = &self.core.recovery_session {
                        ui.colored_label(Color32::YELLOW, "Previous training session found.");
                        ui.label(format!("Run: {}", recovery.run_id));
                        ui.label(format!("Step: {}", recovery.step));
                        ui.label(format!("Epoch: {}", recovery.epoch));
                        ui.label(format!("Checkpoint: {}", recovery.checkpoint.display()));
                        ui.horizontal(|ui| {
                            if ui.button("RESUME").clicked() {
                                self.core.resume_training();
                            }
                            if ui.button("START NEW").clicked() {
                                self.core.discard_recovery_session();
                            }
                            if ui.button("DISCARD SESSION").clicked() {
                                self.core.discard_recovery_session();
                            }
                        });
                    }
                });
        }
    }

    fn wizard(&mut self, ctx: &egui::Context) {
        if !self.core.wizard_open {
            return;
        }
        egui::Window::new("Welcome to Ai")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label(RichText::new("Own Neural Engine").size(22.0).strong());
                ui.label("Five small steps to a working local training session.");
                ui.add_space(10.0);
                match self.core.wizard_step {
                    0 => {
                        ui.label("1 / 5  SYSTEM CHECK");
                        ui.label(format!("CPU: {}", self.core.resources.cpu_name));
                        ui.label(format!(
                            "RAM available: {}",
                            AppCore::format_mb(self.core.resources.ram_available_bytes)
                        ));
                    }
                    1 => {
                        ui.label("2 / 5  STORAGE SETUP — data directory is ready.");
                    }
                    2 => {
                        ui.label("3 / 5  CREATE MODEL");
                        if ui.button("OPEN MODEL").clicked() {
                            self.core.command_page(AppPage::Model);
                        }
                    }
                    3 => {
                        ui.label("4 / 5  WEB LEARNING");
                        ui.label("No manual dataset is required for the default path.");
                        if ui.button("OPEN WEB LEARNING").clicked() {
                            self.core.command_page(AppPage::WebLearning);
                        }
                    }
                    _ => {
                        ui.label("5 / 5  START AUTONOMOUS LEARNING");
                        ui.label("Create the random model, then start Web Learning.");
                        if ui.button("START WEB LEARNING").clicked() {
                            self.core.command_page(AppPage::WebLearning);
                        }
                    }
                }
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if self.core.wizard_step > 0 && ui.button("BACK").clicked() {
                        self.core.wizard_step -= 1;
                    }
                    if self.core.wizard_step < 4 {
                        if ui.button("NEXT").clicked() {
                            self.core.wizard_step += 1;
                        }
                    } else if ui.button("FINISH").clicked() {
                        self.core.wizard_open = false;
                        self.core.config.first_start = false;
                        self.core.save_config();
                    }
                });
            });
    }

    fn self_test_window(&mut self, ctx: &egui::Context) {
        let Some(results) = self.core.self_test_result.clone() else {
            return;
        };
        egui::Window::new("Self Test")
            .resizable(true)
            .show(ctx, |ui| {
                let passed = results.iter().filter(|r| r.ok).count();
                ui.label(format!("{passed}/{} checks passed", results.len()));
                for result in results {
                    let color = if result.ok {
                        Color32::from_rgb(90, 220, 120)
                    } else {
                        Color32::RED
                    };
                    ui.colored_label(
                        color,
                        format!(
                            "[{}] {} — {}",
                            if result.ok { "PASS" } else { "FAIL" },
                            result.name,
                            result.detail
                        ),
                    );
                }
                if ui.button("CLOSE").clicked() {
                    self.core.self_test_result = None;
                }
            });
    }
}

impl eframe::App for AiApplication {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.core.refresh();
        self.core
            .update_training_session_marker(&mut self.last_finalized_run);

        ctx.set_visuals(egui::Visuals::dark());
        self.nav(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            self.header(ui);
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| match self.core.page {
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
                    AppPage::WebLearning => self.web_learning(ui),
                });
        });

        self.recovery_overlay(ctx);
        self.wizard(ctx);
        self.start_training_dialog(ctx);
        self.self_test_window(ctx);

        if self.core.tokenizer_job_rx.is_some() {
            egui::Area::new("tokenizer-job".into())
                .anchor(Align2::CENTER_BOTTOM, [0.0, -22.0])
                .show(ctx, |ui| {
                    card(ui, |ui| {
                        ui.label("Tokenizer training is running in a background worker.");
                    });
                });
        }

        if let Some(error) = self.core.last_error.clone() {
            egui::TopBottomPanel::bottom("error-banner").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(Color32::RED, "ERROR");
                    ui.label(error);
                    if ui.button("DISMISS").clicked() {
                        self.core.last_error = None;
                    }
                    if ui.button("OPEN LOGS").clicked() {
                        self.core.command_page(AppPage::Logs);
                    }
                });
            });
        }

        let repaint = if self.core.is_trainer_running() {
            100
        } else {
            self.core.config.ui_update_ms.clamp(250, 1000)
        };
        ctx.request_repaint_after(Duration::from_millis(repaint));
    }
}

fn status_line(ui: &mut Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(key).small());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(value);
        });
    });
}

fn metric_card(ui: &mut Ui, title: &str, value: impl Into<String>, detail: impl Into<String>) {
    card_with_size(ui, [240.0, 96.0], |ui| {
        ui.label(RichText::new(title).small());
        ui.label(RichText::new(value.into()).size(21.0).strong());
        ui.label(RichText::new(detail.into()).small());
    });
}

fn card(ui: &mut Ui, contents: impl FnOnce(&mut Ui)) {
    egui::Frame::group(ui.style())
        .inner_margin(12.0)
        .show(ui, contents);
}

fn card_with_size(ui: &mut Ui, size: [f32; 2], contents: impl FnOnce(&mut Ui)) {
    egui::Frame::group(ui.style())
        .inner_margin(12.0)
        .show(ui, |ui| {
            ui.set_min_size(Vec2::new(size[0], size[1]));
            contents(ui);
        });
}

fn status_pill(ui: &mut Ui, value: &str) {
    let color = match value {
        "RUNNING" => Color32::from_rgb(90, 220, 120),
        "FAILED" => Color32::RED,
        "PAUSED" => Color32::YELLOW,
        "STOPPING" | "PAUSING" | "SAVING" => Color32::YELLOW,
        _ => ui.visuals().text_color(),
    };
    ui.label(RichText::new(value).monospace().strong().color(color));
}

fn row_value(ui: &mut Ui, key: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(key).small());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(value);
        });
    });
}

fn text_field(ui: &mut Ui, label: &str, value: &mut String) {
    ui.label(label);
    ui.text_edit_singleline(value);
    ui.end_row();
}

fn subsystem(ui: &mut Ui, name: &str, ok: bool, detail: &str) {
    let color = if ok {
        Color32::from_rgb(90, 220, 120)
    } else {
        Color32::RED
    };
    ui.horizontal(|ui| {
        ui.colored_label(color, if ok { "OK" } else { "ERROR" });
        ui.label(RichText::new(name).strong());
        ui.label(detail);
    });
}

fn error_card(ui: &mut Ui, error: &str) {
    card(ui, |ui| {
        ui.colored_label(Color32::RED, "ERROR");
        ui.label(error);
    });
}

fn empty_dash(value: &str) -> &str {
    if value.is_empty() {
        "—"
    } else {
        value
    }
}

fn show_model_info(ui: &mut Ui, model: &ModelInfo, status: &str) {
    let name = model
        .config
        .as_ref()
        .map(|c| c.model_id.as_str())
        .unwrap_or("unknown");
    row_value(ui, "Name", name);
    row_value(
        ui,
        "Architecture",
        model
            .config
            .as_ref()
            .map(|c| c.architecture.as_str())
            .unwrap_or("unknown"),
    );
    row_value(ui, "Status", status);
    row_value(ui, "Parameters", &model.parameter_count.to_string());
    row_value(
        ui,
        "Embedding",
        &model
            .config
            .as_ref()
            .map(|c| c.embedding_dim.to_string())
            .unwrap_or_else(|| "—".into()),
    );
    row_value(
        ui,
        "Hidden",
        &model
            .config
            .as_ref()
            .map(|c| c.hidden_dim.to_string())
            .unwrap_or_else(|| "—".into()),
    );
    row_value(
        ui,
        "Layers",
        &model
            .config
            .as_ref()
            .map(|c| c.layer_count.to_string())
            .unwrap_or_else(|| "—".into()),
    );
    row_value(
        ui,
        "Vocabulary",
        &model
            .config
            .as_ref()
            .map(|c| c.vocab_size.to_string())
            .unwrap_or_else(|| "—".into()),
    );
    row_value(
        ui,
        "Sequence",
        &model
            .config
            .as_ref()
            .map(|c| c.sequence_length.to_string())
            .unwrap_or_else(|| "—".into()),
    );
    row_value(
        ui,
        "Seed",
        &model
            .config
            .as_ref()
            .map(|c| c.seed.to_string())
            .unwrap_or_else(|| "—".into()),
    );
    row_value(ui, "File size", &AppCore::format_mb(model.file_size));
    row_value(ui, "Checksum", &format!("{:016x}", model.checksum));
    row_value(ui, "Loaded from previous", if model.loaded_from_previous { "YES" } else { "NO" });
}

fn model_summary(model: Option<&ModelInfo>) -> String {
    model
        .and_then(|m| {
            m.config.as_ref().map(|c| {
                format!(
                    "{} • {} params • {:016x}",
                    c.architecture, m.parameter_count, m.checksum
                )
            })
        })
        .unwrap_or_else(|| "No model loaded".into())
}

fn dataset_status(dataset: Option<&super::core::DatasetInfo>) -> String {
    dataset
        .map(|d| {
            if d.report.as_ref().is_some_and(|r| r.errors == 0) {
                "VALID".into()
            } else {
                "SELECTED".into()
            }
        })
        .unwrap_or_else(|| "NONE".into())
}

fn dataset_summary(dataset: Option<&super::core::DatasetInfo>) -> String {
    match dataset {
        Some(d) => {
            let name = d
                .path
                .file_name()
                .and_then(|v| v.to_str())
                .unwrap_or("dataset");
            match &d.report {
                Some(r) => format!("{name} • {} samples", r.samples),
                None => format!("{name} • validation required"),
            }
        }
        None => "No dataset".into(),
    }
}

fn loss_string(value: Option<f32>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "—".into())
}

fn steps_per_second(snapshot: &super::core::TrainingSnapshot) -> String {
    if snapshot.elapsed_seconds <= 0.0 {
        "0.00".into()
    } else {
        format!(
            "{:.2}",
            snapshot.step as f64 / snapshot.elapsed_seconds.max(0.001)
        )
    }
}

fn format_eta(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    format!(
        "{}h {:02}m {:02}s",
        total / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

fn estimate_parameters(vocab: usize, embedding: usize, hidden: usize, layers: usize) -> usize {
    vocab
        .saturating_mul(embedding)
        .saturating_add(hidden.saturating_mul(embedding).saturating_add(hidden))
        .saturating_add(
            layers.saturating_mul(
                hidden
                    .saturating_mul(hidden)
                    .saturating_mul(7)
                    .saturating_add(hidden.saturating_mul(4)),
            ),
        )
        .saturating_add(vocab.saturating_mul(hidden).saturating_add(vocab))
}

fn estimate_training_memory(
    parameters: usize,
    hidden: usize,
    layers: usize,
    sequence: usize,
) -> usize {
    let weights = parameters.saturating_mul(4);
    let optimizer = weights.saturating_mul(2);
    let gradients = weights;
    let bptt = sequence
        .saturating_mul(layers)
        .saturating_mul(hidden)
        .saturating_mul(7)
        .saturating_mul(4);
    weights
        .saturating_add(optimizer)
        .saturating_add(gradients)
        .saturating_add(bptt.saturating_mul(2))
        .saturating_add(4 * 1024 * 1024)
}

fn draw_loss_graph(ui: &mut Ui, points: &VecDeque<(u64, f32)>) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 270.0), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_stroke(
        rect,
        7.0,
        Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
        StrokeKind::Outside,
    );

    let sampled = downsample_points(points, 320);
    if sampled.len() < 2 {
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            "Waiting for real training metrics…",
            FontId::proportional(16.0),
            ui.visuals().text_color(),
        );
        return;
    }

    let min_loss = sampled
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::INFINITY, f32::min);
    let max_loss = sampled
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = (max_loss - min_loss).max(1e-6);
    let min_step = sampled.first().map(|p| p.0).unwrap_or(0);
    let max_step = sampled
        .last()
        .map(|p| p.0)
        .unwrap_or(min_step + 1)
        .max(min_step + 1);
    let inner = rect.shrink(22.0);

    let mut previous = None;
    for (step, loss) in sampled {
        let x = inner.left()
            + ((step.saturating_sub(min_step)) as f32 / (max_step - min_step) as f32)
                * inner.width();
        let y = inner.bottom() - ((loss - min_loss) / range) * inner.height();
        let current = egui::pos2(x, y);
        if let Some(previous) = previous {
            painter.line_segment(
                [previous, current],
                Stroke::new(2.0_f32, Color32::from_rgb(120, 180, 255)),
            );
        }
        previous = Some(current);
    }

    painter.text(
        inner.left_top(),
        Align2::LEFT_TOP,
        format!("max {:.4}", max_loss),
        FontId::monospace(11.0),
        ui.visuals().weak_text_color(),
    );
    painter.text(
        inner.left_bottom(),
        Align2::LEFT_BOTTOM,
        format!("min {:.4}", min_loss),
        FontId::monospace(11.0),
        ui.visuals().weak_text_color(),
    );
}

fn downsample_points(points: &VecDeque<(u64, f32)>, limit: usize) -> Vec<(u64, f32)> {
    if points.len() <= limit {
        return points.iter().copied().collect();
    }

    let recent_count = limit.min(1000) / 2;
    let old_count = limit.saturating_sub(recent_count).max(1);
    let split = points.len().saturating_sub(recent_count);
    let old: Vec<_> = points.iter().take(split).copied().collect();
    let recent: Vec<_> = points.iter().skip(split).copied().collect();
    let mut output = Vec::with_capacity(limit);

    let bucket_size = old.len().div_ceil(old_count);
    for bucket in old.chunks(bucket_size.max(1)).take(old_count) {
        if bucket.is_empty() {
            continue;
        }
        let step = bucket[bucket.len() / 2].0;
        let mean = bucket.iter().map(|(_, loss)| *loss as f64).sum::<f64>() / bucket.len() as f64;
        output.push((step, mean as f32));
    }
    output.extend(recent);
    output
}

fn draw_neural_activity(ui: &mut Ui, layers: &[super::core::LayerStats]) {
    let max_activation = layers
        .iter()
        .map(|l| l.activation_min.abs().max(l.activation_max.abs()))
        .fold(1e-6, f32::max);
    let max_gradient = layers
        .iter()
        .map(|l| l.gradient_norm.abs())
        .fold(1e-6, f32::max);

    for layer in layers {
        let width = ui.available_width();
        let height = 72.0;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_stroke(
            rect,
            5.0,
            Stroke::new(1.0_f32, ui.visuals().widgets.noninteractive.bg_stroke.color),
            StrokeKind::Outside,
        );

        let activation_level = (layer.activation_min.abs().max(layer.activation_max.abs())
            / max_activation)
            .clamp(0.0, 1.0);
        let gradient_level = (layer.gradient_norm / max_gradient).clamp(0.0, 1.0);

        let activation_rect = egui::Rect::from_min_max(
            rect.left_top() + Vec2::new(8.0, 30.0),
            egui::pos2(
                rect.left() + 8.0 + (rect.width() - 16.0) * activation_level,
                rect.top() + 42.0,
            ),
        );
        let gradient_rect = egui::Rect::from_min_max(
            rect.left_top() + Vec2::new(8.0, 52.0),
            egui::pos2(
                rect.left() + 8.0 + (rect.width() - 16.0) * gradient_level,
                rect.top() + 64.0,
            ),
        );
        painter.rect_filled(activation_rect, 3.0, Color32::from_rgb(120, 180, 255));
        painter.rect_filled(gradient_rect, 3.0, Color32::from_rgb(220, 150, 90));
        painter.text(
            rect.left_top() + Vec2::new(8.0, 6.0),
            Align2::LEFT_TOP,
            format!("AiCell {}", layer.layer),
            FontId::proportional(13.0),
            ui.visuals().text_color(),
        );
        painter.text(
            rect.left_top() + Vec2::new(8.0, 27.0),
            Align2::LEFT_TOP,
            format!(
                "ACT {:.4} → mean {:.4}  min {:.4}  max {:.4}",
                activation_level, layer.activation_mean, layer.activation_min, layer.activation_max
            ),
            FontId::monospace(10.0),
            ui.visuals().weak_text_color(),
        );
        painter.text(
            rect.left_top() + Vec2::new(8.0, 49.0),
            Align2::LEFT_TOP,
            format!(
                "GRAD {:.4}  MEM {:.4}  WEIGHT {:.4}",
                layer.gradient_norm, layer.memory_norm, layer.weight_norm
            ),
            FontId::monospace(10.0),
            ui.visuals().weak_text_color(),
        );
    }
}
