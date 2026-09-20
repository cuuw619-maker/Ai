use eframe::egui::{self, Align, Align2, Layout, RichText};
use std::path::{Path, PathBuf};

pub fn show_startup_error(details: &str, log_path: &Path) {
    let details = details.to_owned();
    let log_path = log_path.to_path_buf();
    let fallback_details = details.clone();
    let fallback_path = log_path.clone();

    let viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size([720.0, 480.0])
        .with_min_inner_size([620.0, 400.0])
        .with_title("Ai failed to start");
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    let result = eframe::run_native(
        "Ai failed to start",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(StartupErrorApp {
                details,
                log_path,
            }))
        }),
    );

    if result.is_err() {
        let description = format!(
            "Ai failed to start.\n\nError:\n{}\n\nLog:\n{}\n\nFallback actions:\n[OPEN LOG] use Explorer\n[COPY DETAILS] copy this message",
            fallback_details,
            fallback_path.display()
        );
        rfd::MessageDialog::new()
            .set_title("Ai failed to start")
            .set_description(&description)
            .set_buttons(rfd::MessageButtons::Ok)
            .set_level(rfd::MessageLevel::Error)
            .show();
    }
}

struct StartupErrorApp {
    details: String,
    log_path: PathBuf,
}

impl eframe::App for StartupErrorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.with_layout(Layout::top_down(Align::LEFT), |ui| {
                ui.add_space(12.0);
                ui.heading(RichText::new("Ai failed to start").strong());
                ui.colored_label(
                    egui::Color32::RED,
                    "The application stopped before the main laboratory UI could be initialized.",
                );
                ui.add_space(8.0);
                ui.label(RichText::new("Error").strong());
                egui::ScrollArea::vertical()
                    .max_height(230.0)
                    .show(ui, |ui| {
                        ui.monospace(&self.details);
                    });
                ui.separator();
                ui.label(format!("Crash log: {}", self.log_path.display()));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("CLOSE").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if ui.button("COPY DETAILS").clicked() {
                        let text = format!(
                            "Ai failed to start\n\n{}\n\nLog: {}",
                            self.details,
                            self.log_path.display()
                        );
                        ctx.copy_text(text);
                    }
                    if ui.button("OPEN LOG").clicked() {
                        open_log(&self.log_path);
                    }
                });
            });
        });
        ctx.request_repaint_after(std::time::Duration::from_millis(500));
    }
}

fn open_log(path: &Path) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("explorer")
            .args(["/select,", &path.display().to_string()])
            .spawn();
    }
    #[cfg(not(target_os = "windows"))]
    {
        let target = path.parent().unwrap_or_else(|| Path::new("."));
        let _ = std::process::Command::new("xdg-open").arg(target).spawn();
    }
}
