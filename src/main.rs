#![cfg_attr(windows, windows_subsystem = "windows")]

mod cli;

use ai::app::{
    install_panic_hook, show_startup_error, AiApplication, AppCore, CrashContext, Logger,
};
use std::path::PathBuf;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

fn main() {
    let root = app_data_root();
    let log_dir = root.join("logs");
    let logger = match Logger::new(&log_dir) {
        Ok(v) => v,
        Err(error) => {
            show_startup_error(&error, &log_dir.join("crash.log"));
            return;
        }
    };
    let crash_context = Arc::new(Mutex::new(CrashContext::default()));
    install_panic_hook(logger.clone(), Arc::clone(&crash_context));

    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|value| value == "--cli") {
        if let Err(error) = cli::run(args.into_iter().skip(1).collect()) {
            logger.app(format!("CLI error: {error}"));
            rfd::MessageDialog::new()
                .set_title("Ai CLI")
                .set_description(&error)
                .set_buttons(rfd::MessageButtons::Ok)
                .set_level(rfd::MessageLevel::Error)
                .show();
        }
        return;
    }

    let core = match AppCore::bootstrap(&root, logger.clone(), crash_context) {
        Ok(core) => core,
        Err(error) => {
            logger.app(format!("startup failure: {error}"));
            logger.crash(format!("STARTUP ERROR\nmessage={error}"));
            show_startup_error(&error, &log_dir.join("crash.log"));
            return;
        }
    };

    let mut viewport = eframe::egui::ViewportBuilder::default()
        .with_inner_size([core.config.window_width, core.config.window_height])
        .with_min_inner_size([980.0, 640.0])
        .with_title("Ai — Own Neural Engine");
    if let (Some(x), Some(y)) = (core.config.window_x, core.config.window_y) {
        viewport = viewport.with_position([x, y]);
    }

    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        eframe::run_native(
            "Ai — Own Neural Engine",
            options,
            Box::new(move |_cc| Ok(Box::new(AiApplication::new(core)))),
        )
    }));

    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            logger.app(format!("eframe terminated with error: {error}"));
            logger.crash(format!("UI STARTUP/SHUTDOWN ERROR\nmessage={error}"));
            show_startup_error(&error.to_string(), &log_dir.join("crash.log"));
        }
        Err(_) => {
            let details = format!(
                "Unhandled panic reached the application boundary.\n                 See data/logs/crash.log for the panic payload and backtrace.\n                 Application version: {}\n                 Architecture: {}",
                env!("CARGO_PKG_VERSION"),
                std::env::consts::ARCH
            );
            logger.app("unhandled panic recovered at native application boundary");
            logger.crash(&details);
            show_startup_error(&details, &log_dir.join("crash.log"));
        }
    }
}

fn app_data_root() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("data")))
        .unwrap_or_else(|| PathBuf::from("data"))
}
