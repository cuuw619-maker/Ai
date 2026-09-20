mod config;
mod core;
mod logging;
mod resources;
mod ui;

pub use config::{AppConfig, TrainingUiConfig};
pub use core::{AiPage, AppCore, AppPage, DatasetInfo, LayerStats, ModelInfo, ModelStatsSnapshot, SelfTestResult, TrainingRecovery, TrainingSnapshot};
pub use logging::{install_panic_hook, show_startup_error, CrashContext, Logger, RuntimeGuard};
pub use resources::{ResourceMonitor, ResourceSnapshot};
pub use ui::AiApplication;
