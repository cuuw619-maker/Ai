mod config;
mod core;
mod ext;
mod logging;
mod resources;
mod startup;
mod ui_next;

pub use config::{AppConfig, TrainingUiConfig};
pub use core::{
    AppCore, AppPage, ChatMessage, DatasetInfo, LayerStats, ModelInfo, ModelStatsSnapshot,
    SelfTestResult, TrainingRecovery, TrainingSnapshot,
};
pub use logging::{install_panic_hook, CrashContext, Logger, RuntimeGuard};
pub use resources::{ResourceMonitor, ResourceSnapshot};
pub use startup::show_startup_error;
pub use ui_next::AiApplication;
