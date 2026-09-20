use ai::app::{AppConfig, AppCore, CrashContext, Logger};
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_root(label: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("ai-app-{label}-{}-{stamp}", std::process::id()))
}

#[test]
fn empty_environment_bootstraps_without_model_or_dataset() {
    let root = temp_root("empty");
    let logger = Logger::new(root.join("logs")).unwrap();
    let context = Arc::new(Mutex::new(CrashContext::default()));

    let mut app = AppCore::bootstrap(&root, logger, context).unwrap();
    assert!(app.model.is_none());
    assert!(app.dataset.is_none());
    assert!(!app.training.label().is_empty());
    app.mark_clean_shutdown();
    let _ = fs::remove_dir_all(root);
}

#[test]
fn corrupt_config_is_preserved_and_replaced_with_defaults() {
    let root = temp_root("config");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("config.toml"), "this = = not valid toml").unwrap();

    let (config, recovered) = AppConfig::load(&root).unwrap();

    assert!(recovered);
    assert_eq!(config.version, 1);
    assert!(root.join("config.toml").exists());
    assert!(root.join("config.toml.broken").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn stale_runtime_lock_is_reported_as_previous_crash() {
    let root = temp_root("recovery");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("runtime.lock"),
        "started_unix_ms=1
pid=999
version=test
",
    )
    .unwrap();

    let logger = Logger::new(root.join("logs")).unwrap();
    let context = Arc::new(Mutex::new(CrashContext::default()));
    let mut app = AppCore::bootstrap(&root, logger, context).unwrap();

    assert!(app.previous_crash);
    assert!(app.model.is_none());
    app.mark_clean_shutdown();
    let _ = fs::remove_dir_all(root);
}

#[test]
fn crash_marker_is_detected_on_startup() {
    let root = temp_root("marker");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("crash.marker"), "panic_unix_ms=1
version=test
").unwrap();

    let logger = Logger::new(root.join("logs")).unwrap();
    let context = Arc::new(Mutex::new(CrashContext::default()));
    let mut app = AppCore::bootstrap(&root, logger, context).unwrap();

    assert!(app.previous_crash);
    app.mark_clean_shutdown();
    let _ = fs::remove_dir_all(root);
}
