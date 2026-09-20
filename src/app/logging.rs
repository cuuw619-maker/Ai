use std::backtrace::Backtrace;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Default)]
pub struct CrashContext {
    pub application_state: String,
    pub training_state: String,
    pub model_id: String,
    pub dataset_id: String,
    pub last_training_step: u64,
    pub last_checkpoint: String,
    pub cpu: String,
    pub ram_used_mb: u64,
    pub ram_available_mb: u64,
}

#[derive(Clone)]
pub struct Logger {
    directory: PathBuf,
    lock: Arc<Mutex<()>>,
    max_bytes: u64,
}

impl Logger {
    pub fn new(directory: impl AsRef<Path>) -> Result<Self, String> {
        fs::create_dir_all(directory.as_ref())
            .map_err(|e| format!("create log directory: {e}"))?;
        Ok(Self {
            directory: directory.as_ref().to_path_buf(),
            lock: Arc::new(Mutex::new(())),
            max_bytes: 5 * 1024 * 1024,
        })
    }

    pub fn app(&self, message: impl AsRef<str>) {
        self.write("app.log", message.as_ref());
    }

    pub fn training(&self, message: impl AsRef<str>) {
        self.write("training.log", message.as_ref());
    }

    pub fn inference(&self, message: impl AsRef<str>) {
        self.write("inference.log", message.as_ref());
    }

    pub fn crash(&self, message: impl AsRef<str>) {
        self.write("crash.log", message.as_ref());
    }

    pub fn recent(&self, name: &str, max_lines: usize) -> String {
        let path = self.directory.join(name);
        let Ok(content) = fs::read_to_string(path) else {
            return String::new();
        };
        let lines: Vec<&str> = content.lines().collect();
        lines
            .iter()
            .rev()
            .take(max_lines)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    fn write(&self, name: &str, message: &str) {
        let _guard = self.lock.lock().ok();
        let path = self.directory.join(name);
        let _ = rotate_if_needed(&path, self.max_bytes);
        let timestamp = now_ms();
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "[{timestamp}] {message}");
            let _ = file.flush();
        }
    }
}

pub struct RuntimeGuard {
    lock_path: PathBuf,
    active: bool,
}

impl RuntimeGuard {
    pub fn acquire(root: &Path) -> Result<(Self, bool), String> {
        fs::create_dir_all(root).map_err(|e| format!("create runtime root: {e}"))?;
        let path = root.join("runtime.lock");
        let previous = path.exists();
        let body = format!(
            "started_unix_ms={}\npid={}\nversion={}\n",
            now_ms(),
            std::process::id(),
            env!("CARGO_PKG_VERSION")
        );
        fs::write(&path, body).map_err(|e| format!("write runtime lock: {e}"))?;
        Ok((Self { lock_path: path, active: true }, previous))
    }

    pub fn mark_clean(&mut self) {
        if self.active {
            let _ = fs::remove_file(&self.lock_path);
            self.active = false;
        }
    }
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        self.mark_clean();
    }
}

static PANIC_INSTALLED: OnceLock<()> = OnceLock::new();

pub fn install_panic_hook(logger: Logger, context: Arc<Mutex<CrashContext>>) {
    if PANIC_INSTALLED.set(()).is_err() {
        return;
    }
    std::panic::set_hook(Box::new(move |panic_info| {
        let payload = panic_info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| panic_info.payload().downcast_ref::<String>().map(String::as_str))
            .unwrap_or("unknown panic payload");
        let thread_name = thread::current().name().unwrap_or("unnamed").to_string();
        let snapshot = context.lock().map(|v| v.clone()).unwrap_or_default();
        let backtrace = Backtrace::force_capture();
        let text = format!(
            "PANIC\ntimestamp_unix_ms={}\nmessage={}\nthread={}\nversion={}\nos={}\narch={}\ncpu={}\nram_used_mb={}\nram_available_mb={}\napplication_state={}\ntraining_state={}\nmodel_id={}\ndataset_id={}\nlast_training_step={}\nlast_checkpoint={}\nbacktrace=\n{}",
            now_ms(),
            payload,
            thread_name,
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            snapshot.cpu,
            snapshot.ram_used_mb,
            snapshot.ram_available_mb,
            snapshot.application_state,
            snapshot.training_state,
            snapshot.model_id,
            snapshot.dataset_id,
            snapshot.last_training_step,
            snapshot.last_checkpoint,
            backtrace
        );
        logger.crash(text);
    }));
}

pub fn show_startup_error(details: &str, log_path: &Path) {
    let description = format!(
        "Ai failed to start.\n\nError:\n{details}\n\nLog:\n{}",
        log_path.display()
    );
    rfd::MessageDialog::new()
        .set_title("Ai failed to start")
        .set_description(&description)
        .set_buttons(rfd::MessageButtons::Ok)
        .set_level(rfd::MessageLevel::Error)
        .show();
}

fn rotate_if_needed(path: &Path, max_bytes: u64) -> Result<(), String> {
    let len = path.metadata().map(|m| m.len()).unwrap_or(0);
    if len < max_bytes {
        return Ok(());
    }
    let second = path.with_extension("log.2");
    let first = path.with_extension("log.1");
    let _ = fs::remove_file(&second);
    let _ = fs::rename(&first, &second);
    let _ = fs::rename(path, &first);
    Ok(())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
