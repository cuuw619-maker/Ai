use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct ResourceSnapshot {
    pub cpu_percent: f32,
    pub process_cpu_percent: f32,
    pub ram_used_bytes: u64,
    pub ram_available_bytes: u64,
    pub process_ram_bytes: u64,
    pub disk_used_bytes: u64,
    pub disk_total_bytes: u64,
    pub cpu_name: String,
    pub thread_count: Option<usize>,
}

pub struct ResourceMonitor {
    stop: Arc<AtomicBool>,
    pub events: Receiver<ResourceSnapshot>,
    join: Option<thread::JoinHandle<()>>,
}

impl ResourceMonitor {
    pub fn start(interval: Duration) -> Self {
        let (tx, rx) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_worker = Arc::clone(&stop);
        let join = thread::spawn(move || {
            let mut system = sysinfo::System::new_all();
            let pid = sysinfo::Pid::from_u32(std::process::id());
            while !stop_worker.load(Ordering::Relaxed) {
                system.refresh_all();
                let process = system.process(pid);
                let process_cpu_percent = process.map(|p| p.cpu_usage()).unwrap_or(0.0);
                let process_ram_bytes = process.map(|p| p.memory()).unwrap_or(0);
                let disks = sysinfo::Disks::new_with_refreshed_list();
                let mut disk_used_bytes = 0u64;
                let mut disk_total_bytes = 0u64;
                for disk in &disks {
                    disk_total_bytes = disk_total_bytes.saturating_add(disk.total_space());
                    disk_used_bytes = disk_used_bytes
                        .saturating_add(disk.total_space().saturating_sub(disk.available_space()));
                }
                let snapshot = ResourceSnapshot {
                    cpu_percent: system.global_cpu_usage(),
                    process_cpu_percent,
                    ram_used_bytes: system.used_memory(),
                    ram_available_bytes: system.available_memory(),
                    process_ram_bytes,
                    disk_used_bytes,
                    disk_total_bytes,
                    cpu_name: system
                        .cpus()
                        .first()
                        .map(|cpu| cpu.brand().to_string())
                        .unwrap_or_else(|| "Unavailable".into()),
                    thread_count: None,
                };
                let _ = tx.send(snapshot);
                thread::sleep(interval);
            }
        });
        Self {
            stop,
            events: rx,
            join: Some(join),
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for ResourceMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}
