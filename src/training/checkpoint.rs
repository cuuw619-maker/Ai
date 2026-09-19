use super::config::TrainingConfig;
use super::state::{TrainingState, TrainingStatus};
use crate::dataset::DatasetCursor;
use crate::model::AiNet;
use crate::optimizer::AdamWState;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"AICHPv01";
const VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct CheckpointData {
    pub run_id: String,
    pub timestamp_unix_ms: u64,
    pub model_checksum: u64,
    pub dataset_id: String,
    pub tokenizer_id: String,
    pub state: TrainingState,
    pub config: TrainingConfig,
    pub cursor: DatasetCursor,
    pub optimizer: AdamWState,
    pub random_state: u64,
    pub model_bytes: Vec<u8>,
}

pub struct Checkpoint;

impl Checkpoint {
    pub fn save_latest(
        path: &Path,
        run_id: &str,
        model: &AiNet,
        optimizer: &AdamWState,
        dataset_id: &str,
        tokenizer_id: &str,
        state: &TrainingState,
        config: &TrainingConfig,
        random_state: u64,
        timestamp_unix_ms: u64,
    ) -> Result<(), String> {
        let data = CheckpointData {
            run_id: run_id.into(),
            timestamp_unix_ms,
            model_checksum: model.weights_checksum(),
            dataset_id: dataset_id.into(),
            tokenizer_id: tokenizer_id.into(),
            state: state.clone(),
            config: config.clone(),
            cursor: state.cursor.clone(),
            optimizer: optimizer.clone(),
            random_state,
            model_bytes: model.to_aimodel_bytes()?,
        };
        atomic_save(path, &encode(&data)?)
    }

    pub fn load(path: &Path) -> Result<CheckpointData, String> {
        let bytes = fs::read(path).map_err(|e| format!("read checkpoint {}: {e}", path.display()))?;
        decode_container(&bytes)
    }

    pub fn load_latest_or_previous(path: &Path) -> Result<(CheckpointData, PathBuf), String> {
        match Self::load(path) {
            Ok(data) => Ok((data, path.to_path_buf())),
            Err(primary) => {
                let previous = path.with_extension("aicheckpoint.prev");
                match Self::load(&previous) {
                    Ok(data) => Ok((data, previous)),
                    Err(fallback) => Err(format!("latest checkpoint invalid: {primary}; previous invalid: {fallback}")),
                }
            }
        }
    }

    pub fn list(dir: &Path) -> Result<Vec<PathBuf>, String> {
        let mut files = Vec::new();
        if !dir.exists() { return Ok(files); }
        for entry in fs::read_dir(dir).map_err(|e| format!("read checkpoint directory: {e}"))? {
            let path = entry.map_err(|e| format!("read checkpoint entry: {e}"))?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("aicheckpoint") {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }
}

fn encode(data: &CheckpointData) -> Result<Vec<u8>, String> {
    let mut w = Writer::default();
    w.string(&data.run_id);
    w.u64(data.timestamp_unix_ms);
    w.u64(data.model_checksum);
    w.string(&data.dataset_id);
    w.string(&data.tokenizer_id);
    w.string(&data.state.run_id);
    w.u8(status_byte(data.state.status));
    w.u64(data.state.epoch);
    w.u64(data.state.step);
    w.u64(data.state.tokens_seen);
    w.u64(data.state.tokens_this_run);
    w.u64(data.state.tokens_this_epoch);
    w.string(&data.cursor.dataset_id);
    w.string(&data.cursor.file_path);
    w.u64(data.cursor.file_offset);
    w.u64(data.cursor.sample_index);
    w.u64(data.cursor.token_position);
    write_config(&mut w, &data.config);
    w.u64(data.random_state);
    write_optimizer(&mut w, &data.optimizer);
    w.u64(data.model_bytes.len() as u64);
    w.bytes.extend_from_slice(&data.model_bytes);
    let checksum = fnv1a64(&w.bytes);
    let mut out = Vec::with_capacity(28 + w.bytes.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&checksum.to_le_bytes());
    out.extend_from_slice(&(w.bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&w.bytes);
    Ok(out)
}

fn decode_container(bytes: &[u8]) -> Result<CheckpointData, String> {
    if bytes.len() < 28 || &bytes[..8] != MAGIC { return Err("invalid .aicheckpoint header".into()); }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION { return Err(format!("unsupported .aicheckpoint version {version}")); }
    let expected = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let len = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
    if bytes.len() != 28 + len { return Err("invalid .aicheckpoint payload length".into()); }
    let payload = &bytes[28..];
    if fnv1a64(payload) != expected { return Err("checkpoint checksum mismatch".into()); }

    let mut r = Reader::new(payload);
    let run_id = r.string()?;
    let timestamp_unix_ms = r.u64()?;
    let model_checksum = r.u64()?;
    let dataset_id = r.string()?;
    let tokenizer_id = r.string()?;
    let state_run_id = r.string()?;
    let status = status_from_byte(r.u8()?)?;
    let epoch = r.u64()?;
    let step = r.u64()?;
    let tokens_seen = r.u64()?;
    let tokens_this_run = r.u64()?;
    let tokens_this_epoch = r.u64()?;
    let cursor = DatasetCursor {
        dataset_id: r.string()?,
        file_path: r.string()?,
        file_offset: r.u64()?,
        sample_index: r.u64()?,
        token_position: r.u64()?,
    };
    let config = read_config(&mut r)?;
    let random_state = r.u64()?;
    let optimizer = read_optimizer(&mut r)?;
    let model_len = r.u64()? as usize;
    let model_bytes = r.take(model_len)?.to_vec();
    if !r.finished() { return Err("trailing bytes in checkpoint".into()); }

    Ok(CheckpointData {
        run_id,
        timestamp_unix_ms,
        model_checksum,
        dataset_id,
        tokenizer_id,
        state: TrainingState {
            status,
            run_id: state_run_id,
            epoch,
            step,
            tokens_seen,
            tokens_this_run,
            tokens_this_epoch,
            cursor: cursor.clone(),
            random_state,
        },
        config,
        cursor,
        optimizer,
        random_state,
        model_bytes,
    })
}

fn write_config(w: &mut Writer, c: &TrainingConfig) {
    w.u64(c.epochs);
    w.u64(c.sequence_length as u64);
    w.u64(c.micro_batch as u64);
    w.u64(c.gradient_accumulation as u64);
    w.f32(c.learning_rate);
    w.f32(c.weight_decay);
    w.f32(c.max_grad_norm);
    w.u64(c.checkpoint_interval_steps);
    w.u64(c.max_cpu_threads as u64);
    w.u64(c.memory_budget_mb as u64);
    w.u8(c.continuous as u8);
    match c.max_steps {
        Some(v) => { w.u8(1); w.u64(v); },
        None => w.u8(0),
    }
}

fn read_config(r: &mut Reader) -> Result<TrainingConfig, String> {
    let epochs = r.u64()?;
    let sequence_length = r.u64()? as usize;
    let micro_batch = r.u64()? as usize;
    let gradient_accumulation = r.u64()? as usize;
    let learning_rate = r.f32()?;
    let weight_decay = r.f32()?;
    let max_grad_norm = r.f32()?;
    let checkpoint_interval_steps = r.u64()?;
    let max_cpu_threads = r.u64()? as usize;
    let memory_budget_mb = r.u64()? as usize;
    let continuous = r.u8()? != 0;
    let max_steps = if r.u8()? != 0 { Some(r.u64()?) } else { None };
    Ok(TrainingConfig {
        epochs, sequence_length, micro_batch, gradient_accumulation,
        learning_rate, weight_decay, max_grad_norm, checkpoint_interval_steps,
        max_cpu_threads, memory_budget_mb, continuous, max_steps,
    })
}

fn write_optimizer(w: &mut Writer, o: &AdamWState) {
    w.f32(o.learning_rate);
    w.f32(o.weight_decay);
    w.f32(o.beta1);
    w.f32(o.beta2);
    w.f32(o.epsilon);
    w.u64(o.step);
    w.u64(o.parameter_names.len() as u64);
    for (i, name) in o.parameter_names.iter().enumerate() {
        w.string(name);
        w.u64(o.m[i].len() as u64);
        for v in &o.m[i] { w.f32(*v); }
        w.u64(o.v[i].len() as u64);
        for v in &o.v[i] { w.f32(*v); }
    }
}

fn read_optimizer(r: &mut Reader) -> Result<AdamWState, String> {
    let learning_rate = r.f32()?;
    let weight_decay = r.f32()?;
    let beta1 = r.f32()?;
    let beta2 = r.f32()?;
    let epsilon = r.f32()?;
    let step = r.u64()?;
    let count = r.u64()? as usize;
    let mut names = Vec::with_capacity(count);
    let mut m = Vec::with_capacity(count);
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        names.push(r.string()?);
        let ml = r.u64()? as usize;
        let mut mv = Vec::with_capacity(ml);
        for _ in 0..ml { mv.push(r.f32()?); }
        let vl = r.u64()? as usize;
        let mut vv = Vec::with_capacity(vl);
        for _ in 0..vl { vv.push(r.f32()?); }
        m.push(mv);
        v.push(vv);
    }
    Ok(AdamWState { learning_rate, weight_decay, beta1, beta2, epsilon, step, parameter_names: names, m, v })
}

fn status_byte(status: TrainingStatus) -> u8 {
    use TrainingStatus::*;
    match status { Idle => 0, Starting => 1, Running => 2, Pausing => 3, Paused => 4, Resuming => 5, Saving => 6, Completed => 7, Stopping => 8, Stopped => 9, Failed => 10 }
}

fn status_from_byte(value: u8) -> Result<TrainingStatus, String> {
    use TrainingStatus::*;
    Ok(match value { 0 => Idle, 1 => Starting, 2 => Running, 3 => Pausing, 4 => Paused, 5 => Resuming, 6 => Saving, 7 => Completed, 8 => Stopping, 9 => Stopped, 10 => Failed, _ => return Err(format!("invalid checkpoint status {value}")) })
}

fn atomic_save(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() { fs::create_dir_all(parent).map_err(|e| format!("create checkpoint directory: {e}"))?; }
    let temp = path.with_extension("aicheckpoint.tmp");
    let previous = path.with_extension("aicheckpoint.prev");
    {
        let mut file = File::create(&temp).map_err(|e| format!("create checkpoint temp: {e}"))?;
        file.write_all(bytes).map_err(|e| format!("write checkpoint temp: {e}"))?;
        file.flush().map_err(|e| format!("flush checkpoint temp: {e}"))?;
        file.sync_all().map_err(|e| format!("sync checkpoint temp: {e}"))?;
    }
    if path.exists() {
        if previous.exists() { fs::remove_file(&previous).map_err(|e| format!("remove previous checkpoint: {e}"))?; }
        fs::rename(path, &previous).map_err(|e| format!("rotate checkpoint: {e}"))?;
    }
    match fs::rename(&temp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            if previous.exists() && !path.exists() { let _ = fs::rename(&previous, path); }
            let _ = fs::remove_file(&temp);
            Err(format!("install checkpoint: {e}"))
        }
    }
}

#[derive(Default)]
struct Writer { bytes: Vec<u8> }
impl Writer {
    fn u8(&mut self, v: u8) { self.bytes.push(v); }
    fn u64(&mut self, v: u64) { self.bytes.extend_from_slice(&v.to_le_bytes()); }
    fn f32(&mut self, v: f32) { self.bytes.extend_from_slice(&v.to_le_bytes()); }
    fn string(&mut self, v: &str) { self.u64(v.len() as u64); self.bytes.extend_from_slice(v.as_bytes()); }
}
struct Reader<'a> { data: &'a [u8], pos: usize }
impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() { return Err("truncated checkpoint".into()); }
        let out = &self.data[self.pos..self.pos+n]; self.pos += n; Ok(out)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn f32(&mut self) -> Result<f32, String> { Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|e| format!("invalid checkpoint UTF-8: {e}"))
    }
    fn finished(&self) -> bool { self.pos == self.data.len() }
}
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data { hash ^= *byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    hash
}
