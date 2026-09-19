use super::{DatasetFormat, DatasetReader};
use crate::tokenizer::Tokenizer;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DatasetMetadata {
    pub dataset_id: String,
    pub path: String,
    pub format: DatasetFormat,
    pub size_bytes: u64,
    pub modified_unix_ms: u128,
    pub content_hash: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ValidationReport {
    pub samples: u64,
    pub errors: u64,
    pub empty_samples: u64,
    pub bytes: u64,
    pub estimated_tokens: Option<u64>,
}

pub fn validate(path: &Path, format: DatasetFormat, tokenizer: Option<&Tokenizer>) -> Result<ValidationReport, String> {
    let mut reader = DatasetReader::open(path, format)?;
    let mut report = ValidationReport { samples: 0, errors: 0, empty_samples: 0, bytes: 0, estimated_tokens: tokenizer.map(|_| 0) };
    loop {
        match reader.next_sample() {
            Ok(Some(sample)) => {
                report.samples += 1;
                report.bytes += sample.text.len() as u64;
                if sample.text.trim().is_empty() { report.empty_samples += 1; }
                if let Some(tokenizer) = tokenizer {
                    report.estimated_tokens = report.estimated_tokens.map(|n| n + tokenizer.encode(&sample.text).len() as u64);
                }
            }
            Ok(None) => break,
            Err(_) => report.errors += 1,
        }
    }
    Ok(report)
}

pub fn dataset_metadata(path: &Path, format: DatasetFormat) -> Result<DatasetMetadata, String> {
    let meta = fs::metadata(path).map_err(|e| format!("dataset metadata: {e}"))?;
    let modified_unix_ms = meta.modified().ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis())
        .unwrap_or(0);

    let mut reader = BufReader::new(File::open(path).map_err(|e| format!("open dataset for hashing: {e}"))?);
    let mut buf = [0u8; 64 * 1024];
    let mut hash = 0xcbf29ce484222325u64;
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("read dataset for hash: {e}"))?;
        if n == 0 { break; }
        for byte in &buf[..n] {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    let path_string = path.to_string_lossy();
    let mut id_hash = hash;
    for byte in path_string.as_bytes() {
        id_hash ^= *byte as u64;
        id_hash = id_hash.wrapping_mul(0x100000001b3);
    }
    id_hash ^= meta.len();
    id_hash = id_hash.wrapping_mul(0x100000001b3);
    id_hash ^= modified_unix_ms as u64;
    let dataset_id = format!("dataset-{id_hash:016x}");

    Ok(DatasetMetadata {
        dataset_id,
        path: path_string.into_owned(),
        format,
        size_bytes: meta.len(),
        modified_unix_ms,
        content_hash: hash,
    })
}
