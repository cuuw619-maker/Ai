use super::{cursor::DatasetCursor, format::DatasetFormat};
use crate::error::{AiError, AiResult};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const DEFAULT_MAX_LINE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug)]
pub struct RawSample {
    pub text: String,
    pub file_offset: u64,
    pub sample_index: u64,
}

pub struct DatasetReader {
    pub path: PathBuf,
    pub format: DatasetFormat,
    reader: BufReader<File>,
    next_sample_index: u64,
    pub max_line_bytes: usize,
}

impl DatasetReader {
    pub fn open(path: &Path, format: DatasetFormat) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open dataset {}: {e}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            format,
            reader: BufReader::new(file),
            next_sample_index: 0,
            max_line_bytes: DEFAULT_MAX_LINE_BYTES,
        })
    }

    pub fn seek_cursor(&mut self, cursor: &DatasetCursor) -> Result<(), String> {
        if cursor.file_path != self.path.to_string_lossy() {
            return Err("dataset cursor file path mismatch".into());
        }
        self.reader
            .seek(SeekFrom::Start(cursor.file_offset))
            .map_err(|e| format!("seek dataset cursor: {e}"))?;
        self.next_sample_index = cursor.sample_index;
        Ok(())
    }

    pub fn current_offset(&mut self) -> Result<u64, String> {
        self.reader.stream_position().map_err(|e| format!("dataset position: {e}"))
    }

    pub fn next_sample_index(&self) -> u64 {
        self.next_sample_index
    }

    pub fn next_sample(&mut self) -> AiResult<Option<RawSample>> {
        let offset = self.current_offset().map_err(AiError::Dataset)?;
        let mut line = String::new();
        let read = self.reader.read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        if read > self.max_line_bytes {
            return Err(AiError::Dataset(format!(
                "sample {} exceeds max line size {} bytes",
                self.next_sample_index, self.max_line_bytes
            )));
        }
        let index = self.next_sample_index;
        self.next_sample_index += 1;
        let text = match self.format {
            DatasetFormat::Txt => line.trim_end_matches(&['\r', '\n'][..]).to_owned(),
            DatasetFormat::Jsonl => parse_jsonl(&line)?,
        };
        Ok(Some(RawSample { text, file_offset: offset, sample_index: index }))
    }
}

fn parse_jsonl(line: &str) -> AiResult<String> {
    let value: Value = serde_json::from_str(line)?;
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return Ok(text.to_owned());
    }
    if let Some(user) = value.get("user").and_then(Value::as_str) {
        let assistant = value.get("assistant").and_then(Value::as_str).unwrap_or("");
        let system = value.get("system").and_then(Value::as_str);
        let mut out = String::new();
        if let Some(system) = system {
            out.push_str("<|system|>");
            out.push_str(system);
        }
        out.push_str("<|user|>");
        out.push_str(user);
        out.push_str("<|assistant|>");
        out.push_str(assistant);
        return Ok(out);
    }
    Err(AiError::Dataset("JSONL record requires text or user/assistant fields".into()))
}
