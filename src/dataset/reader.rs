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
    csv_headers: Option<Vec<String>>,
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
            csv_headers: None,
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
        self.reader
            .stream_position()
            .map_err(|e| format!("dataset position: {e}"))
    }

    pub fn next_sample_index(&self) -> u64 {
        self.next_sample_index
    }

    pub fn next_sample(&mut self) -> AiResult<Option<RawSample>> {
        loop {
            let offset = self.current_offset().map_err(AiError::Dataset)?;
            let mut line = String::new();
            let read = self.reader.read_line(&mut line)?;
            if read == 0 {
                return Ok(None);
            }
            if matches!(self.format, DatasetFormat::Aicorpus)
                && self.next_sample_index == 0
                && line.trim_end().eq("# AiCorpus v1")
            {
                continue;
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
            DatasetFormat::Txt | DatasetFormat::Aicorpus => line.trim_end_matches(&['\r', '\n'][..]).to_owned(),
            DatasetFormat::Jsonl | DatasetFormat::Json => parse_jsonl(&line)?,
            DatasetFormat::Csv => self.parse_csv(&line)?,
        };
            Ok(Some(RawSample {
                text,
                file_offset: offset,
                sample_index: index,
            }))
        }
    }

    fn parse_csv(&mut self, line: &str) -> AiResult<String> {
        let values = parse_csv_record(line);
        if self.csv_headers.is_none() {
            self.csv_headers = Some(values.iter().map(|v| v.to_ascii_lowercase()).collect());
            return self.next_sample()?.map(|s| s.text).ok_or_else(|| AiError::Dataset("CSV contains header only".into()));
        }
        let headers = self.csv_headers.as_ref().unwrap();
        for key in ["text", "content", "body", "article", "document", "description"] {
            if let Some(index) = headers.iter().position(|v| v == key) {
                if let Some(value) = values.get(index) {
                    if !value.trim().is_empty() { return Ok(value.clone()); }
                }
            }
        }
        for (a,b) in [("question","answer"),("prompt","response"),("user","assistant")] {
            if let (Some(ai), Some(bi)) = (headers.iter().position(|v| v == a), headers.iter().position(|v| v == b)) {
                if let (Some(left), Some(right)) = (values.get(ai), values.get(bi)) {
                    return Ok(format!("{left}\n{right}"));
                }
            }
        }
        Err(AiError::Dataset("CSV row has no recognizable text fields".into()))
    }
}

fn parse_csv_record(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut chars = line.trim_end_matches(['\r', '\n']).chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => { current.push('"'); let _ = chars.next(); }
            '"' => quoted = !quoted,
            ',' if !quoted => { out.push(current.clone()); current.clear(); }
            _ => current.push(ch),
        }
    }
    out.push(current);
    out
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
    Err(AiError::Dataset(
        "JSONL record requires text or user/assistant fields".into(),
    ))
}
