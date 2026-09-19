use super::{DatasetCursor, DatasetReader};
use crate::tokenizer::Tokenizer;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
struct TokenChunk {
    cursor: DatasetCursor,
    tokens: Vec<u32>,
    position: usize,
}

#[derive(Clone, Debug)]
pub struct TrainingSequence {
    pub input: Vec<usize>,
    pub target: Vec<usize>,
    pub cursor_before: DatasetCursor,
    pub cursor_after: DatasetCursor,
}

pub struct TrainingStream {
    reader: DatasetReader,
    tokenizer: Tokenizer,
    sequence_length: usize,
    dataset_id: String,
    chunks: VecDeque<TokenChunk>,
    resume_token_position: Option<u64>,
}

impl TrainingStream {
    pub fn new(reader: DatasetReader, tokenizer: Tokenizer, sequence_length: usize, dataset_id: String) -> Result<Self, String> {
        if sequence_length == 0 { return Err("sequence length must be non-zero".into()); }
        Ok(Self {
            reader,
            tokenizer,
            sequence_length,
            dataset_id,
            chunks: VecDeque::new(),
            resume_token_position: None,
        })
    }

    pub fn resume(
        reader: DatasetReader,
        tokenizer: Tokenizer,
        sequence_length: usize,
        dataset_id: String,
        cursor: &DatasetCursor,
    ) -> Result<Self, String> {
        if cursor.dataset_id != dataset_id {
            return Err("dataset cursor dataset_id mismatch".into());
        }
        let mut stream = Self::new(reader, tokenizer, sequence_length, dataset_id)?;
        stream.reader.seek_cursor(cursor)?;
        stream.resume_token_position = Some(cursor.token_position);
        Ok(stream)
    }

    fn fill(&mut self, required: usize) -> Result<(), String> {
        while self.available_tokens() < required {
            let Some(sample) = self.reader.next_sample().map_err(|e| e.to_string())? else {
                break;
            };
            let mut tokens: Vec<u32> = self.tokenizer.encode(&sample.text);
            tokens.push(self.tokenizer.eos_id());
            if tokens.is_empty() { continue; }
            let position = if let Some(skip) = self.resume_token_position.take() {
                (skip as usize).min(tokens.len())
            } else { 0 };
            self.chunks.push_back(TokenChunk {
                cursor: DatasetCursor {
                    dataset_id: self.dataset_id.clone(),
                    file_path: self.reader.path.to_string_lossy().into_owned(),
                    file_offset: sample.file_offset,
                    sample_index: sample.sample_index,
                    token_position: position as u64,
                },
                tokens,
                position,
            });
        }
        Ok(())
    }

    fn available_tokens(&self) -> usize {
        self.chunks.iter().map(|c| c.tokens.len().saturating_sub(c.position)).sum()
    }

    fn normalize(&mut self) {
        while self.chunks.front().is_some_and(|c| c.position >= c.tokens.len()) {
            self.chunks.pop_front();
        }
    }

    pub fn cursor(&mut self) -> Result<DatasetCursor, String> {
        self.normalize();
        if let Some(chunk) = self.chunks.front() {
            let mut cursor = chunk.cursor.clone();
            cursor.token_position = chunk.position as u64;
            return Ok(cursor);
        }
        Ok(DatasetCursor {
            dataset_id: self.dataset_id.clone(),
            file_path: self.reader.path.to_string_lossy().into_owned(),
            file_offset: self.reader.current_offset()?,
            sample_index: 0,
            token_position: 0,
        })
    }

    pub fn next_sequence(&mut self) -> Result<Option<TrainingSequence>, String> {
        self.fill(self.sequence_length + 1)?;
        if self.available_tokens() < self.sequence_length + 1 {
            return Ok(None);
        }
        let before = self.cursor()?;
        let mut ids = Vec::with_capacity(self.sequence_length + 1);
        for _ in 0..self.sequence_length + 1 {
            if self.chunks.front().is_none() {
                return Err("training stream internal buffer underflow".into());
            }
            let chunk = self.chunks.front_mut().unwrap();
            ids.push(chunk.tokens[chunk.position]);
            chunk.position += 1;
        }
        self.normalize();
        self.fill(1)?;
        let after = self.cursor()?;
        let input = ids[..self.sequence_length].iter().map(|v| *v as usize).collect();
        let target = ids[1..].iter().map(|v| *v as usize).collect();
        Ok(Some(TrainingSequence { input, target, cursor_before: before, cursor_after: after }))
    }
}
