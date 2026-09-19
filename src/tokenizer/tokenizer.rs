use super::format;
use super::merge::Merge;
use super::vocab::{default_special_tokens, SpecialToken, BYTE_VOCAB_SIZE};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub special_tokens: Vec<SpecialToken>,
    pub merges: Vec<Merge>,
    pair_to_new: HashMap<(u32, u32), u32>,
    token_bytes: HashMap<u32, Vec<u8>>,
}

impl Tokenizer {
    pub fn new_default() -> Self {
        Self::from_parts(default_special_tokens(), Vec::new()).expect("default tokenizer")
    }

    pub fn from_parts(special_tokens: Vec<SpecialToken>, merges: Vec<Merge>) -> Result<Self, String> {
        let mut tokenizer = Self {
            special_tokens,
            merges,
            pair_to_new: HashMap::new(),
            token_bytes: HashMap::new(),
        };
        tokenizer.rebuild_and_validate()?;
        Ok(tokenizer)
    }

    pub fn vocab_size(&self) -> usize {
        BYTE_VOCAB_SIZE as usize + self.special_tokens.len() + self.merges.len()
    }

    pub fn tokenizer_id(&self) -> String {
        let payload = format::encode_payload(self).unwrap_or_default();
        format!("aitok-{:016x}", fnv1a64(&payload))
    }

    pub fn eos_id(&self) -> u32 {
        self.special_id("eos").expect("default tokenizer has eos")
    }

    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.special_tokens.iter().find(|t| t.name == name).map(|t| t.id)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut output = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let special = self
                .special_tokens
                .iter()
                .filter(|t| bytes[i..].starts_with(t.text.as_bytes()))
                .max_by_key(|t| t.text.len());
            if let Some(token) = special {
                output.push(token.id);
                i += token.text.len();
                continue;
            }
            let start = i;
            i += 1;
            while i < bytes.len()
                && !self.special_tokens.iter().any(|t| bytes[i..].starts_with(t.text.as_bytes()))
            {
                i += 1;
            }
            output.extend(self.encode_bytes(&bytes[start..i]));
        }
        output
    }

    fn encode_bytes(&self, bytes: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = bytes.iter().map(|v| *v as u32).collect();
        for merge in &self.merges {
            if ids.len() < 2 {
                break;
            }
            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0;
            let mut changed = false;
            while i < ids.len() {
                if i + 1 < ids.len()
                    && ids[i] == merge.left
                    && ids[i + 1] == merge.right
                {
                    out.push(merge.new_id);
                    i += 2;
                    changed = true;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            if changed {
                ids = out;
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(token) = self.special_tokens.iter().find(|t| t.id == *id) {
                bytes.extend_from_slice(token.text.as_bytes());
            } else if *id < BYTE_VOCAB_SIZE {
                bytes.push(*id as u8);
            } else if let Some(token_bytes) = self.token_bytes.get(id) {
                bytes.extend_from_slice(token_bytes);
            } else {
                return Err(format!("unknown token id {id}"));
            }
        }
        String::from_utf8(bytes).map_err(|e| format!("decoded token stream is not UTF-8: {e}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut ids = HashMap::new();
        for token in &self.special_tokens {
            if token.id < BYTE_VOCAB_SIZE {
                return Err(format!("special token {} overlaps byte vocabulary", token.name));
            }
            if ids.insert(token.id, token.name.clone()).is_some() {
                return Err(format!("duplicate special token id {}", token.id));
            }
        }
        let expected_start = BYTE_VOCAB_SIZE + self.special_tokens.len() as u32;
        let mut known = HashMap::<u32, Vec<u8>>::new();
        for id in 0..BYTE_VOCAB_SIZE {
            known.insert(id, vec![id as u8]);
        }
        for token in &self.special_tokens {
            known.insert(token.id, token.text.as_bytes().to_vec());
        }
        for (index, merge) in self.merges.iter().enumerate() {
            let expected_id = expected_start + index as u32;
            if merge.new_id != expected_id {
                return Err(format!("merge {} has id {}, expected {}", index, merge.new_id, expected_id));
            }
            let left = known.get(&merge.left).ok_or_else(|| format!("merge {} references unknown left token {}", index, merge.left))?;
            let right = known.get(&merge.right).ok_or_else(|| format!("merge {} references unknown right token {}", index, merge.right))?;
            let mut combined = Vec::with_capacity(left.len() + right.len());
            combined.extend_from_slice(left);
            combined.extend_from_slice(right);
            known.insert(merge.new_id, combined);
        }
        Ok(())
    }

    fn rebuild_and_validate(&mut self) -> Result<(), String> {
        self.validate()?;
        self.pair_to_new.clear();
        self.token_bytes.clear();

        for id in 0..BYTE_VOCAB_SIZE {
            self.token_bytes.insert(id, vec![id as u8]);
        }
        for token in &self.special_tokens {
            self.token_bytes.insert(token.id, token.text.as_bytes().to_vec());
        }
        for merge in &self.merges {
            let left = self.token_bytes.get(&merge.left).cloned().ok_or_else(|| format!("unknown merge left {}", merge.left))?;
            let right = self.token_bytes.get(&merge.right).cloned().ok_or_else(|| format!("unknown merge right {}", merge.right))?;
            let mut combined = left;
            combined.extend_from_slice(&right);
            self.token_bytes.insert(merge.new_id, combined);
            self.pair_to_new.insert((merge.left, merge.right), merge.new_id);
        }
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        format::save(self, path.as_ref())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        format::load(path.as_ref())
    }
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
