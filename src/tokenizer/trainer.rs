use super::{default_special_tokens, Merge, Tokenizer};
use crate::dataset::{DatasetFormat, DatasetReader};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct TokenizerTrainerConfig {
    pub target_vocab_size: usize,
    pub min_frequency: u64,
    pub max_merges: usize,
    pub sample_bytes: usize,
}

impl Default for TokenizerTrainerConfig {
    fn default() -> Self {
        Self {
            target_vocab_size: 4096,
            min_frequency: 2,
            max_merges: 3840,
            sample_bytes: 8 * 1024 * 1024,
        }
    }
}

pub struct TokenizerTrainer;

impl TokenizerTrainer {
    pub fn train(
        corpus: impl AsRef<Path>,
        format: DatasetFormat,
        config: &TokenizerTrainerConfig,
    ) -> Result<Tokenizer, String> {
        if config.target_vocab_size < 257 {
            return Err("target vocabulary must exceed the 256-byte base vocabulary".into());
        }
        let specials = default_special_tokens();
        let merge_capacity = config
            .target_vocab_size
            .saturating_sub(256 + specials.len());
        let max_merges = config.max_merges.min(merge_capacity);

        let mut reader = DatasetReader::open(corpus.as_ref(), format.clone())?;
        let mut sequences = Vec::<Vec<u32>>::new();
        let mut sampled = 0usize;

        while sampled < config.sample_bytes {
            match reader.next_sample() {
                Ok(Some(sample)) => {
                    if sample.text.is_empty() {
                        continue;
                    }
                    let bytes = sample.text.as_bytes();
                    let remaining = config.sample_bytes - sampled;
                    let take = bytes.len().min(remaining);
                    if take == 0 {
                        break;
                    }
                    sequences.push(bytes[..take].iter().map(|b| *b as u32).collect());
                    sampled += take;
                }
                Ok(None) => break,
                Err(_) => continue,
            }
        }

        let base = 256 + specials.len() as u32;
        let mut merges = Vec::with_capacity(max_merges);
        for index in 0..max_merges {
            let mut counts = HashMap::<(u32, u32), u64>::new();
            for seq in &sequences {
                for pair in seq.windows(2) {
                    *counts.entry((pair[0], pair[1])).or_default() += 1;
                }
            }
            let Some((&(left, right), &_frequency)) = counts
                .iter()
                .filter(|(_, freq)| **freq >= config.min_frequency)
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            else {
                break;
            };
            let merge = Merge {
                left,
                right,
                new_id: base + index as u32,
            };
            for seq in &mut sequences {
                apply_merge(seq, merge.left, merge.right, merge.new_id);
            }
            merges.push(merge);
        }

        Tokenizer::from_parts(specials, merges)
    }
}

fn apply_merge(seq: &mut Vec<u32>, left: u32, right: u32, new_id: u32) {
    if seq.len() < 2 {
        return;
    }
    let mut out = Vec::with_capacity(seq.len());
    let mut i = 0;
    while i < seq.len() {
        if i + 1 < seq.len() && seq[i] == left && seq[i + 1] == right {
            out.push(new_id);
            i += 2;
        } else {
            out.push(seq[i]);
            i += 1;
        }
    }
    *seq = out;
}
