use super::sampling::{sample, GenerationConfig, GenerationRng};
use crate::model::AiNet;
use crate::tokenizer::Tokenizer;
use std::path::Path;

pub struct InferenceEngine {
    model: Option<AiNet>,
    rng: GenerationRng,
}

impl InferenceEngine {
    pub fn new(seed: u64) -> Self {
        Self {
            model: None,
            rng: GenerationRng::new(seed ^ 0x9E3779B97F4A7C15),
        }
    }

    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        self.model = Some(AiNet::load(path)?);
        Ok(())
    }

    pub fn load_model(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        self.load(path)
    }

    pub fn unload(&mut self) {
        self.model = None;
    }

    pub fn unload_model(&mut self) {
        self.unload();
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.model
            .as_mut()
            .ok_or_else(|| "no model loaded".to_string())?
            .reset_state();
        Ok(())
    }

    pub fn reset_state(&mut self) -> Result<(), String> {
        self.reset()
    }

    pub fn feed_token(&mut self, token: u32) -> Result<Vec<f32>, String> {
        self.model
            .as_mut()
            .ok_or_else(|| "no model loaded".to_string())?
            .inference_logits(token as usize)
    }

    pub fn generate_next(
        &mut self,
        logits: &[f32],
        config: &GenerationConfig,
    ) -> Result<u32, String> {
        Ok(sample(logits, config, &mut self.rng)? as u32)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        tokenizer: &Tokenizer,
        max_new_tokens: usize,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>, String> {
        let ids = tokenizer.encode(prompt);
        self.generate_ids(&ids, tokenizer.eos_id(), max_new_tokens, config)
    }

    pub fn generate_ids(
        &mut self,
        prompt: &[u32],
        eos_id: u32,
        max_new_tokens: usize,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err("generation prompt must not be empty".into());
        }
        self.reset()?;
        let mut logits = Vec::new();
        for token in prompt {
            logits = self.feed_token(*token)?;
        }

        let mut result = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let next = sample(&logits, config, &mut self.rng)? as u32;
            result.push(next);
            if next == eos_id {
                break;
            }
            logits = self.feed_token(next)?;
        }
        Ok(result)
    }

    pub fn generate_stream<F: FnMut(u32)>(
        &mut self,
        prompt: &[u32],
        eos_id: u32,
        max_new_tokens: usize,
        config: &GenerationConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>, String> {
        if prompt.is_empty() {
            return Err("generation prompt must not be empty".into());
        }
        self.reset()?;
        let mut logits = Vec::new();
        for token in prompt {
            logits = self.feed_token(*token)?;
        }

        let mut result = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let next = sample(&logits, config, &mut self.rng)? as u32;
            result.push(next);
            on_token(next);
            if next == eos_id {
                break;
            }
            logits = self.feed_token(next)?;
        }
        Ok(result)
    }

    pub fn set_generation_seed(&mut self, seed: u64) {
        self.rng = GenerationRng::new(seed ^ 0x9E3779B97F4A7C15);
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some()
    }
}
