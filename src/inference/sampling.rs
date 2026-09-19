#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub greedy: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { temperature: 0.8, top_k: 40, top_p: 0.9, greedy: false }
    }
}

#[derive(Clone, Debug)]
pub struct GenerationRng { state: u64 }

impl GenerationRng {
    pub fn new(seed: u64) -> Self { Self { state: seed.max(1) } }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as u32) as f32 / (1u32 << 24) as f32
    }
}

pub fn sample(
    logits: &[f32],
    config: &GenerationConfig,
    rng: &mut GenerationRng,
) -> Result<usize, String> {
    if logits.is_empty() { return Err("cannot sample from empty logits".into()); }
    if config.greedy || config.temperature <= 0.0 { return Ok(argmax(logits)); }

    let temperature = config.temperature.max(1e-4);
    let mut scaled = logits.to_vec();
    for value in &mut scaled { *value /= temperature; }

    let mut indices: Vec<usize> = (0..scaled.len()).collect();
    indices.sort_by(|a, b| scaled[*b].partial_cmp(&scaled[*a]).unwrap());
    if config.top_k > 0 && config.top_k < indices.len() { indices.truncate(config.top_k); }

    let max_logit = indices.iter().map(|i| scaled[*i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = indices.iter().map(|i| (scaled[*i] - max_logit).exp()).collect();
    let total: f32 = probs.iter().sum();
    if total <= 0.0 || !total.is_finite() { return Err("invalid probability mass".into()); }
    for p in &mut probs { *p /= total; }

    if config.top_p > 0.0 && config.top_p < 1.0 {
        let mut cumulative = 0.0;
        let mut keep = 0;
        for p in &probs {
            cumulative += *p;
            keep += 1;
            if cumulative >= config.top_p { break; }
        }
        indices.truncate(keep.max(1));
        probs.truncate(keep.max(1));
        let norm: f32 = probs.iter().sum();
        for p in &mut probs { *p /= norm; }
    }

    let mut threshold = rng.next_f32();
    for (i, probability) in probs.iter().enumerate() {
        if threshold <= *probability { return Ok(indices[i]); }
        threshold -= *probability;
    }
    Ok(*indices.last().unwrap())
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() { if values[i] > values[best] { best = i; } }
    best
}

#[cfg(test)]
mod tests {
    use super::{sample, GenerationConfig, GenerationRng};
    #[test]
    fn greedy_sampling_is_deterministic() {
        let config = GenerationConfig { greedy: true, ..Default::default() };
        let mut rng = GenerationRng::new(1);
        assert_eq!(sample(&[0.1, 4.0, 1.0], &config, &mut rng).unwrap(), 1);
        assert_eq!(sample(&[0.1, 4.0, 1.0], &config, &mut rng).unwrap(), 1);
    }
}
