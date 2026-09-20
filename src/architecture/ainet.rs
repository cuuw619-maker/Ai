#![allow(clippy::needless_range_loop)]
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterDType {
    F32 = 0,
    F16 = 1,
    Q8 = 2,
}

impl ParameterDType {
    fn from_u8(value: u8) -> Result<Self, String> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q8),
            _ => Err(format!("unknown parameter dtype {value}")),
        }
    }
}

use crate::neural::Parameter;
use std::fs;
use std::path::Path;

const ANR_BLOCK_SIZE: usize = 16;
const ANR_DEFAULT_TOP_K: usize = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelMemoryReport {
    pub embedding_bytes: usize,
    pub projection_bytes: usize,
    pub cell_weights_bytes: usize,
    pub output_weights_bytes: usize,
    pub gradients_bytes: usize,
    pub optimizer_m_bytes: usize,
    pub optimizer_v_bytes: usize,
    pub bptt_cache_bytes: usize,
    pub hidden_history_bytes: usize,
    pub temporary_bytes: usize,
    pub dataset_buffer_bytes: usize,
}

impl ModelMemoryReport {
    pub fn total_bytes(&self) -> usize {
        self.embedding_bytes
            + self.projection_bytes
            + self.cell_weights_bytes
            + self.output_weights_bytes
            + self.gradients_bytes
            + self.optimizer_m_bytes
            + self.optimizer_v_bytes
            + self.bptt_cache_bytes
            + self.hidden_history_bytes
            + self.temporary_bytes
            + self.dataset_buffer_bytes
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RoutingStats {
    pub active_blocks: usize,
    pub total_blocks: usize,
    pub active_channels: usize,
    pub total_channels: usize,
    pub skipped_channels: usize,
    pub active_ratio: f32,
    pub entropy: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub architecture: String,
    pub model_id: String,
    pub vocab_size: usize,
    pub embedding_dim: usize,
    pub hidden_dim: usize,
    pub layer_count: usize,
    pub sequence_length: usize,
    pub seed: u64,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.architecture != "AiNet-v1" && self.architecture != "AiNet-v1.1" {
            return Err(format!(
                "unsupported architecture id: {}",
                self.architecture
            ));
        }
        if self.vocab_size < 2
            || self.embedding_dim == 0
            || self.hidden_dim == 0
            || self.layer_count == 0
            || self.sequence_length == 0
        {
            return Err("invalid model dimensions".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
struct CellCache {
    x: Vec<f32>,
    m_prev: Vec<f32>,
    keep: Vec<f32>,
    write: Vec<f32>,
    candidate: Vec<f32>,
    new_memory: Vec<f32>,
    output_activation: Vec<f32>,
    cell_activation: Vec<f32>,
    routing_probs: Vec<f32>,
    routing_gate: Vec<f32>,
    routing_active: Vec<bool>,
}

pub struct AiCell {
    dim: usize,
    pub w_keep: Parameter,
    pub u_keep: Parameter,
    pub b_keep: Parameter,
    pub w_write: Parameter,
    pub u_write: Parameter,
    pub b_write: Parameter,
    pub w_candidate: Parameter,
    pub u_candidate: Parameter,
    pub b_candidate: Parameter,
    pub w_out: Parameter,
    pub b_out: Parameter,
    pub router_w: Parameter,
    pub router_b: Parameter,
}

impl AiCell {
    fn new(prefix: &str, dim: usize, rng: &mut XorShift64) -> Self {
        let matrix = |name: String, out: usize, input: usize, rng: &mut XorShift64| {
            Parameter::new(name, xavier_uniform(out, input, rng))
        };
        Self {
            dim,
            w_keep: matrix(format!("{prefix}.W_keep"), dim, dim, rng),
            u_keep: matrix(format!("{prefix}.U_keep"), dim, dim, rng),
            b_keep: Parameter::zeros(format!("{prefix}.b_keep"), dim),
            w_write: matrix(format!("{prefix}.W_write"), dim, dim, rng),
            u_write: matrix(format!("{prefix}.U_write"), dim, dim, rng),
            b_write: Parameter::zeros(format!("{prefix}.b_write"), dim),
            w_candidate: matrix(format!("{prefix}.W_candidate"), dim, dim, rng),
            u_candidate: matrix(format!("{prefix}.U_candidate"), dim, dim, rng),
            b_candidate: Parameter::zeros(format!("{prefix}.b_candidate"), dim),
            w_out: matrix(format!("{prefix}.W_out"), dim, dim, rng),
            b_out: Parameter::zeros(format!("{prefix}.b_out"), dim),
            router_w: Parameter::new(
                format!("{prefix}.router.weight"),
                xavier_uniform(route_block_count(dim), dim, rng),
            ),
            router_b: Parameter::zeros(
                format!("{prefix}.router.bias"),
                route_block_count(dim),
            ),
        }
    }

    fn total_blocks(&self) -> usize {
        route_block_count(self.dim)
    }

    fn top_k(&self) -> usize {
        ANR_DEFAULT_TOP_K.min(self.total_blocks()).max(1)
    }

    fn forward(&self, x: &[f32], memory: &[f32]) -> CellCache {
        self.forward_internal(x, memory, false)
    }

    fn forward_inference(&self, x: &[f32], memory: &[f32]) -> CellCache {
        self.forward_internal(x, memory, true)
    }

    fn forward_internal(&self, x: &[f32], memory: &[f32], hard_route: bool) -> CellCache {
        let total_blocks = self.total_blocks();
        let mut scores = matvec(&self.router_w.data, x, total_blocks);
        for (score, bias) in scores.iter_mut().zip(&self.router_b.data) {
            *score += *bias;
        }
        let probs = stable_softmax(&scores);
        let mut active = vec![true; total_blocks];
        let mut gate = vec![0.0; total_blocks];
        if hard_route {
            active.fill(false);
            let mut order: Vec<usize> = (0..total_blocks).collect();
            order.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap_or(std::cmp::Ordering::Equal));
            let active_blocks = self.top_k();
            let hard_gate = total_blocks as f32 / active_blocks as f32;
            for block in order.into_iter().take(active_blocks) {
                active[block] = true;
                gate[block] = hard_gate;
            }
        } else {
            for block in 0..total_blocks {
                gate[block] = probs[block] * total_blocks as f32;
            }
        }

        let mut channel_gate = vec![0.0; self.dim];
        for block in 0..total_blocks {
            let start = block * ANR_BLOCK_SIZE;
            let end = (start + ANR_BLOCK_SIZE).min(self.dim);
            for channel in start..end {
                channel_gate[channel] = gate[block];
            }
        }

        let mut keep = vec![1.0; self.dim];
        let mut write = vec![0.0; self.dim];
        let mut candidate = vec![0.0; self.dim];
        if hard_route {
            affine_selected(
                &self.w_keep.data, &self.u_keep.data, &self.b_keep.data,
                x, memory, self.dim, &active, ANR_BLOCK_SIZE, &mut keep, true
            );
            affine_selected(
                &self.w_write.data, &self.u_write.data, &self.b_write.data,
                x, memory, self.dim, &active, ANR_BLOCK_SIZE, &mut write, true
            );
            let kept_memory: Vec<f32> = memory.iter().zip(&keep).map(|(m, k)| m * k).collect();
            affine_selected(
                &self.w_candidate.data, &self.u_candidate.data, &self.b_candidate.data,
                x, &kept_memory, self.dim, &active, ANR_BLOCK_SIZE, &mut candidate, false
            );
        } else {
            let keep_z = affine(&self.w_keep.data, &self.u_keep.data, &self.b_keep.data, x, memory, self.dim);
            keep = keep_z.into_iter().map(sigmoid).collect();
            let write_z = affine(&self.w_write.data, &self.u_write.data, &self.b_write.data, x, memory, self.dim);
            write = write_z.into_iter().map(sigmoid).collect();
            let kept_memory: Vec<f32> = memory.iter().zip(&keep).map(|(m, k)| m * k).collect();
            let candidate_z = affine(&self.w_candidate.data, &self.u_candidate.data, &self.b_candidate.data, x, &kept_memory, self.dim);
            candidate = candidate_z.into_iter().map(f32::tanh).collect();
        }

        let new_memory: Vec<f32> = memory.iter().zip(&write).zip(&candidate)
            .map(|((m, w), c)| m * (1.0 - w) + c * w).collect();

        let mut cell_activation = vec![0.0; self.dim];
        if hard_route {
            matvec_selected(&self.w_out.data, &new_memory, self.dim, &active, ANR_BLOCK_SIZE, &mut cell_activation);
        } else {
            let output_z = matvec(&self.w_out.data, &new_memory, self.dim);
            cell_activation.copy_from_slice(&output_z);
        }
        for value in &mut cell_activation {
            *value = value.tanh();
        }
        let output_activation = cell_activation.iter().zip(&channel_gate)
            .map(|(value, gate)| value * gate).collect();

        CellCache {
            x: x.to_vec(),
            m_prev: memory.to_vec(),
            keep,
            write,
            candidate,
            new_memory,
            output_activation,
            cell_activation,
            routing_probs: probs,
            routing_gate: channel_gate,
            routing_active: active,
        }
    }

    fn backward(
        &mut self,
        cache: &CellCache,
        grad_output: &[f32],
        grad_memory_future: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.dim;
        let mut dx = grad_output.to_vec();
        let mut dnew = grad_memory_future.to_vec();

        // ANR soft-routing gradient: output = tanh(cell) * gate.
        // The gate is a scaled softmax over fixed-size channel blocks.
        let mut d_scores = vec![0.0; cache.routing_probs.len()];
        let mut weighted_gate_gradient = 0.0f32;
        for i in 0..d {
            let block = i / ANR_BLOCK_SIZE;
            let d_gate = grad_output[i] * cache.cell_activation[i];
            weighted_gate_gradient += d_gate * cache.routing_probs[block];
        }
        for block in 0..cache.routing_probs.len() {
            let mut d_gate = 0.0f32;
            let start = block * ANR_BLOCK_SIZE;
            let end = (start + ANR_BLOCK_SIZE).min(d);
            for i in start..end {
                d_gate += grad_output[i] * cache.cell_activation[i];
            }
            d_scores[block] = cache.routing_probs[block] * (d_gate * cache.routing_probs.len() as f32 - weighted_gate_gradient * cache.routing_probs.len() as f32);
        }
        for block in 0..cache.routing_probs.len() {
            let row = &self.router_w.data[block * d..(block + 1) * d];
            for i in 0..d {
                dx[i] += row[i] * d_scores[block];
                self.router_w.grad[block * d + i] += d_scores[block] * cache.x[i];
            }
            self.router_b.grad[block] += d_scores[block];
        }

        let mut dz_out = vec![0.0; d];
        for i in 0..d {
            let gate = cache.routing_gate[i];
            dz_out[i] =
                grad_output[i] * gate * (1.0 - cache.cell_activation[i] * cache.cell_activation[i]);
        }
        outer_add(&mut self.w_out.grad, &dz_out, &cache.new_memory, d, d);
        for i in 0..d {
            self.b_out.grad[i] += dz_out[i];
        }
        add_matvec_t(&mut dnew, &self.w_out.data, &dz_out, d, d);

        let mut dkeep = vec![0.0; d];
        let mut dwrite = vec![0.0; d];
        let mut dcandidate = vec![0.0; d];
        let mut dm_prev = vec![0.0; d];

        for i in 0..d {
            dm_prev[i] += dnew[i] * (1.0 - cache.write[i]);
            dwrite[i] += dnew[i] * (cache.candidate[i] - cache.m_prev[i]);
            dcandidate[i] += dnew[i] * cache.write[i];
        }

        let dcandidate_z: Vec<f32> = dcandidate
            .iter()
            .zip(&cache.candidate)
            .map(|(g, c)| g * (1.0 - c * c))
            .collect();
        let kept_memory: Vec<f32> = cache
            .m_prev
            .iter()
            .zip(&cache.keep)
            .map(|(m, k)| m * k)
            .collect();
        outer_add(&mut self.w_candidate.grad, &dcandidate_z, &cache.x, d, d);
        outer_add(
            &mut self.u_candidate.grad,
            &dcandidate_z,
            &kept_memory,
            d,
            d,
        );
        for i in 0..d {
            self.b_candidate.grad[i] += dcandidate_z[i];
        }
        add_matvec_t(&mut dx, &self.w_candidate.data, &dcandidate_z, d, d);
        let mut dkept = vec![0.0; d];
        add_matvec_t(&mut dkept, &self.u_candidate.data, &dcandidate_z, d, d);

        for i in 0..d {
            dm_prev[i] += dkept[i] * cache.keep[i];
            dkeep[i] += dkept[i] * cache.m_prev[i];
        }

        let dwrite_z: Vec<f32> = dwrite
            .iter()
            .zip(&cache.write)
            .map(|(g, w)| g * w * (1.0 - w))
            .collect();
        outer_add(&mut self.w_write.grad, &dwrite_z, &cache.x, d, d);
        outer_add(&mut self.u_write.grad, &dwrite_z, &cache.m_prev, d, d);
        for i in 0..d {
            self.b_write.grad[i] += dwrite_z[i];
        }
        add_matvec_t(&mut dx, &self.w_write.data, &dwrite_z, d, d);
        add_matvec_t(&mut dm_prev, &self.u_write.data, &dwrite_z, d, d);

        let dkeep_z: Vec<f32> = dkeep
            .iter()
            .zip(&cache.keep)
            .map(|(g, k)| g * k * (1.0 - k))
            .collect();
        outer_add(&mut self.w_keep.grad, &dkeep_z, &cache.x, d, d);
        outer_add(&mut self.u_keep.grad, &dkeep_z, &cache.m_prev, d, d);
        for i in 0..d {
            self.b_keep.grad[i] += dkeep_z[i];
        }
        add_matvec_t(&mut dx, &self.w_keep.data, &dkeep_z, d, d);
        add_matvec_t(&mut dm_prev, &self.u_keep.data, &dkeep_z, d, d);

        (dx, dm_prev)
    }

    fn parameters_mut(&mut self) -> [&mut Parameter; 13] {
        [
            &mut self.w_keep,
            &mut self.u_keep,
            &mut self.b_keep,
            &mut self.w_write,
            &mut self.u_write,
            &mut self.b_write,
            &mut self.w_candidate,
            &mut self.u_candidate,
            &mut self.b_candidate,
            &mut self.w_out,
            &mut self.b_out,
            &mut self.router_w,
            &mut self.router_b,
        ]
    }

    fn parameter_count(&self) -> usize {
        self.w_keep.len()
            + self.u_keep.len()
            + self.b_keep.len()
            + self.w_write.len()
            + self.u_write.len()
            + self.b_write.len()
            + self.w_candidate.len()
            + self.u_candidate.len()
            + self.b_candidate.len()
            + self.w_out.len()
            + self.b_out.len()
            + self.router_w.len()
            + self.router_b.len()
    }
}

pub struct AiNet {
    pub config: ModelConfig,
    pub embedding: Parameter,
    pub input_projection_w: Option<Parameter>,
    pub input_projection_b: Option<Parameter>,
    pub cells: Vec<AiCell>,
    pub output_w: Parameter,
    pub output_b: Parameter,
    runtime_memory: Vec<Vec<f32>>,
    last_activations: Vec<Vec<f32>>,
    last_routing: Vec<RoutingStats>,
}

impl AiNet {
    pub fn new(config: ModelConfig) -> Result<Self, String> {
        config.validate()?;
        let mut rng = XorShift64::new(config.seed);
        let embedding = Parameter::new(
            "embedding.weight",
            uniform(config.vocab_size * config.embedding_dim, 0.1, &mut rng),
        );
        let (input_projection_w, input_projection_b) = if config.architecture == "AiNet-v1.1" {
            (
                Some(Parameter::new(
                    "input_projection.weight",
                    xavier_uniform(config.hidden_dim, config.embedding_dim, &mut rng),
                )),
                Some(Parameter::zeros("input_projection.bias", config.hidden_dim)),
            )
        } else {
            (None, None)
        };
        let mut cells = Vec::with_capacity(config.layer_count);
        for layer in 0..config.layer_count {
            cells.push(AiCell::new(
                &format!("cell.{layer}"),
                config.hidden_dim,
                &mut rng,
            ));
        }
        let output_w = Parameter::new(
            "output.weight",
            xavier_uniform(config.vocab_size, config.hidden_dim, &mut rng),
        );
        let output_b = Parameter::zeros("output.bias", config.vocab_size);
        let runtime_memory = vec![vec![0.0; config.hidden_dim]; config.layer_count];
        let last_activations = vec![vec![0.0; config.hidden_dim]; config.layer_count];
        let last_routing = vec![RoutingStats::default(); config.layer_count];
        Ok(Self {
            config,
            embedding,
            input_projection_w,
            input_projection_b,
            cells,
            output_w,
            output_b,
            runtime_memory,
            last_activations,
            last_routing,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.embedding.len()
            + self.input_projection_w.as_ref().map_or(0, Parameter::len)
            + self.input_projection_b.as_ref().map_or(0, Parameter::len)
            + self.output_w.len()
            + self.output_b.len()
            + self
                .cells
                .iter()
                .map(AiCell::parameter_count)
                .sum::<usize>()
    }

    pub fn zero_grad(&mut self) {
        self.embedding.zero_grad();
        if let Some(p) = &mut self.input_projection_w {
            p.zero_grad();
        }
        if let Some(p) = &mut self.input_projection_b {
            p.zero_grad();
        }
        for cell in &mut self.cells {
            for p in cell.parameters_mut() {
                p.zero_grad();
            }
        }
        self.output_w.zero_grad();
        self.output_b.zero_grad();
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut raw = Vec::<*mut Parameter>::with_capacity(self.config.layer_count * 13 + 3);
        raw.push(&mut self.embedding);
        if let Some(p) = &mut self.input_projection_w {
            raw.push(p);
        }
        if let Some(p) = &mut self.input_projection_b {
            raw.push(p);
        }
        for cell in &mut self.cells {
            raw.push(&mut cell.w_keep);
            raw.push(&mut cell.u_keep);
            raw.push(&mut cell.b_keep);
            raw.push(&mut cell.w_write);
            raw.push(&mut cell.u_write);
            raw.push(&mut cell.b_write);
            raw.push(&mut cell.w_candidate);
            raw.push(&mut cell.u_candidate);
            raw.push(&mut cell.b_candidate);
            raw.push(&mut cell.w_out);
            raw.push(&mut cell.b_out);
        }
        raw.push(&mut self.output_w);
        raw.push(&mut self.output_b);
        for cell in &mut self.cells {
            raw.push(&mut cell.router_w);
            raw.push(&mut cell.router_b);
        }
        // Each pointer is created from a distinct Parameter field above and
        // remains valid for the returned vector because no vector reallocation
        // or mutation of self occurs while the caller uses these references.
        unsafe { raw.into_iter().map(|ptr| &mut *ptr).collect() }
    }

    pub fn train_step(
        &mut self,
        input_tokens: &[usize],
        target_tokens: &[usize],
        initial_memory: Option<&[Vec<f32>]>,
    ) -> Result<f32, String> {
        self.train_step_with_state(input_tokens, target_tokens, initial_memory)
            .map(|(loss, _)| loss)
    }

    pub fn train_step_with_state(
        &mut self,
        input_tokens: &[usize],
        target_tokens: &[usize],
        initial_memory: Option<&[Vec<f32>]>,
    ) -> Result<(f32, Vec<Vec<f32>>), String> {
        self.train_step_core(input_tokens, target_tokens, initial_memory, true)
    }

    pub fn accumulate_train_step_with_state(
        &mut self,
        input_tokens: &[usize],
        target_tokens: &[usize],
        initial_memory: Option<&[Vec<f32>]>,
    ) -> Result<(f32, Vec<Vec<f32>>), String> {
        self.train_step_core(input_tokens, target_tokens, initial_memory, false)
    }

    fn train_step_core(
        &mut self,
        input_tokens: &[usize],
        target_tokens: &[usize],
        initial_memory: Option<&[Vec<f32>]>,
        reset_gradients: bool,
    ) -> Result<(f32, Vec<Vec<f32>>), String> {
        if input_tokens.is_empty() || input_tokens.len() != target_tokens.len() {
            return Err("input and target sequence lengths must match and be non-zero".into());
        }
        if input_tokens.len() > self.config.sequence_length {
            return Err("sequence exceeds model sequence_length".into());
        }
        for &token in input_tokens.iter().chain(target_tokens) {
            if token >= self.config.vocab_size {
                return Err("token id outside vocabulary".into());
            }
        }

        if reset_gradients {
            self.zero_grad();
        }
        let mut memory = initial_memory
            .map(|m| m.to_vec())
            .unwrap_or_else(|| vec![vec![0.0; self.config.hidden_dim]; self.config.layer_count]);
        if memory.len() != self.config.layer_count {
            return Err("initial memory layer count mismatch".into());
        }

        let mut caches: Vec<Vec<CellCache>> = (0..self.config.layer_count)
            .map(|_| Vec::with_capacity(input_tokens.len()))
            .collect();
        let mut hidden_history = Vec::with_capacity(input_tokens.len());
        let mut embedding_history = Vec::with_capacity(input_tokens.len());

        for &token in input_tokens {
            let embedding = embedding_row(&self.embedding.data, token, self.config.embedding_dim);
            embedding_history.push(embedding.clone());
            let mut x = self.project_input(&embedding)?;
            for (layer, cell) in self.cells.iter().enumerate() {
                let cache = cell.forward(&x, &memory[layer]);
                self.last_activations[layer] = cache.output_activation.clone();
                self.last_routing[layer] = routing_stats_from_probs(&cache.routing_probs, self.config.hidden_dim, cell.top_k());
                memory[layer] = cache.new_memory.clone();
                x = cache
                    .x
                    .iter()
                    .zip(&cache.output_activation)
                    .map(|(a, b)| a + b)
                    .collect();
                caches[layer].push(cache);
            }
            hidden_history.push(x);
        }

        let time = input_tokens.len();
        let scale = 1.0 / time as f32;
        let mut loss = 0.0;
        let mut hidden_grads = vec![vec![0.0; self.config.hidden_dim]; time];

        for t in 0..time {
            let mut logits = matvec(
                &self.output_w.data,
                &hidden_history[t],
                self.config.vocab_size,
            );
            for v in 0..self.config.vocab_size {
                logits[v] += self.output_b.data[v];
            }
            let probs = stable_softmax(&logits);
            loss += cross_entropy(&logits, target_tokens[t]);
            let mut dlogits = probs;
            dlogits[target_tokens[t]] -= 1.0;
            for v in 0..self.config.vocab_size {
                let g = dlogits[v] * scale;
                self.output_b.grad[v] += g;
                for h in 0..self.config.hidden_dim {
                    self.output_w.grad[v * self.config.hidden_dim + h] += g * hidden_history[t][h];
                }
                for h in 0..self.config.hidden_dim {
                    hidden_grads[t][h] += self.output_w.data[v * self.config.hidden_dim + h] * g;
                }
            }
        }

        let mut memory_grads = vec![vec![0.0; self.config.hidden_dim]; self.config.layer_count];

        for t in (0..time).rev() {
            let mut upstream = hidden_grads[t].clone();
            for layer in (0..self.config.layer_count).rev() {
                let cell = &mut self.cells[layer];
                let (dx, dm) = cell.backward(&caches[layer][t], &upstream, &memory_grads[layer]);
                memory_grads[layer] = dm;
                upstream = dx;
            }
            let token = input_tokens[t];
            let embedding_dim = self.config.embedding_dim;
            let base = token * embedding_dim;
            let embedding = &embedding_history[t];

            if let (Some(w), Some(b)) = (&mut self.input_projection_w, &mut self.input_projection_b)
            {
                outer_add(
                    &mut w.grad,
                    &upstream,
                    embedding,
                    self.config.hidden_dim,
                    embedding_dim,
                );
                for i in 0..self.config.hidden_dim {
                    b.grad[i] += upstream[i];
                }
                let grad_embedding = matvec_transposed(
                    w.data.as_slice(),
                    &upstream,
                    self.config.hidden_dim,
                    embedding_dim,
                );
                for i in 0..embedding_dim {
                    self.embedding.grad[base + i] += grad_embedding[i];
                }
            } else {
                let copy = embedding_dim.min(upstream.len());
                for i in 0..copy {
                    self.embedding.grad[base + i] += upstream[i];
                }
            }
        }

        Ok((loss * scale, memory))
    }

    pub fn sequence_loss(
        &self,
        input_tokens: &[usize],
        target_tokens: &[usize],
        initial_memory: Option<&[Vec<f32>]>,
    ) -> Result<(f32, Vec<Vec<f32>>), String> {
        if input_tokens.is_empty() || input_tokens.len() != target_tokens.len() {
            return Err("input and target sequence lengths must match and be non-zero".into());
        }
        if input_tokens.len() > self.config.sequence_length {
            return Err("sequence exceeds model sequence_length".into());
        }
        if input_tokens
            .iter()
            .chain(target_tokens)
            .any(|token| *token >= self.config.vocab_size)
        {
            return Err("token id outside vocabulary".into());
        }

        let mut memory = initial_memory
            .map(|m| m.to_vec())
            .unwrap_or_else(|| vec![vec![0.0; self.config.hidden_dim]; self.config.layer_count]);
        if memory.len() != self.config.layer_count
            || memory
                .iter()
                .any(|layer| layer.len() != self.config.hidden_dim)
        {
            return Err("initial memory shape mismatch".into());
        }

        let mut loss = 0.0;
        for (input, &target) in input_tokens.iter().zip(target_tokens) {
            let embedding = embedding_row(&self.embedding.data, *input, self.config.embedding_dim);
            let mut x = self.project_input(&embedding)?;
            for (layer, cell) in self.cells.iter().enumerate() {
                let cache = cell.forward(&x, &memory[layer]);
                memory[layer] = cache.new_memory;
                x = x
                    .iter()
                    .zip(&cache.output_activation)
                    .map(|(a, b)| a + b)
                    .collect();
            }
            let mut logits = matvec(&self.output_w.data, &x, self.config.vocab_size);
            for (value, bias) in logits.iter_mut().zip(&self.output_b.data) {
                *value += *bias;
            }
            loss += cross_entropy(&logits, target);
        }
        Ok((loss / input_tokens.len() as f32, memory))
    }

    pub fn trainable_architecture(&self) -> &str {
        &self.config.architecture
    }

    pub fn weights_checksum(&self) -> u64 {
        let mut bytes = Vec::with_capacity(self.parameter_count() * 4);
        for (_, data) in self.parameter_snapshot() {
            for value in data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        fnv1a64(&bytes)
    }

    pub fn routing_stats(&self) -> Vec<RoutingStats> {
        self.last_routing.clone()
    }

    pub fn layer_activation_stats(&self) -> Vec<(f32, f32, f32)> {
        self.last_activations
            .iter()
            .map(|values| {
                if values.is_empty() {
                    return (0.0, 0.0, 0.0);
                }
                let sum = values.iter().map(|v| *v as f64).sum::<f64>();
                let mean = sum / values.len() as f64;
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (mean as f32, min, max)
            })
            .collect()
    }

    pub fn parameter_names(&self) -> Vec<String> {
        self.parameter_snapshot()
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut result = Vec::with_capacity(self.config.layer_count * 13 + 3);
        result.push(&self.embedding);
        if let Some(p) = &self.input_projection_w {
            result.push(p);
        }
        if let Some(p) = &self.input_projection_b {
            result.push(p);
        }
        for cell in &self.cells {
            let ps = [
                &cell.w_keep,
                &cell.u_keep,
                &cell.b_keep,
                &cell.w_write,
                &cell.u_write,
                &cell.b_write,
                &cell.w_candidate,
                &cell.u_candidate,
                &cell.b_candidate,
                &cell.w_out,
                &cell.b_out,
            ];
            result.extend(ps);
        }
        result.push(&self.output_w);
        result.push(&self.output_b);
        for cell in &self.cells {
            result.push(&cell.router_w);
            result.push(&cell.router_b);
        }
        result
    }

    pub fn inference_logits(&mut self, token: usize) -> Result<Vec<f32>, String> {
        self.inference_token(token)
    }

    pub fn global_gradient_norm(&mut self) -> f32 {
        let mut sum = 0.0f64;
        let params = self.parameters_mut();
        for p in params {
            for g in &p.grad {
                sum += (*g as f64) * (*g as f64);
            }
        }
        sum.sqrt() as f32
    }

    pub fn scale_gradients(&mut self, scale: f32) {
        if !scale.is_finite() {
            return;
        }
        for p in self.parameters_mut() {
            for g in &mut p.grad {
                *g *= scale;
            }
        }
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32) -> (f32, bool) {
        let norm = self.global_gradient_norm();
        if max_norm > 0.0 && norm > max_norm {
            let scale = max_norm / norm;
            let params = self.parameters_mut();
            for p in params {
                for g in &mut p.grad {
                    *g *= scale;
                }
            }
            (norm, true)
        } else {
            (norm, false)
        }
    }

    pub fn memory_report(&self, sequence_length: usize) -> ModelMemoryReport {
        let parameter_bytes = self.parameter_count() * std::mem::size_of::<f32>();
        let bptt_cache_bytes = sequence_length
            * self.config.layer_count
            * self.config.hidden_dim
            * 7
            * std::mem::size_of::<f32>();
        let hidden_history_bytes =
            sequence_length * self.config.hidden_dim * std::mem::size_of::<f32>();
        ModelMemoryReport {
            embedding_bytes: self.embedding.len() * std::mem::size_of::<f32>(),
            projection_bytes: self.input_projection_w.as_ref().map_or(0, Parameter::len)
                * std::mem::size_of::<f32>()
                + self.input_projection_b.as_ref().map_or(0, Parameter::len)
                    * std::mem::size_of::<f32>(),
            cell_weights_bytes: self
                .cells
                .iter()
                .map(AiCell::parameter_count)
                .sum::<usize>()
                * std::mem::size_of::<f32>(),
            output_weights_bytes: (self.output_w.len() + self.output_b.len())
                * std::mem::size_of::<f32>(),
            gradients_bytes: parameter_bytes,
            optimizer_m_bytes: parameter_bytes,
            optimizer_v_bytes: parameter_bytes,
            bptt_cache_bytes,
            hidden_history_bytes,
            temporary_bytes: hidden_history_bytes,
            dataset_buffer_bytes: 4 * 1024 * 1024
                + (sequence_length + 1) * std::mem::size_of::<u32>(),
        }
    }

    pub fn estimated_training_bytes(&self, sequence_length: usize) -> usize {
        self.memory_report(sequence_length).total_bytes()
    }

    fn project_input(&self, embedding: &[f32]) -> Result<Vec<f32>, String> {
        if let (Some(w), Some(b)) = (&self.input_projection_w, &self.input_projection_b) {
            let mut x = matvec(w.data.as_slice(), embedding, self.config.hidden_dim);
            for i in 0..self.config.hidden_dim {
                x[i] += b.data[i];
            }
            Ok(x)
        } else {
            Ok(project_embedding(embedding, self.config.hidden_dim))
        }
    }

    pub fn reset_state(&mut self) {
        for state in &mut self.runtime_memory {
            state.fill(0.0);
        }
    }

    pub fn generate_tokens(
        &mut self,
        prompt: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, String> {
        if prompt.is_empty() {
            return Err("generation requires at least one prompt token".into());
        }
        self.reset_state();
        let mut last_logits = Vec::new();
        for &token in prompt {
            last_logits = self.inference_token(token)?;
        }
        let mut result = prompt.to_vec();
        for _ in 0..max_new_tokens {
            let next = argmax(&last_logits);
            result.push(next);
            last_logits = self.inference_token(next)?;
        }
        Ok(result)
    }

    fn inference_token(&mut self, token: usize) -> Result<Vec<f32>, String> {
        if token >= self.config.vocab_size {
            return Err("token id outside vocabulary".into());
        }
        let embedding = embedding_row(&self.embedding.data, token, self.config.embedding_dim);
        let mut x = self.project_input(&embedding)?;
        for layer in 0..self.config.layer_count {
            let cell = &self.cells[layer];
            let cache = cell.forward_inference(&x, &self.runtime_memory[layer]);
            self.last_routing[layer] = routing_stats_from_hard(&cache.routing_active, self.config.hidden_dim, &cache.routing_probs);
            self.last_activations[layer] = cache.output_activation.clone();
            self.runtime_memory[layer] = cache.new_memory;
            x = x
                .iter()
                .zip(&cache.output_activation)
                .map(|(a, b)| a + b)
                .collect();
        }
        let mut logits = matvec(&self.output_w.data, &x, self.config.vocab_size);
        for v in 0..self.config.vocab_size {
            logits[v] += self.output_b.data[v];
        }
        Ok(logits)
    }

    pub fn to_aimodel_bytes(&self) -> Result<Vec<u8>, String> {
        let payload = self.encode_payload_v2()?;
        let checksum = fnv1a64(&payload);
        let mut bytes = Vec::with_capacity(28 + payload.len());
        bytes.extend_from_slice(b"AIMDLv02");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&checksum.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn from_aimodel_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 28 {
            return Err("invalid .aimodel header".into());
        }
        if &bytes[..8] == b"AIMDLv01" {
            return Self::load_v1(bytes);
        }
        if &bytes[..8] != b"AIMDLv02" {
            return Err("invalid .aimodel magic".into());
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != 2 {
            return Err(format!("unsupported .aimodel version {version}"));
        }
        let expected_checksum = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let payload_len = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
        if bytes.len() != 28 + payload_len {
            return Err("invalid .aimodel payload length".into());
        }
        let payload = &bytes[28..];
        if fnv1a64(payload) != expected_checksum {
            return Err("model checksum mismatch".into());
        }
        Self::decode_payload_v2(payload)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let bytes = self.to_aimodel_bytes()?;
        atomic_save_with_previous(path.as_ref(), &bytes)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("read model: {e}"))?;
        Self::from_aimodel_bytes(&bytes)
    }

    fn load_v1(bytes: &[u8]) -> Result<Self, String> {
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != 1 {
            return Err(format!("unsupported legacy .aimodel version {version}"));
        }
        let expected_checksum = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        let payload_len = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
        if bytes.len() != 28 + payload_len {
            return Err("invalid legacy .aimodel payload length".into());
        }
        let payload = &bytes[28..];
        if fnv1a64(payload) != expected_checksum {
            return Err("legacy model checksum mismatch".into());
        }
        Self::decode_payload_v1(payload)
    }

    #[cfg(test)]
    fn encode_payload_v1(&self) -> Result<Vec<u8>, String> {
        let mut w = Writer::default();
        w.str(&self.config.architecture);
        w.u64(self.config.vocab_size as u64);
        w.u64(self.config.embedding_dim as u64);
        w.u64(self.config.hidden_dim as u64);
        w.u64(self.config.layer_count as u64);
        w.u64(self.config.sequence_length as u64);
        w.u64(self.config.seed);
        let params = self.parameter_snapshot();
        w.u64(params.len() as u64);
        for (name, data) in params {
            w.str(&name);
            w.u64(data.len() as u64);
            for value in data {
                w.f32(value);
            }
        }
        Ok(w.bytes)
    }

    fn encode_payload_v2(&self) -> Result<Vec<u8>, String> {
        let mut w = Writer::default();
        w.str(&self.config.architecture);
        w.str(&self.config.model_id);
        w.u64(self.config.vocab_size as u64);
        w.u64(self.config.embedding_dim as u64);
        w.u64(self.config.hidden_dim as u64);
        w.u64(self.config.layer_count as u64);
        w.u64(self.config.sequence_length as u64);
        w.u64(self.config.seed);
        let params = self.parameter_snapshot();
        w.u64(params.len() as u64);
        for (name, data) in params {
            w.str(&name);
            w.u8(ParameterDType::F32 as u8);
            w.u64(data.len() as u64);
            for value in data {
                w.f32(value);
            }
        }
        Ok(w.bytes)
    }

    fn decode_payload_v2(payload: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(payload);
        let config = ModelConfig {
            architecture: r.str()?,
            model_id: r.str()?,
            vocab_size: r.u64()? as usize,
            embedding_dim: r.u64()? as usize,
            hidden_dim: r.u64()? as usize,
            layer_count: r.u64()? as usize,
            sequence_length: r.u64()? as usize,
            seed: r.u64()?,
        };
        let mut model = AiNet::new(config)?;
        let count = r.u64()? as usize;
        let mut expected = model.parameter_snapshot();
        let legacy_count = expected.len().saturating_sub(model.config.layer_count * 2);
        let legacy_router_payload = count == legacy_count;
        if count != expected.len() && !legacy_router_payload {
            return Err("parameter count mismatch".into());
        }
        for (expected_name, expected_data) in expected.iter_mut().take(count) {
            let name = r.str()?;
            if &name != expected_name {
                return Err(format!("parameter order/name mismatch: {name}"));
            }
            let dtype = ParameterDType::from_u8(r.u8()?)?;
            if dtype != ParameterDType::F32 {
                return Err(format!(
                    "parameter dtype {:?} is not implemented yet",
                    dtype
                ));
            }
            let len = r.u64()? as usize;
            if len != expected_data.len() {
                return Err(format!("parameter length mismatch for {name}"));
            }
            for value in expected_data.iter_mut() {
                *value = r.f32()?;
            }
        }
        if !r.finished() {
            return Err("trailing bytes in .aimodel v2 payload".into());
        }
        let mut params = model.parameters_mut();
        for (param, (_, data)) in params.iter_mut().zip(expected) {
            param.data.copy_from_slice(&data);
            param.grad.fill(0.0);
        }
        if legacy_router_payload {
            zero_router_parameters(&mut model);
        }
        Ok(model)
    }

    fn decode_payload_v1(payload: &[u8]) -> Result<Self, String> {
        let mut r = Reader::new(payload);
        let architecture = r.str()?;
        let vocab_size = r.u64()? as usize;
        let embedding_dim = r.u64()? as usize;
        let hidden_dim = r.u64()? as usize;
        let layer_count = r.u64()? as usize;
        let sequence_length = r.u64()? as usize;
        let seed = r.u64()?;
        let config = ModelConfig {
            architecture,
            model_id: format!("legacy-{seed:016x}"),
            vocab_size,
            embedding_dim,
            hidden_dim,
            layer_count,
            sequence_length,
            seed,
        };
        let mut model = AiNet::new(config)?;
        let count = r.u64()? as usize;
        let mut expected = model.parameter_snapshot();
        let legacy_count = expected.len().saturating_sub(model.config.layer_count * 2);
        let legacy_router_payload = count == legacy_count;
        if count != expected.len() && !legacy_router_payload {
            return Err("parameter count mismatch".into());
        }
        for (expected_name, expected_data) in expected.iter_mut().take(count) {
            let name = r.str()?;
            if &name != expected_name {
                return Err(format!("parameter order/name mismatch: {name}"));
            }
            let len = r.u64()? as usize;
            if len != expected_data.len() {
                return Err(format!("parameter length mismatch for {name}"));
            }
            for value in expected_data.iter_mut() {
                *value = r.f32()?;
            }
        }
        if !r.finished() {
            return Err("trailing bytes in .aimodel".into());
        }
        let mut params = model.parameters_mut();
        for (param, (_, data)) in params.iter_mut().zip(expected) {
            param.data.copy_from_slice(&data);
            param.grad.fill(0.0);
        }
        if legacy_router_payload {
            zero_router_parameters(&mut model);
        }
        Ok(model)
    }

    fn parameter_snapshot(&self) -> Vec<(String, Vec<f32>)> {
        let mut result = Vec::new();
        result.push((self.embedding.name.clone(), self.embedding.data.clone()));
        if let Some(p) = &self.input_projection_w {
            result.push((p.name.clone(), p.data.clone()));
        }
        if let Some(p) = &self.input_projection_b {
            result.push((p.name.clone(), p.data.clone()));
        }
        for cell in &self.cells {
            let ps = [
                &cell.w_keep,
                &cell.u_keep,
                &cell.b_keep,
                &cell.w_write,
                &cell.u_write,
                &cell.b_write,
                &cell.w_candidate,
                &cell.u_candidate,
                &cell.b_candidate,
                &cell.w_out,
                &cell.b_out,
            ];
            for p in ps {
                result.push((p.name.clone(), p.data.clone()));
            }
        }
        result.push((self.output_w.name.clone(), self.output_w.data.clone()));
        result.push((self.output_b.name.clone(), self.output_b.data.clone()));
        for cell in &self.cells {
            result.push((cell.router_w.name.clone(), cell.router_w.data.clone()));
            result.push((cell.router_b.name.clone(), cell.router_b.data.clone()));
        }
        result
    }
}

fn atomic_save_with_previous(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    let temp = path.with_extension("aimodel.tmp");
    let previous = path.with_extension("aimodel.prev");

    {
        let mut file = fs::File::create(&temp).map_err(|e| format!("create model temp: {e}"))?;
        file.write_all(bytes)
            .map_err(|e| format!("write model temp: {e}"))?;
        file.flush().map_err(|e| format!("flush model temp: {e}"))?;
        file.sync_all()
            .map_err(|e| format!("sync model temp: {e}"))?;
    }

    if path.exists() {
        if previous.exists() {
            fs::remove_file(&previous).map_err(|e| format!("remove previous model: {e}"))?;
        }
        fs::rename(path, &previous).map_err(|e| format!("rotate current model: {e}"))?;
    }

    match fs::rename(&temp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            if previous.exists() && !path.exists() {
                let _ = fs::rename(&previous, path);
            }
            let _ = fs::remove_file(&temp);
            Err(format!("install model atomically: {e}"))
        }
    }
}

fn route_block_count(dim: usize) -> usize {
    dim.div_ceil(ANR_BLOCK_SIZE).max(1)
}

fn routing_entropy(probs: &[f32]) -> f32 {
    if probs.len() <= 1 {
        return 0.0;
    }
    let entropy = probs
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -*p * p.ln())
        .sum::<f32>();
    (entropy / (probs.len() as f32).ln()).clamp(0.0, 1.0)
}

fn routing_stats_from_probs(probs: &[f32], channels: usize, _top_k: usize) -> RoutingStats {
    // Training deliberately uses soft routing, so every block participates in the
    // forward/backward computation. Entropy still exposes how concentrated the
    // learned router is without pretending skipped work occurred.
    let total_blocks = probs.len();
    RoutingStats {
        active_blocks: total_blocks,
        total_blocks,
        active_channels: channels,
        total_channels: channels,
        skipped_channels: 0,
        active_ratio: 1.0,
        entropy: routing_entropy(probs),
    }
}

fn routing_stats_from_hard(active: &[bool], channels: usize, probs: &[f32]) -> RoutingStats {
    let active_blocks = active.iter().filter(|value| **value).count();
    let active_channels = (0..channels)
        .filter(|channel| active[*channel / ANR_BLOCK_SIZE])
        .count();
    RoutingStats {
        active_blocks,
        total_blocks: active.len(),
        active_channels,
        total_channels: channels,
        skipped_channels: channels.saturating_sub(active_channels),
        active_ratio: active_channels as f32 / channels.max(1) as f32,
        entropy: routing_entropy(probs),
    }
}

fn affine_selected(
    w: &[f32],
    u: &[f32],
    b: &[f32],
    x: &[f32],
    memory: &[f32],
    dim: usize,
    active: &[bool],
    block_size: usize,
    output: &mut [f32],
    sigmoid_output: bool,
) {
    for block in 0..active.len() {
        if !active[block] {
            continue;
        }
        let start = block * block_size;
        let end = (start + block_size).min(dim);
        for row in start..end {
            let mut sum = b[row];
            for col in 0..dim {
                sum += w[row * dim + col] * x[col] + u[row * dim + col] * memory[col];
            }
            output[row] = if sigmoid_output { sigmoid(sum) } else { sum.tanh() };
        }
    }
}

fn matvec_selected(
    matrix: &[f32],
    vector: &[f32],
    rows: usize,
    active: &[bool],
    block_size: usize,
    output: &mut [f32],
) {
    for block in 0..active.len() {
        if !active[block] {
            continue;
        }
        let start = block * block_size;
        let end = (start + block_size).min(rows);
        for row in start..end {
            let mut sum = 0.0;
            for col in 0..vector.len() {
                sum += matrix[row * vector.len() + col] * vector[col];
            }
            output[row] = sum;
        }
    }
}

fn zero_router_parameters(model: &mut AiNet) {
    for cell in &mut model.cells {
        cell.router_w.data.fill(0.0);
        cell.router_b.data.fill(0.0);
        cell.router_w.grad.fill(0.0);
        cell.router_b.grad.fill(0.0);
    }
}

fn matvec_transposed(matrix: &[f32], vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c] += matrix[r * cols + c] * vector[r];
        }
    }
    out
}

fn embedding_row(data: &[f32], token: usize, dim: usize) -> Vec<f32> {
    data[token * dim..(token + 1) * dim].to_vec()
}

fn project_embedding(x: &[f32], hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0; hidden];
    let copy = x.len().min(hidden);
    out[..copy].copy_from_slice(&x[..copy]);
    out
}

fn affine(w: &[f32], u: &[f32], b: &[f32], x: &[f32], m: &[f32], dim: usize) -> Vec<f32> {
    let mut z = matvec(w, x, dim);
    let um = matvec(u, m, dim);
    for i in 0..dim {
        z[i] += um[i] + b[i];
    }
    z
}

fn matvec(matrix: &[f32], vector: &[f32], rows: usize) -> Vec<f32> {
    let cols = vector.len();
    let mut out = vec![0.0; rows];
    for r in 0..rows {
        let mut sum = 0.0;
        for c in 0..cols {
            sum += matrix[r * cols + c] * vector[c];
        }
        out[r] = sum;
    }
    out
}

fn add_matvec_t(
    destination: &mut [f32],
    matrix: &[f32],
    grad_output: &[f32],
    rows: usize,
    cols: usize,
) {
    for r in 0..rows {
        for c in 0..cols {
            destination[c] += matrix[r * cols + c] * grad_output[r];
        }
    }
}

fn outer_add(destination: &mut [f32], left: &[f32], right: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            destination[r * cols + c] += left[r] * right[c];
        }
    }
}

fn stable_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = logits.iter().map(|v| (*v - max).exp()).collect();
    let sum: f32 = values.iter().sum();
    for v in &mut values {
        *v /= sum;
    }
    values
}

fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|v| (*v - max).exp()).sum();
    max + sum_exp.ln() - logits[target]
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}

fn xavier_uniform(out: usize, input: usize, rng: &mut XorShift64) -> Vec<f32> {
    let limit = (6.0f32 / (out + input) as f32).sqrt();
    (0..out * input)
        .map(|_| rng.next_f32() * 2.0 * limit - limit)
        .collect()
}

fn uniform(count: usize, scale: f32, rng: &mut XorShift64) -> Vec<f32> {
    (0..count)
        .map(|_| rng.next_f32() * 2.0 * scale - scale)
        .collect()
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
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

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn str(&mut self, value: &str) {
        self.u64(value.len() as u64);
        self.bytes.extend_from_slice(value.as_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn f32(&mut self, value: f32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("truncated .aimodel payload".into());
        }
        let result = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(result)
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn str(&mut self) -> Result<String, String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| format!("invalid utf-8: {e}"))
    }

    fn finished(&self) -> bool {
        self.pos == self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{AiNet, ModelConfig};
    use crate::optimizer::AdamW;
    use std::fs;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            architecture: "AiNet-v1.1".into(),
            model_id: "tiny-test".into(),
            vocab_size: 8,
            embedding_dim: 8,
            hidden_dim: 8,
            layer_count: 2,
            sequence_length: 8,
            seed: 12345,
        }
    }

    #[test]
    fn tiny_dataset_loss_decreases_and_weights_change() {
        let mut model = AiNet::new(tiny_config()).unwrap();
        let input = [1, 2, 3, 4, 5, 6, 7, 1];
        let target = [2, 3, 4, 5, 6, 7, 1, 2];
        let before = model.output_w.data.clone();
        let mut opt = AdamW::new(0.01, 0.0);

        let first = model.train_step(&input, &target, None).unwrap();
        opt.step(model.parameters_mut());
        let mut last = first;
        for _ in 0..120 {
            last = model.train_step(&input, &target, None).unwrap();
            opt.step(model.parameters_mut());
        }

        assert!(last.is_finite());
        assert!(last < first, "loss did not decrease: {first} -> {last}");
        assert_ne!(before, model.output_w.data, "weights did not change");
    }

    #[test]
    fn binary_model_round_trip_preserves_weights() {
        let mut model = AiNet::new(tiny_config()).unwrap();
        let input = [1, 2, 3, 4];
        let target = [2, 3, 4, 5];
        let mut opt = AdamW::new(0.01, 0.0);
        model.train_step(&input, &target, None).unwrap();
        opt.step(model.parameters_mut());

        let path = std::env::temp_dir().join("ainet-test.aimodel");
        model.save(&path).unwrap();
        let loaded = AiNet::load(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(model.config, loaded.config);
        assert_eq!(model.parameter_count(), loaded.parameter_count());
        assert_eq!(model.parameter_snapshot(), loaded.parameter_snapshot());
    }

    #[test]
    fn projection_and_embedding_gradients_match_numerical() {
        let mut model = AiNet::new(ModelConfig {
            architecture: "AiNet-v1.1".into(),
            model_id: "projection-test".into(),
            vocab_size: 8,
            embedding_dim: 5,
            hidden_dim: 7,
            layer_count: 1,
            sequence_length: 4,
            seed: 67890,
        })
        .unwrap();
        let input = [1usize, 2, 3, 4];
        let target = [2usize, 3, 4, 5];

        model.train_step(&input, &target, None).unwrap();

        let (projection_index, analytical_projection) = model
            .input_projection_w
            .as_ref()
            .unwrap()
            .grad
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, g)| (i, *g))
            .unwrap();

        let (embedding_index, analytical_embedding) = model
            .embedding
            .grad
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, g)| (i, *g))
            .unwrap();

        let eps = 1e-2f32;

        let original_projection = model.input_projection_w.as_ref().unwrap().data[projection_index];
        model.input_projection_w.as_mut().unwrap().data[projection_index] =
            original_projection + eps;
        let plus_projection = loss_without_grads(&mut model, &input, &target);
        model.input_projection_w.as_mut().unwrap().data[projection_index] =
            original_projection - eps;
        let minus_projection = loss_without_grads(&mut model, &input, &target);
        model.input_projection_w.as_mut().unwrap().data[projection_index] = original_projection;

        let numerical_projection = (plus_projection - minus_projection) / (2.0 * eps);
        let proj_den = analytical_projection
            .abs()
            .max(numerical_projection.abs())
            .max(1e-5);
        assert!(
            (analytical_projection - numerical_projection).abs() < 5e-4
                || (analytical_projection - numerical_projection).abs() / proj_den < 1e-1,
            "projection gradient mismatch: analytical={analytical_projection}, numerical={numerical_projection}"
        );

        let original_embedding = model.embedding.data[embedding_index];
        model.embedding.data[embedding_index] = original_embedding + eps;
        let plus_embedding = loss_without_grads(&mut model, &input, &target);
        model.embedding.data[embedding_index] = original_embedding - eps;
        let minus_embedding = loss_without_grads(&mut model, &input, &target);
        model.embedding.data[embedding_index] = original_embedding;

        let numerical_embedding = (plus_embedding - minus_embedding) / (2.0 * eps);
        let emb_den = analytical_embedding
            .abs()
            .max(numerical_embedding.abs())
            .max(1e-5);
        assert!(
            (analytical_embedding - numerical_embedding).abs() < 5e-4
                || (analytical_embedding - numerical_embedding).abs() / emb_den < 1e-1,
            "embedding gradient mismatch: analytical={analytical_embedding}, numerical={numerical_embedding}"
        );
    }

    #[test]
    fn adaptive_routing_uses_top_k_blocks_in_inference() {
        let mut model = AiNet::new(ModelConfig {
            architecture: "AiNet-v1.1".into(),
            model_id: "anr-test".into(),
            vocab_size: 8,
            embedding_dim: 8,
            hidden_dim: 128,
            layer_count: 1,
            sequence_length: 8,
            seed: 9876,
        })
        .unwrap();

        let _ = model.inference_logits(1).unwrap();
        let stats = model.routing_stats();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].total_blocks, 8);
        assert_eq!(stats[0].active_blocks, 6);
        assert_eq!(stats[0].active_channels, 96);
        assert_eq!(stats[0].skipped_channels, 32);
        assert!((stats[0].active_ratio - 0.75).abs() < 1e-6);
    }

    #[test]
    fn router_parameters_receive_training_gradients() {
        let mut model = AiNet::new(ModelConfig {
            architecture: "AiNet-v1.1".into(),
            model_id: "anr-gradient-test".into(),
            vocab_size: 8,
            embedding_dim: 8,
            hidden_dim: 128,
            layer_count: 1,
            sequence_length: 8,
            seed: 1357,
        })
        .unwrap();
        let input = [1usize, 2, 3, 4];
        let target = [2usize, 3, 4, 5];
        model.train_step(&input, &target, None).unwrap();
        let router_norm: f32 = model.cells[0]
            .router_w
            .grad
            .iter()
            .map(|value| value.abs())
            .sum::<f32>()
            + model.cells[0]
                .router_b
                .grad
                .iter()
                .map(|value| value.abs())
                .sum::<f32>();
        assert!(router_norm.is_finite());
        assert!(router_norm > 0.0);
    }

    #[test]
    fn deterministic_initialization_matches_for_equal_seed() {
        let a = AiNet::new(tiny_config()).unwrap();
        let b = AiNet::new(tiny_config()).unwrap();
        assert_eq!(a.parameter_snapshot(), b.parameter_snapshot());
    }

    #[test]
    fn legacy_v1_aimodel_loads_without_projection() {
        let model = AiNet::new(ModelConfig {
            architecture: "AiNet-v1".into(),
            model_id: "legacy-source".into(),
            vocab_size: 8,
            embedding_dim: 8,
            hidden_dim: 8,
            layer_count: 1,
            sequence_length: 8,
            seed: 24680,
        })
        .unwrap();

        let payload = model.encode_payload_v1().unwrap();
        let checksum = super::fnv1a64(&payload);
        let mut bytes = Vec::with_capacity(28 + payload.len());
        bytes.extend_from_slice(b"AIMDLv01");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&checksum.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);

        let path = std::env::temp_dir().join("ainet-legacy-v1-test.aimodel");
        std::fs::write(&path, bytes).unwrap();
        let loaded = AiNet::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.config.architecture, "AiNet-v1");
        assert_eq!(loaded.config.model_id, "legacy-0000000000006068");
        assert_eq!(loaded.parameter_count(), model.parameter_count());
        assert_eq!(loaded.parameter_snapshot(), model.parameter_snapshot());
        assert!(loaded.input_projection_w.is_none());
    }

    #[test]
    fn gradient_check_for_ai_cell() {
        let mut model = AiNet::new(tiny_config()).unwrap();
        let input = [1usize, 2];
        let target = [2usize, 3];

        model.train_step(&input, &target, None).unwrap();

        let (index, analytical) = model.cells[0]
            .w_keep
            .grad
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, g)| (i, *g))
            .expect("non-empty gradient");

        let original = model.cells[0].w_keep.data[index];
        let eps = 1e-2f32;
        model.cells[0].w_keep.data[index] = original + eps;
        let plus = loss_without_grads(&mut model, &input, &target);
        model.cells[0].w_keep.data[index] = original - eps;
        let minus = loss_without_grads(&mut model, &input, &target);
        model.cells[0].w_keep.data[index] = original;

        let numerical = (plus - minus) / (2.0 * eps);
        let absolute = (analytical - numerical).abs();
        let denom = analytical.abs().max(numerical.abs()).max(1e-5);
        let relative = absolute / denom;
        assert!(
            absolute < 5e-4 || relative < 1e-1,
            "gradient mismatch at index {index}: analytical={analytical}, numerical={numerical}, relative={relative}"
        );
    }

    fn loss_without_grads(model: &mut AiNet, input: &[usize], target: &[usize]) -> f32 {
        let mut memory = vec![vec![0.0; model.config.hidden_dim]; model.config.layer_count];
        let mut hidden = Vec::new();
        for &token in input {
            let embedding =
                super::embedding_row(&model.embedding.data, token, model.config.embedding_dim);
            let mut x = model.project_input(&embedding).unwrap();
            for layer in 0..model.config.layer_count {
                let cache = model.cells[layer].forward(&x, &memory[layer]);
                memory[layer] = cache.new_memory;
                x = x
                    .iter()
                    .zip(&cache.output_activation)
                    .map(|(a, b)| a + b)
                    .collect();
            }
            hidden.push(x);
        }
        let mut loss = 0.0;
        for (h, &target_id) in hidden.iter().zip(target) {
            let mut logits = super::matvec(&model.output_w.data, h, model.config.vocab_size);
            for v in 0..model.config.vocab_size {
                logits[v] += model.output_b.data[v];
            }
            loss += super::cross_entropy(&logits, target_id);
        }
        loss / input.len() as f32
    }
}
