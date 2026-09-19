# Ai

Ai is a from-scratch local neural-network platform. The first architecture is AiNet v1, built around the project-specific AiCell recurrent block.

The repository starts from zero. AiNet v1 does not load pretrained weights, does not use an existing LLM runtime, and does not use a neural-network framework as its training engine.

Current phase 1 includes:
- Rust native project.
- Tensor with shape/stride and optional gradient storage.
- Basic tensor math: matrix multiplication, reductions, activations, softmax, cross-entropy.
- Custom deterministic random initialization.
- AiCell keep/write/candidate gates and recurrent memory.
- Explicit truncated BPTT over an unrolled sequence.
- Manual AdamW optimizer.
- Custom .aimodel binary container with version and checksum.
- Tiny overfit test, parameter mutation check, numerical gradient check.
- Model save/load round-trip.
- Minimal CLI that trains, saves, reloads, and generates token IDs.

Build with:
    cargo test
    cargo run --release

The default CLI uses a deliberately tiny model so the neural loop can be tested on low-end CPUs.

The .aimodel container stores the architecture identifier, dimensions, seed, parameter names, parameter sizes, and raw f32 weights. Large model files are intentionally not committed.

Next stages are the custom tokenizer, streaming datasets, trainer/checkpoint management, pause/resume, inference/chat, SQLite memory, evaluation, native UI, and Windows packaging.

Development rule: measured neural behavior first, UI second.
