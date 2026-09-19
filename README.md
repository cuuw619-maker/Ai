# Ai

Ai is a native Rust local neural-network project. The first architecture is **AiNet v1**, now extended with the trainable **AiNet v1.1** input projection.

This project trains its own parameters from random initialization. It does not load pretrained weights and does not use an LLM runtime, llama.cpp, GGUF, LoRA, QLoRA, fine-tuning, or remote AI APIs.

## Current Phase 2

The repository now contains a console-first training pipeline:

- custom f32 Tensor and neural operations;
- AiCell recurrent block with explicit BPTT;
- trainable embedding to hidden projection in AiNet v1.1;
- legacy loading of the original AiNet v1 / AIMDLv01 container;
- new versioned AIMDLv02 .aimodel container with model ID, dtype metadata and checksum;
- crash-safe model save with .prev rotation;
- custom byte-level BPE tokenizer with .aitok format;
- UTF-8-safe Russian, Ukrainian, English, digits, punctuation and emoji handling;
- streaming TXT/JSONL dataset reader;
- deterministic dataset IDs and resumable file/token cursors;
- fixed-size token buffering instead of loading a corpus into RAM;
- true gradient accumulation;
- global gradient clipping;
- serializable AdamW state;
- binary .aicheckpoint files with model, optimizer, state, cursor and dataset/tokenizer IDs;
- atomic checkpoint rotation with latest/previous fallback;
- training state machine and channel-based worker/events;
- cross-process CLI pause/stop control through a local command file;
- deterministic inference with argmax plus temperature/top-k/top-p sampling;
- metrics written incrementally to metrics.jsonl;
- bounded training-memory estimate and low-end defaults;
- Windows x64 CI with fmt, clippy, tests and release artifact.

## CLI

Create a new model from random weights:

    Ai.exe model create --name my-model --vocab 4096 --embedding 192 --hidden 192 --layers 6 --sequence 128 --seed 1 --output data/models/model.aimodel

Train the project tokenizer:

    Ai.exe tokenizer train --dataset corpus.txt --format txt --vocab 4096 --output data/tokenizer/tokenizer.aitok

Validate a dataset:

    Ai.exe dataset validate --dataset corpus.jsonl --format jsonl --tokenizer data/tokenizer/tokenizer.aitok

Start real training:

    Ai.exe train --model data/models/model.aimodel --tokenizer data/tokenizer/tokenizer.aitok --dataset corpus.txt --format txt

The low-end profile defaults to 2 CPU threads, micro-batch 1, sequence 128, gradient accumulation 8 and a 3 GB training memory budget. Neural matrix operations remain intentionally simple and CPU-oriented at this stage.

Control a running CLI trainer from another process:

    Ai.exe train pause
    Ai.exe train stop
    Ai.exe train status

Resume from a saved checkpoint explicitly:

    Ai.exe train resume --checkpoint data/runs/run-000001/checkpoints/checkpoint-latest.aicheckpoint --tokenizer data/tokenizer/tokenizer.aitok --dataset corpus.txt --format txt

List checkpoints:

    Ai.exe checkpoint list
    Ai.exe checkpoint list --run run-000001

Generate using the same project-owned model and tokenizer:

    Ai.exe generate --model data/models/model.aimodel --tokenizer data/tokenizer/tokenizer.aitok --prompt "Привет" --tokens 32 --temperature 0.8 --top-k 40 --top-p 0.9

Use --greedy for deterministic argmax generation.

## Dataset format

TXT uses one training sample per line.

JSONL supports:

    {"text":"Привет мир"}

and dialogue records:

    {"user":"Привет","assistant":"Здравствуйте!"}

Dialogue records are converted to the project-owned special tokens <|system|>, <|user|> and <|assistant|>.

The dataset reader streams records. Validation and dataset ID calculation also operate incrementally over the file.

## Resume guarantees

A checkpoint records the dataset ID, tokenizer ID, file offset, sample index, token position, model checksum, optimizer moments, optimizer step, training counters, configuration and RNG state.

A resume attempt against a different dataset revision or tokenizer is rejected instead of silently continuing on incompatible data.

## Tests

Run locally:

    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo test --all-targets
    cargo run --release -- benchmark

The regression suite covers neural loss reduction, weight mutation, numerical gradients, deterministic initialization, .aimodel round-trip, legacy AIMDLv01 loading, tokenizer UTF-8 round-trip and checksum validation, exact dataset cursor resume, and checkpoint continuation equivalence.

## Project status

The next stages are richer resource monitoring, fuller evaluation, persistent SQLite memory, native UI and final Windows application UX. These remain after the neural and training correctness layer.
