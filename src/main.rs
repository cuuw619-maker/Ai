mod neural;
mod model;
mod optimizer;

use model::{AiNet, ModelConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Ai / AiNet v1 — from-scratch CPU neural core");
    println!("No pretrained weights are loaded.");

    let mut model = AiNet::new(ModelConfig {
        architecture: "AiNet-v1".to_owned(),
        vocab_size: 16,
        embedding_dim: 8,
        hidden_dim: 8,
        layer_count: 2,
        sequence_length: 8,
        seed: 0xA11CE001,
    })?;

    let inputs = [1usize, 2, 3, 4, 5, 6, 7, 1];
    let targets = [2usize, 3, 4, 5, 6, 7, 1, 2];

    let mut optimizer = optimizer::AdamW::new(0.01, 0.0);
    let initial_loss = model.train_step(&inputs, &targets, None)?;
    optimizer.step(model.parameters_mut());

    println!("initial_loss={initial_loss:.6}");

    let steps = std::env::args()
        .position(|a| a == "--steps")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(100);

    for step in 1..=steps {
        let loss = model.train_step(&inputs, &targets, None)?;
        optimizer.step(model.parameters_mut());
        if step == 1 || step % 10 == 0 || step == steps {
            println!("step={step:04} loss={loss:.6}");
        }
    }

    let path = "model.aimodel";
    model.save(path)?;
    println!("saved={path}");

    let mut loaded = AiNet::load(path)?;
    loaded.reset_state();
    let generated = loaded.generate_tokens(&[1, 2, 3], 12)?;
    println!("generated_ids={generated:?}");
    println!("tiny_cycle=PASS");

    Ok(())
}
