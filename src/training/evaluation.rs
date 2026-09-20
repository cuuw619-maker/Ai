use crate::model::AiNet;

pub trait Evaluator {
    fn evaluate(&mut self, model: &AiNet) -> Result<f32, String>;
}

pub struct TinyEvaluator {
    input: Vec<usize>,
    target: Vec<usize>,
}

impl TinyEvaluator {
    pub fn new(input: Vec<usize>, target: Vec<usize>) -> Result<Self, String> {
        if input.is_empty() || input.len() != target.len() {
            return Err("tiny evaluator requires equal non-empty input and target".into());
        }
        Ok(Self { input, target })
    }
}

impl Evaluator for TinyEvaluator {
    fn evaluate(&mut self, model: &AiNet) -> Result<f32, String> {
        model
            .sequence_loss(&self.input, &self.target, None)
            .map(|(loss, _)| loss)
    }
}

#[cfg(test)]
mod tests {
    use super::{Evaluator, TinyEvaluator};
    use crate::model::{AiNet, ModelConfig};

    #[test]
    fn tiny_evaluator_does_not_change_weights() {
        let model = AiNet::new(ModelConfig {
            architecture: "AiNet-v1.1".into(),
            model_id: "evaluator-test".into(),
            vocab_size: 8,
            embedding_dim: 4,
            hidden_dim: 4,
            layer_count: 1,
            sequence_length: 4,
            seed: 3,
        })
        .unwrap();
        let before = model.weights_checksum();
        let mut evaluator = TinyEvaluator::new(vec![1, 2, 3, 4], vec![2, 3, 4, 5]).unwrap();
        let loss = evaluator.evaluate(&model).unwrap();
        assert!(loss.is_finite());
        assert_eq!(before, model.weights_checksum());
    }
}
