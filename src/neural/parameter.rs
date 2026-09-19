#[derive(Clone, Debug)]
pub struct Parameter {
    pub name: String,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Parameter {
    pub fn new(name: impl Into<String>, data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            name: name.into(),
            data,
            grad: vec![0.0; len],
        }
    }

    pub fn zeros(name: impl Into<String>, len: usize) -> Self {
        Self::new(name, vec![0.0; len])
    }

    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::Parameter;

    #[test]
    fn parameter_gradient_matches_storage() {
        let mut p = Parameter::new("p", vec![1.0, 2.0, 3.0]);
        p.grad[1] = 4.0;
        assert_eq!(p.len(), p.grad.len());
        p.zero_grad();
        assert!(p.grad.iter().all(|v| *v == 0.0));
    }
}
