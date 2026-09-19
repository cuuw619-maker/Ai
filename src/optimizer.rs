use crate::neural::Parameter;

pub struct AdamW {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    step: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            step: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step<'a, I>(&mut self, parameters: I)
    where
        I: IntoIterator<Item = &'a mut Parameter>,
    {
        let mut params: Vec<&mut Parameter> = parameters.into_iter().collect();
        if self.m.len() != params.len() {
            self.m = params.iter().map(|p| vec![0.0; p.len()]).collect();
            self.v = params.iter().map(|p| vec![0.0; p.len()]).collect();
            self.step = 0;
        }

        self.step += 1;
        let bias1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias2 = 1.0 - self.beta2.powi(self.step as i32);

        for (index, param) in params.iter_mut().enumerate() {
            for i in 0..param.data.len() {
                let g = param.grad[i];
                self.m[index][i] = self.beta1 * self.m[index][i] + (1.0 - self.beta1) * g;
                self.v[index][i] = self.beta2 * self.v[index][i] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[index][i] / bias1;
                let v_hat = self.v[index][i] / bias2;
                param.data[i] -= self.learning_rate
                    * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * param.data[i]);
                param.grad[i] = 0.0;
            }
        }
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn state(&self) -> (&[Vec<f32>], &[Vec<f32>]) {
        (&self.m, &self.v)
    }
}
