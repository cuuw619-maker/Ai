use crate::neural::Parameter;

#[derive(Clone, Debug, PartialEq)]
pub struct AdamWState {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub step: u64,
    pub parameter_names: Vec<String>,
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

pub struct AdamW {
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    step: u64,
    parameter_names: Vec<String>,
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
            parameter_names: Vec::new(),
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn step<'a, I>(&mut self, parameters: I)
    where
        I: IntoIterator<Item = &'a mut Parameter>,
    {
        let mut params: Vec<&mut Parameter> = parameters.into_iter().collect();

        if self.m.is_empty() {
            self.parameter_names = params.iter().map(|p| p.name.clone()).collect();
            self.m = params.iter().map(|p| vec![0.0; p.len()]).collect();
            self.v = params.iter().map(|p| vec![0.0; p.len()]).collect();
        } else {
            assert_eq!(self.m.len(), params.len(), "AdamW parameter count changed");
            for (index, param) in params.iter().enumerate() {
                assert_eq!(
                    self.parameter_names[index], param.name,
                    "AdamW parameter order changed"
                );
                assert_eq!(
                    self.m[index].len(),
                    param.len(),
                    "AdamW parameter size changed"
                );
            }
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

    pub fn export_state<'a, I>(&self, parameters: I) -> Result<AdamWState, String>
    where
        I: IntoIterator<Item = &'a Parameter>,
    {
        let params: Vec<&Parameter> = parameters.into_iter().collect();
        if self.m.is_empty() {
            return Ok(AdamWState {
                learning_rate: self.learning_rate,
                weight_decay: self.weight_decay,
                beta1: self.beta1,
                beta2: self.beta2,
                epsilon: self.epsilon,
                step: self.step,
                parameter_names: params.iter().map(|p| p.name.clone()).collect(),
                m: params.iter().map(|p| vec![0.0; p.len()]).collect(),
                v: params.iter().map(|p| vec![0.0; p.len()]).collect(),
            });
        }
        self.validate_parameters(&params)?;
        Ok(AdamWState {
            learning_rate: self.learning_rate,
            weight_decay: self.weight_decay,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            step: self.step,
            parameter_names: self.parameter_names.clone(),
            m: self.m.clone(),
            v: self.v.clone(),
        })
    }

    pub fn load_state<'a, I>(&mut self, state: AdamWState, parameters: I) -> Result<(), String>
    where
        I: IntoIterator<Item = &'a Parameter>,
    {
        let params: Vec<&Parameter> = parameters.into_iter().collect();
        if state.parameter_names.len() != params.len()
            || state.m.len() != params.len()
            || state.v.len() != params.len()
        {
            return Err("AdamW state parameter count mismatch".into());
        }
        for (index, param) in params.iter().enumerate() {
            if state.parameter_names[index] != param.name {
                return Err(format!(
                    "AdamW parameter mismatch at {index}: checkpoint={} model={}",
                    state.parameter_names[index], param.name
                ));
            }
            if state.m[index].len() != param.len() || state.v[index].len() != param.len() {
                return Err(format!(
                    "AdamW parameter length mismatch for {}",
                    param.name
                ));
            }
        }
        self.learning_rate = state.learning_rate;
        self.weight_decay = state.weight_decay;
        self.beta1 = state.beta1;
        self.beta2 = state.beta2;
        self.epsilon = state.epsilon;
        self.step = state.step;
        self.parameter_names = state.parameter_names;
        self.m = state.m;
        self.v = state.v;
        Ok(())
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    pub fn state(&self) -> (&[Vec<f32>], &[Vec<f32>]) {
        (&self.m, &self.v)
    }

    pub fn validate_parameters(&self, params: &[&Parameter]) -> Result<(), String> {
        if self.parameter_names.len() != params.len() {
            return Err("AdamW parameter count mismatch".into());
        }
        for (index, param) in params.iter().enumerate() {
            if self.parameter_names[index] != param.name {
                return Err(format!("AdamW parameter mismatch at {index}"));
            }
            if self.m[index].len() != param.len() || self.v[index].len() != param.len() {
                return Err(format!(
                    "AdamW parameter length mismatch for {}",
                    param.name
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::AdamW;
    use crate::neural::Parameter;

    #[test]
    fn state_round_trip_preserves_optimizer_step() {
        let mut p = Parameter::new("p", vec![1.0, -2.0]);
        p.grad.copy_from_slice(&[0.5, -0.25]);
        let mut opt = AdamW::new(0.01, 0.001);
        opt.step([&mut p]);

        let state = opt.export_state([&p]).unwrap();
        let mut restored = AdamW::new(0.5, 0.5);
        restored.load_state(state.clone(), [&p]).unwrap();

        assert_eq!(restored.step_count(), state.step);
        assert_eq!(restored.parameter_names(), state.parameter_names.as_slice());
        assert_eq!(restored.state().0, state.m.as_slice());
        assert_eq!(restored.state().1, state.v.as_slice());
    }
}
