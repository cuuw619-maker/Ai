#![allow(clippy::needless_range_loop)]
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub grad: Option<Box<Tensor>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Result<Self, String> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(format!(
                "tensor data length {} does not match shape {:?}",
                data.len(),
                shape
            ));
        }
        let mut stride = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        Ok(Self {
            data,
            shape: shape.to_vec(),
            stride,
            grad: None,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(vec![0.0; shape.iter().product()], shape).expect("valid zero tensor shape")
    }

    pub fn from_scalar(value: f32) -> Self {
        Self::new(vec![value], &[]).expect("scalar tensor")
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn reshape(&self, shape: &[usize]) -> Result<Self, String> {
        Self::new(self.data.clone(), shape)
    }

    fn check_same_shape(&self, rhs: &Self) -> Result<(), String> {
        if self.shape != rhs.shape {
            return Err(format!(
                "shape mismatch: {:?} vs {:?}",
                self.shape, rhs.shape
            ));
        }
        Ok(())
    }

    fn binary<F>(&self, rhs: &Self, op: F) -> Result<Self, String>
    where
        F: Fn(f32, f32) -> f32,
    {
        self.check_same_shape(rhs)?;
        let data = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| op(*a, *b))
            .collect();
        Self::new(data, &self.shape)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, String> {
        self.binary(rhs, |a, b| a + b)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self.binary(rhs, |a, b| a - b)
    }

    pub fn mul_elem(&self, rhs: &Self) -> Result<Self, String> {
        self.binary(rhs, |a, b| a * b)
    }

    pub fn div_elem(&self, rhs: &Self) -> Result<Self, String> {
        self.binary(rhs, |a, b| a / b)
    }

    pub fn transpose_2d(&self) -> Result<Self, String> {
        if self.shape.len() != 2 {
            return Err("transpose_2d requires rank 2".into());
        }
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut out = vec![0.0; self.numel()];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = self.data[r * cols + c];
            }
        }
        Self::new(out, &[cols, rows])
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err("matmul requires two rank-2 tensors".into());
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (rhs.shape[0], rhs.shape[1]);
        if k != k2 {
            return Err(format!("matmul inner dimension mismatch: {k} vs {k2}"));
        }
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                for j in 0..n {
                    out[i * n + j] += a * rhs.data[p * n + j];
                }
            }
        }
        Self::new(out, &[m, n])
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.numel().max(1) as f32
    }

    pub fn sqrt(&self) -> Self {
        Self::new(self.data.iter().map(|v| v.sqrt()).collect(), &self.shape).expect("same shape")
    }

    pub fn exp(&self) -> Self {
        Self::new(self.data.iter().map(|v| v.exp()).collect(), &self.shape).expect("same shape")
    }

    pub fn log(&self) -> Self {
        Self::new(self.data.iter().map(|v| v.ln()).collect(), &self.shape).expect("same shape")
    }

    pub fn sigmoid(&self) -> Self {
        Self::new(
            self.data
                .iter()
                .map(|x| {
                    if *x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let e = x.exp();
                        e / (1.0 + e)
                    }
                })
                .collect(),
            &self.shape,
        )
        .expect("same shape")
    }

    pub fn tanh(&self) -> Self {
        Self::new(self.data.iter().map(|v| v.tanh()).collect(), &self.shape).expect("same shape")
    }

    pub fn softmax(&self) -> Result<Self, String> {
        match self.shape.as_slice() {
            [n] => softmax_row(&self.data, *n),
            [rows, cols] => {
                let mut out = vec![0.0; self.data.len()];
                for r in 0..*rows {
                    let row = &self.data[r * cols..(r + 1) * cols];
                    let sm = softmax_row(row, *cols)?;
                    out[r * cols..(r + 1) * cols].copy_from_slice(&sm.data);
                }
                Self::new(out, &self.shape)
            }
            _ => Err("softmax supports only rank-1 or rank-2 tensors".into()),
        }
    }

    pub fn cross_entropy(&self, target: usize) -> Result<f32, String> {
        if self.shape.len() != 1 || target >= self.shape[0] {
            return Err("cross_entropy expects a rank-1 logits tensor and valid target".into());
        }
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = self.data.iter().map(|v| (*v - max).exp()).sum();
        let log_z = max + sum_exp.ln();
        Ok(-(self.data[target] - log_z))
    }
}

fn softmax_row(row: &[f32], cols: usize) -> Result<Tensor, String> {
    if row.len() != cols || cols == 0 {
        return Err("invalid softmax row".into());
    }
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = Vec::with_capacity(cols);
    let mut total = 0.0;
    for value in row {
        let e = (*value - max).exp();
        exps.push(e);
        total += e;
    }
    for value in &mut exps {
        *value /= total;
    }
    Tensor::new(exps, &[cols])
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, numel={})", self.shape, self.numel())
    }
}

#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn matrix_multiplication_is_correct() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let p = x.softmax().unwrap();
        assert!((p.sum() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_is_finite() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert!(x.cross_entropy(2).unwrap().is_finite());
    }
}
