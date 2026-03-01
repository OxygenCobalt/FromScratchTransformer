use std::io::{self, Write};

use crate::tensor::cpu2::{FillUninit, Tensor};

pub struct Activation {
    function: ActivationFn,
    dropout: f64,
    mask: Option<Tensor>,
    input_shape: Vec<usize>,
}

impl Activation {
    pub fn new(input_shape: Vec<usize>, function: ActivationFn, dropout: f64) -> Self {
        Self {
            function,
            dropout,
            mask: None,
            input_shape,
        }
    }

    pub fn prepare(&mut self, train: bool, batch: usize) {
        if !train || self.dropout <= 0.0 {
            self.mask = None;
            return;
        }
        let mut shape = self.input_shape.clone();
        shape.push(batch);
        let mut mask = Tensor::init(unsafe { FillUninit::new(shape) }).unwrap();
        let classes = mask.shape[0];
        let cols = mask.data.len() / classes;
        let flat_mask = mask.r_mut(&[classes, cols]).unwrap();
        let mask_data = flat_mask.tensor.data.as_mut_slice();
        let mut idx = 0;
        for _ in 0..classes {
            let x = if rand::random_range(0.0..1.0) < self.dropout {
                0.0
            } else {
                1.0
            };
            for _ in 0..cols {
                mask_data[idx] = x;
                idx += flat_mask.stride[1];
            }
            idx -= flat_mask.stride[1] * cols;
            idx += flat_mask.stride[0];
        }
        self.mask = Some(mask);
    }

    pub fn forward(&self, mut a_in: Tensor) -> Tensor {
        match self.function {
            ActivationFn::Sigmoid => {
                self.arithmetic_forwards(&mut a_in, |x| 1.0 / (1.0 + (-x).exp()))
            }
            ActivationFn::ReLU => {
                self.arithmetic_forwards(&mut a_in, |x| if x > 0.0 { x } else { 0.0 })
            }
            ActivationFn::SiLU => self.arithmetic_forwards(&mut a_in, |x| x / (1.0 + (-x).exp())),
            ActivationFn::Softmax => self.softmax_forwards(&mut a_in),
            ActivationFn::Tanh => self.arithmetic_forwards(&mut a_in, |x| x.tanh()),
        }
        a_in
    }

    #[inline(always)]
    fn arithmetic_forwards(&self, a_in: &mut Tensor, block: impl Fn(f64) -> f64) {
        let base = a_in.data.iter_mut();
        match self.mask {
            Some(ref mask) => {
                base.zip(mask.data.iter())
                    .for_each(|(v, m)| *v = block(*v) * *m);
            }
            None => base.for_each(|v| *v = block(*v)),
        }
    }

    fn softmax_forwards(&self, a_in: &mut Tensor) {
        // what we want to do is be able to take the softmax
        // while preserving many-dimensional structures, so
        // we flatten down to [classes, cols] where cols is
        // just the flattened remaining dimensions that we
        // all factor into the norm.
        let classes = a_in.shape[0];
        let cols = a_in.data.len() / classes;
        let a_in_flat = a_in.r_mut(&[classes, cols]).unwrap();
        let mut idx = 0;
        let a_in_flat_data = a_in_flat.tensor.data.as_mut_slice();
        for _ in 0..cols {
            let mut max = f64::NEG_INFINITY;
            let mut max_idx = idx;
            for _ in 0..classes {
                let x = a_in_flat_data[max_idx];
                if x > max {
                    max = x;
                }
                max_idx += a_in_flat.stride[0];
            }
            let mut norm = 0.0;
            let mut pass1_idx = idx;
            for _ in 0..classes {
                a_in_flat_data[pass1_idx] = (a_in_flat_data[pass1_idx] - max).exp();
                norm += a_in_flat_data[pass1_idx];
                pass1_idx += a_in_flat.stride[0];
            }
            match self.mask {
                Some(ref mask) => {
                    let mut pass2_idx = idx;
                    for _ in 0..classes {
                        a_in_flat_data[pass2_idx] /= norm;
                        a_in_flat_data[pass2_idx] *= mask.data[pass2_idx];
                        pass2_idx += a_in_flat.stride[0];
                    }
                }
                None => {
                    let mut pass2_idx = idx;
                    for _ in 0..classes {
                        a_in_flat_data[pass2_idx] /= norm;
                        pass2_idx += a_in_flat.stride[0];
                    }
                }
            }
            idx += a_in_flat.stride[1];
        }
    }

    pub fn backward(&mut self, c: f64, a_in: Tensor, grad: Tensor) -> Tensor {
        match self.function {
            ActivationFn::Sigmoid => self.arithmetic_backwards(a_in, grad, |v, g| {
                let sig = 1.0 / (1.0 + (-v).exp());
                sig * (1.0 - sig) * (g)
            }),
            ActivationFn::ReLU => {
                self.arithmetic_backwards(a_in, grad, |v, g| if v > 0.0 { g } else { 0.0 })
            }
            ActivationFn::SiLU => self.arithmetic_backwards(a_in, grad, |v, g| {
                let sig = 1.0 / (1.0 + (-v).exp());
                (sig + v * sig * (1.0 - sig)) * (g)
            }),
            ActivationFn::Softmax => self.softmax_backwards(a_in, grad),
            ActivationFn::Tanh => self.arithmetic_backwards(a_in, grad, |v, g| {
                let t = v.tanh();
                (1.0 - t * t) * (g)
            }),
        }
    }

    #[inline(always)]
    pub fn arithmetic_backwards(
        &self,
        a_in: Tensor,
        mut grad: Tensor,
        block: impl Fn(f64, f64) -> f64,
    ) -> Tensor {
        let base = a_in.data.iter().zip(grad.data.iter_mut());
        match self.mask {
            Some(ref mask) => base
                .zip(mask.data.as_slice())
                .for_each(|((v, g), m)| *g = block(*v, *g) * *m),
            None => base.for_each(|(v, g)| *g = block(*v, *g)),
        }
        grad
    }

    pub fn softmax_backwards(&self, a_in: Tensor, grad: Tensor) -> Tensor {
        let orig = a_in.shape.to_vec();
        let classes = a_in.shape[0];
        let cols = a_in.data.len() / classes;
        let softmax = self.forward(a_in);
        let flat_softmax = softmax.r(&[classes, cols]).unwrap().materialize();
        let mut flat_grad = grad.r(&[classes, cols]).unwrap().materialize();
        // step 1: calculate sum(s * g) across cols
        let mut sum_s_dot_g =
            Tensor::init(unsafe { FillUninit::new(vec![classes, cols]) }).unwrap();
        let mut flat_idx = 0;
        for _ in 0..cols {
            let mut sum = 0.0;
            for _ in 0..classes {
                sum += flat_grad.data[flat_idx] * flat_softmax.data[flat_idx];
                flat_idx += flat_grad.stride[0];
            }
            flat_idx -= flat_grad.stride[0] * classes;
            for _ in 0..classes {
                sum_s_dot_g.data[flat_idx] = sum;
                flat_idx += flat_grad.stride[0];
            }
            flat_idx -= flat_grad.stride[0] * classes;
            flat_idx += flat_grad.stride[1];
        }
        // calculate (g - sum(s * g)) * s
        let base = flat_grad
            .data
            .iter_mut()
            .zip(sum_s_dot_g.data.iter())
            .zip(flat_softmax.data.iter());
        match self.mask {
            Some(ref mask) => {
                base.zip(mask.data.iter())
                    .for_each(|(((g, ssdg), s), m)| *g = (*g * *s - *s * ssdg) * *m);
            }
            None => {
                base.for_each(|((g, ssdg), s)| *g = *g * *s - *s * ssdg);
            }
        }
        return flat_grad.r(&orig).unwrap().materialize();
    }

    pub fn read(read: &mut impl io::Read) -> io::Result<Self> {
        let mut shape_len_bytes = [0u8; 8];
        read.read_exact(&mut shape_len_bytes)?;
        let shape_len = usize::from_le_bytes(shape_len_bytes);
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_bytes = [0u8; 8];
            read.read_exact(&mut dim_bytes)?;
            shape.push(usize::from_le_bytes(dim_bytes));
        }
        let mut id = [0u8; 8];
        read.read_exact(&mut id)?;
        let function = ActivationFn::from_id(&id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "invalid activation function signature",
            )
        })?;
        let mut dropout_bytes = [0u8; 8];
        read.read_exact(&mut dropout_bytes)?;
        let dropout = f64::from_le_bytes(dropout_bytes);
        Ok(Self::new(shape, function, dropout))
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(&self.input_shape.len().to_le_bytes())?;
        for dim in &self.input_shape {
            write.write_all(&dim.to_le_bytes())?;
        }
        write.write_all(self.function.id())?;
        write.write_all(&self.dropout.to_le_bytes())?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum ActivationFn {
    Sigmoid,
    ReLU,
    SiLU,
    Softmax,
    Tanh,
}

impl ActivationFn {
    pub fn id(&self) -> &'static [u8; 8] {
        match self {
            Self::Sigmoid => b"ActvSigm",
            Self::ReLU => b"ActvReLU",
            Self::SiLU => b"ActvSiLU",
            Self::Softmax => b"ActvSfmx",
            Self::Tanh => b"ActvTanh",
        }
    }

    pub fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"ActvSigm" => Some(Self::Sigmoid),
            b"ActvReLU" => Some(Self::ReLU),
            b"ActvSiLU" => Some(Self::SiLU),
            b"ActvSfmx" => Some(Self::Softmax),
            b"ActvTanh" => Some(Self::Tanh),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::cpu2::Fill;

    fn tensor_with_data(shape: Vec<usize>, values: &[f64]) -> Tensor {
        let mut t = Tensor::init(Fill { shape, with: 0.0 }).unwrap();
        assert_eq!(t.data.len(), values.len(), "shape/data length mismatch");
        t.data.as_mut_slice().clone_from_slice(values);
        t
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "index {}: expected {}, got {} (diff {})",
                i,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn softmax(values: &[f64]) -> Vec<f64> {
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = values.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    // ---- Forward pass tests ----

    #[test]
    fn test_sigmoid_forward() {
        let act = Activation::new(vec![5], ActivationFn::Sigmoid, 0.0);
        let vals = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = tensor_with_data(vec![5], &vals);
        let output = act.forward(input);
        let expected: Vec<f64> = vals.iter().map(|&x| sigmoid(x)).collect();
        assert_eq!(output.shape, vec![5]);
        assert_close(output.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn test_relu_forward() {
        let act = Activation::new(vec![5], ActivationFn::ReLU, 0.0);
        let input = tensor_with_data(vec![5], &[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let output = act.forward(input);
        assert_eq!(output.shape, vec![5]);
        assert_close(output.data.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0], 1e-12);
    }

    #[test]
    fn test_silu_forward() {
        let act = Activation::new(vec![5], ActivationFn::SiLU, 0.0);
        let vals = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = tensor_with_data(vec![5], &vals);
        let output = act.forward(input);
        let expected: Vec<f64> = vals.iter().map(|&x| x * sigmoid(x)).collect();
        assert_close(output.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn test_tanh_forward() {
        let act = Activation::new(vec![5], ActivationFn::Tanh, 0.0);
        let vals = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = tensor_with_data(vec![5], &vals);
        let output = act.forward(input);
        let expected: Vec<f64> = vals.iter().map(|&x| x.tanh()).collect();
        assert_close(output.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn test_softmax_forward_1d() {
        let act = Activation::new(vec![3], ActivationFn::Softmax, 0.0);
        let vals = [1.0, 2.0, 3.0];
        let input = tensor_with_data(vec![3], &vals);
        let output = act.forward(input);
        let expected = softmax(&vals);
        assert_eq!(output.shape, vec![3]);
        assert_close(output.data.as_slice(), &expected, 1e-12);
        let total: f64 = output.data.as_slice().iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "softmax should sum to 1");
    }

    #[test]
    fn test_softmax_forward_2d() {
        // Shape [3, 2]: 3 classes, 2 columns. Softmax along dim 0 per column.
        // Data layout row-major: [x[0,0], x[0,1], x[1,0], x[1,1], x[2,0], x[2,1]]
        // Column 0 = indices 0,2,4 = [1.0, 2.0, 0.0]
        // Column 1 = indices 1,3,5 = [0.0, 0.0, 3.0]
        let act = Activation::new(vec![3, 2], ActivationFn::Softmax, 0.0);
        let input = tensor_with_data(vec![3, 2], &[1.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
        let output = act.forward(input);

        let col0_s = softmax(&[1.0, 2.0, 0.0]);
        let col1_s = softmax(&[0.0, 0.0, 3.0]);
        // Interleaved back into row-major
        let expected = [
            col0_s[0], col1_s[0], col0_s[1], col1_s[1], col0_s[2], col1_s[2],
        ];
        assert_close(output.data.as_slice(), &expected, 1e-12);

        // Each column should sum to 1
        let col0_total = output.data[0] + output.data[2] + output.data[4];
        let col1_total = output.data[1] + output.data[3] + output.data[5];
        assert!((col0_total - 1.0).abs() < 1e-12);
        assert!((col1_total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow exp() without max subtraction
        let act = Activation::new(vec![3], ActivationFn::Softmax, 0.0);
        let input = tensor_with_data(vec![3], &[1000.0, 1001.0, 1002.0]);
        let output = act.forward(input);
        // Due to shift-invariance, same as softmax([0, 1, 2])
        let expected = softmax(&[0.0, 1.0, 2.0]);
        assert_close(output.data.as_slice(), &expected, 1e-10);
        assert!(output.data.as_slice().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_softmax_uniform_input() {
        let act = Activation::new(vec![4], ActivationFn::Softmax, 0.0);
        let input = tensor_with_data(vec![4], &[1.0, 1.0, 1.0, 1.0]);
        let output = act.forward(input);
        // Uniform input -> uniform output
        assert_close(output.data.as_slice(), &[0.25, 0.25, 0.25, 0.25], 1e-12);
    }

    // ---- Backward pass tests ----

    #[test]
    fn test_sigmoid_backward() {
        let mut act = Activation::new(vec![4], ActivationFn::Sigmoid, 0.0);
        let vals = [-1.0, 0.0, 1.0, 2.0];
        let a_in = tensor_with_data(vec![4], &vals);
        let grad = tensor_with_data(vec![4], &[1.0, 1.0, 1.0, 1.0]);
        let result = act.backward(1.0, a_in, grad);
        let expected: Vec<f64> = vals
            .iter()
            .map(|&x| {
                let s = sigmoid(x);
                s * (1.0 - s)
            })
            .collect();
        assert_close(result.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn test_sigmoid_backward_scaled_grad() {
        let mut act = Activation::new(vec![3], ActivationFn::Sigmoid, 0.0);
        let vals = [0.0, 1.0, -1.0];
        let grads = [2.0, 0.5, -1.0];
        let a_in = tensor_with_data(vec![3], &vals);
        let grad = tensor_with_data(vec![3], &grads);
        let result = act.backward(1.0, a_in, grad);
        let expected: Vec<f64> = vals
            .iter()
            .zip(grads.iter())
            .map(|(&x, &g)| {
                let s = sigmoid(x);
                g * s * (1.0 - s)
            })
            .collect();
        assert_close(result.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn test_relu_backward() {
        let mut act = Activation::new(vec![5], ActivationFn::ReLU, 0.0);
        let a_in = tensor_with_data(vec![5], &[-2.0, -0.5, 0.0, 0.5, 2.0]);
        let grad = tensor_with_data(vec![5], &[1.0, 1.0, 1.0, 1.0, 1.0]);
        let result = act.backward(1.0, a_in, grad);
        // ReLU'(x) = 1 if x > 0, 0 otherwise (0 at x=0)
        assert_close(result.data.as_slice(), &[0.0, 0.0, 0.0, 1.0, 1.0], 1e-12);
    }

    #[test]
    fn test_relu_backward_scaled_grad() {
        let mut act = Activation::new(vec![4], ActivationFn::ReLU, 0.0);
        let a_in = tensor_with_data(vec![4], &[-1.0, 1.0, -0.5, 2.0]);
        let grad = tensor_with_data(vec![4], &[3.0, 2.0, -1.0, 0.5]);
        let result = act.backward(1.0, a_in, grad);
        assert_close(result.data.as_slice(), &[0.0, 2.0, 0.0, 0.5], 1e-12);
    }

    #[test]
    fn test_silu_backward() {
        let mut act = Activation::new(vec![4], ActivationFn::SiLU, 0.0);
        let vals = [-1.0, 0.0, 1.0, 2.0];
        let a_in = tensor_with_data(vec![4], &vals);
        let grad = tensor_with_data(vec![4], &[1.0, 1.0, 1.0, 1.0]);
        let result = act.backward(1.0, a_in, grad);
        // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let expected: Vec<f64> = vals
            .iter()
            .map(|&x| {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            })
            .collect();
        assert_close(result.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn test_tanh_backward() {
        let mut act = Activation::new(vec![4], ActivationFn::Tanh, 0.0);
        let vals = [-1.0, 0.0, 1.0, 2.0];
        let a_in = tensor_with_data(vec![4], &vals);
        let grad = tensor_with_data(vec![4], &[1.0, 1.0, 1.0, 1.0]);
        let result = act.backward(1.0, a_in, grad);
        // tanh'(x) = 1 - tanh(x)^2
        let expected: Vec<f64> = vals
            .iter()
            .map(|&x| {
                let t = x.tanh();
                1.0 - t * t
            })
            .collect();
        assert_close(result.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn test_softmax_backward_1d() {
        // Correct softmax gradient: dL/dx_j = s_j * (g_j - dot(g, s))
        let mut act = Activation::new(vec![3], ActivationFn::Softmax, 0.0);
        let vals = [1.0, 2.0, 3.0];
        let grads = [1.0, 0.0, 0.0];
        let a_in = tensor_with_data(vec![3], &vals);
        let grad = tensor_with_data(vec![3], &grads);
        let result = act.backward(1.0, a_in, grad);

        let s = softmax(&vals);
        let dot: f64 = grads.iter().zip(s.iter()).map(|(g, si)| g * si).sum();
        let expected: Vec<f64> = s
            .iter()
            .zip(grads.iter())
            .map(|(si, gi)| si * (gi - dot))
            .collect();
        assert_close(result.data.as_slice(), &expected, 1e-10);
    }

    #[test]
    fn test_softmax_backward_gradient_sums_to_zero() {
        // The softmax backward gradient should always sum to zero
        let mut act = Activation::new(vec![4], ActivationFn::Softmax, 0.0);
        let a_in = tensor_with_data(vec![4], &[0.5, -0.5, 1.0, -1.0]);
        let grad = tensor_with_data(vec![4], &[1.0, -0.5, 0.3, 0.2]);
        let result = act.backward(1.0, a_in, grad);
        let total: f64 = result.data.as_slice().iter().sum();
        assert!(
            total.abs() < 1e-10,
            "softmax gradient should sum to 0, got {}",
            total
        );
    }

    #[test]
    fn test_softmax_backward_2d() {
        // Shape [3, 2]: 3 classes, 2 columns
        let mut act = Activation::new(vec![3, 2], ActivationFn::Softmax, 0.0);
        let data = [1.0, 0.0, 2.0, 0.0, 0.0, 3.0];
        let grad_data = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let a_in = tensor_with_data(vec![3, 2], &data);
        let grad = tensor_with_data(vec![3, 2], &grad_data);
        let result = act.backward(1.0, a_in, grad);

        // Column 0: values [1,2,0], grad [1,0,0]
        let col0_s = softmax(&[1.0, 2.0, 0.0]);
        let col0_g = [1.0, 0.0, 0.0];
        let col0_dot: f64 = col0_g.iter().zip(col0_s.iter()).map(|(g, s)| g * s).sum();
        let col0_expected: Vec<f64> = col0_s
            .iter()
            .zip(col0_g.iter())
            .map(|(s, g)| s * (g - col0_dot))
            .collect();

        // Column 1: values [0,0,3], grad [0,1,0]
        let col1_s = softmax(&[0.0, 0.0, 3.0]);
        let col1_g = [0.0, 1.0, 0.0];
        let col1_dot: f64 = col1_g.iter().zip(col1_s.iter()).map(|(g, s)| g * s).sum();
        let col1_expected: Vec<f64> = col1_s
            .iter()
            .zip(col1_g.iter())
            .map(|(s, g)| s * (g - col1_dot))
            .collect();

        let expected = [
            col0_expected[0],
            col1_expected[0],
            col0_expected[1],
            col1_expected[1],
            col0_expected[2],
            col1_expected[2],
        ];
        assert_close(result.data.as_slice(), &expected, 1e-10);
    }

    // ---- Numerical gradient checks (finite differences) ----

    fn numerical_gradient_check_elementwise(func: ActivationFn, values: &[f64]) {
        let eps = 1e-7;
        let n = values.len();
        let mut act = Activation::new(vec![n], func, 0.0);

        let a_in = tensor_with_data(vec![n], values);
        let grad = tensor_with_data(vec![n], &vec![1.0; n]);
        let analytical = act.backward(1.0, a_in, grad);

        let mut numerical = vec![0.0; n];
        for i in 0..n {
            let mut plus = values.to_vec();
            let mut minus = values.to_vec();
            plus[i] += eps;
            minus[i] -= eps;
            let out_plus = act.forward(tensor_with_data(vec![n], &plus));
            let out_minus = act.forward(tensor_with_data(vec![n], &minus));
            numerical[i] = (out_plus.data[i] - out_minus.data[i]) / (2.0 * eps);
        }
        assert_close(analytical.data.as_slice(), &numerical, 1e-5);
    }

    #[test]
    fn test_numerical_grad_sigmoid() {
        numerical_gradient_check_elementwise(ActivationFn::Sigmoid, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_numerical_grad_relu() {
        // Avoid x=0 where ReLU is non-differentiable
        numerical_gradient_check_elementwise(ActivationFn::ReLU, &[-1.0, -0.5, 0.5, 1.0, 2.0]);
    }

    #[test]
    fn test_numerical_grad_silu() {
        numerical_gradient_check_elementwise(ActivationFn::SiLU, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_numerical_grad_tanh() {
        numerical_gradient_check_elementwise(ActivationFn::Tanh, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_numerical_grad_softmax() {
        let n = 3;
        let eps = 1e-7;
        let values = [1.0, 2.0, 3.0];
        let upstream = [1.0, 0.5, -0.5];

        let mut act = Activation::new(vec![n], ActivationFn::Softmax, 0.0);
        let a_in = tensor_with_data(vec![n], &values);
        let grad = tensor_with_data(vec![n], &upstream);
        let analytical = act.backward(1.0, a_in, grad);

        // For softmax, output[j] depends on all inputs, so we need full Jacobian
        let mut numerical = vec![0.0; n];
        for i in 0..n {
            let mut plus = values.to_vec();
            let mut minus = values.to_vec();
            plus[i] += eps;
            minus[i] -= eps;
            let out_plus = act.forward(tensor_with_data(vec![n], &plus));
            let out_minus = act.forward(tensor_with_data(vec![n], &minus));
            // dL/dx_i = sum_j upstream_j * d_output_j / d_input_i
            let mut grad_i = 0.0;
            for j in 0..n {
                grad_i += upstream[j] * (out_plus.data[j] - out_minus.data[j]) / (2.0 * eps);
            }
            numerical[i] = grad_i;
        }
        assert_close(analytical.data.as_slice(), &numerical, 1e-5);
    }

    // ---- Dropout tests ----

    #[test]
    fn test_prepare_no_dropout_no_mask() {
        let mut act = Activation::new(vec![3], ActivationFn::ReLU, 0.0);
        act.prepare(true, 1);
        assert!(act.mask.is_none());
    }

    #[test]
    fn test_prepare_inference_no_mask() {
        let mut act = Activation::new(vec![3], ActivationFn::ReLU, 0.5);
        act.prepare(false, 1);
        assert!(act.mask.is_none());
    }

    #[test]
    fn test_prepare_creates_binary_mask() {
        let mut act = Activation::new(vec![4], ActivationFn::ReLU, 0.5);
        act.prepare(true, 1);
        assert!(act.mask.is_some());
        let mask = act.mask.as_ref().unwrap();
        assert_eq!(mask.data.len(), 4);
        for &v in mask.data.as_slice() {
            assert!(
                v == 0.0 || v == 1.0,
                "mask values must be 0 or 1, got {}",
                v
            );
        }
    }

    #[test]
    fn test_prepare_mask_is_class_level() {
        // Dropout should zero entire classes (rows), not individual elements
        let mut act = Activation::new(vec![4, 3], ActivationFn::ReLU, 0.5);
        act.prepare(true, 1);
        let mask = act.mask.as_ref().unwrap();
        let classes = mask.shape[0];
        let cols = mask.data.len() / classes;
        for c in 0..classes {
            let first = mask.data[c * cols];
            for j in 1..cols {
                assert_eq!(
                    mask.data[c * cols + j],
                    first,
                    "class {} mask should be uniform, but index {} differs",
                    c,
                    j
                );
            }
        }
    }

    #[test]
    fn test_dropout_zeros_output() {
        // High dropout should eventually produce zeros
        let mut act = Activation::new(vec![4], ActivationFn::ReLU, 0.99);
        let mut saw_zero = false;
        for _ in 0..100 {
            act.prepare(true, 1);
            let input = tensor_with_data(vec![4], &[1.0, 2.0, 3.0, 4.0]);
            let output = act.forward(input);
            if output.data.as_slice().iter().any(|&x| x == 0.0) {
                saw_zero = true;
                break;
            }
        }
        assert!(saw_zero, "dropout=0.99 should produce zeros in output");
    }

    #[test]
    fn test_dropout_backward_zeros_masked_gradients() {
        // Manually set mask to test deterministic dropout backward behavior.
        // With a known mask, masked classes should have zero gradient.
        let mut act = Activation::new(vec![4], ActivationFn::ReLU, 0.5);
        // Manually inject a known mask: classes 0,2 active, classes 1,3 dropped
        act.mask = Some(tensor_with_data(vec![4], &[1.0, 0.0, 1.0, 0.0]));
        let a_in = tensor_with_data(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let grad = tensor_with_data(vec![4], &[1.0, 1.0, 1.0, 1.0]);
        let result = act.backward(1.0, a_in, grad);
        // Classes 1,3 are masked -> gradient should be 0
        assert_eq!(result.data[1], 0.0);
        assert_eq!(result.data[3], 0.0);
        // Classes 0,2 are active -> normal ReLU gradient (input > 0 so grad passes through)
        assert_eq!(result.data[0], 1.0);
        assert_eq!(result.data[2], 1.0);
    }

    #[test]
    fn test_dropout_forward_applies_mask() {
        let mut act = Activation::new(vec![4], ActivationFn::Sigmoid, 0.5);
        // Manually inject mask: classes 0,1 active, classes 2,3 dropped
        act.mask = Some(tensor_with_data(vec![4], &[1.0, 1.0, 0.0, 0.0]));
        let vals = [0.0, 1.0, 2.0, -1.0];
        let input = tensor_with_data(vec![4], &vals);
        let output = act.forward(input);
        assert_close(
            output.data.as_slice(),
            &[sigmoid(0.0), sigmoid(1.0), 0.0, 0.0],
            1e-12,
        );
    }

    // ---- Serialization tests ----

    #[test]
    fn test_serialization_roundtrip() {
        let act = Activation::new(vec![3, 4], ActivationFn::SiLU, 0.25);
        let mut buf = Vec::new();
        act.write(&mut buf).unwrap();
        let mut cursor = std::io::Cursor::new(buf);
        let restored = Activation::read(&mut cursor).unwrap();
        assert_eq!(restored.input_shape, vec![3, 4]);
        assert_eq!(restored.dropout, 0.25);
        assert_eq!(restored.function.id(), act.function.id());
    }

    #[test]
    fn test_serialization_all_variants() {
        let fns = [
            ActivationFn::Sigmoid,
            ActivationFn::ReLU,
            ActivationFn::SiLU,
            ActivationFn::Softmax,
            ActivationFn::Tanh,
        ];
        for func in fns {
            let act = Activation::new(vec![2], func, 0.0);
            let mut buf = Vec::new();
            act.write(&mut buf).unwrap();
            let mut cursor = std::io::Cursor::new(buf);
            let restored = Activation::read(&mut cursor).unwrap();
            assert_eq!(restored.function.id(), func.id());
        }
    }

    #[test]
    fn test_activation_fn_id_roundtrip() {
        let fns = [
            ActivationFn::Sigmoid,
            ActivationFn::ReLU,
            ActivationFn::SiLU,
            ActivationFn::Softmax,
            ActivationFn::Tanh,
        ];
        for func in fns {
            let id = func.id();
            let restored = ActivationFn::from_id(id);
            assert!(restored.is_some(), "from_id failed for {:?}", id);
            assert_eq!(restored.unwrap().id(), id);
        }
    }

    #[test]
    fn test_activation_fn_from_invalid_id() {
        assert!(ActivationFn::from_id(b"BadIdent").is_none());
    }
}
