use std::simd::{Simd, num::SimdFloat};

use crate::{
    ml::activation::Activation,
    tensor::cpu2::{self, Tensor, Fill, FillUninit, Generate},
};
use rand_distr::{Distribution, Normal};
use rayon::{iter::{IndexedParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

const LANES: usize = 8; // number of SIMD lanes
const FORWARD_MIN_FLOPS_PER_CORE: usize = 512_000 / 24; // measured on bench suite. adjusted to apply to any core count
// Backward pass thresholds derived from benching/benchmarks.csv (multithread vs initial).
// Backward total crossover ~1.18e7 FLOPs; split across two passes ~= 5.9e6 FLOPs.
const BACKWARD_PASS1_MIN_FLOPS_PER_CORE: usize = 6_000_000 / 24;
const BACKWARD_PASS2_MIN_FLOPS_PER_CORE: usize = 6_000_000 / 24;

pub struct FeedForward<'a> {
    weights: Tensor<'a>,
    biases: Tensor<'a>,
    activation: Activation,
    // avoid recomputing these every forward/backward pass
    fan_in: usize,
    // might be better to keep on stack rather than reading weights.shape[0] over and over
    // todo: evaluate if this is better and maybe try to move other things on-stack
    neurons: usize,
    batch_idx: usize
}

impl <'a> FeedForward<'a> {
    pub fn new(input_shape: Vec<usize>, neurons: usize, activation: Activation) -> Self {
        let fan_in = cpu2::length_of(&input_shape);
        let xavier = Normal::new(0.0, 2.0 / (fan_in + neurons) as f64).unwrap();
        Self {
            weights: Tensor::init(Generate {
                shape: vec![neurons, fan_in],
                with: || xavier.sample(&mut rand::rng()),
            })
            .unwrap(),
            biases: Tensor::init(Fill {
                shape: vec![neurons],
                with: 0.0,
            })
            .unwrap(),
            activation,
            fan_in,
            neurons,
            batch_idx: input_shape.len()
        }
    }

    pub fn forward<'i, 'o>(&self, a_in: Tensor<'i>) -> Tensor<'o> {
        // forward pass: Wx + b -> [neurons, fan_in] x [fan_in, batch] -> [neurons, batch] + [neurons]
        let batch = *a_in.shape.get(self.batch_idx).unwrap_or(&1);
        // dbg!(self.fan_in);
        let a_in_flat_t = a_in
            .reshape(&[self.fan_in, batch])
            .unwrap()
            .transpose(&[1, 0])
            .unwrap()
            .materialize();
        let mut a_out_flat =
            Tensor::init(unsafe { FillUninit::new(vec![self.neurons, batch]) }).unwrap();
        // explicit slice definitions to signal to the compiler about aliasing
        let mut out_data = a_out_flat.data.to_mut().as_mut_slice();
        let parallelism = rayon::current_num_threads();
        let flops_per_core = (self.neurons * self.fan_in * batch) / parallelism;
        if flops_per_core >= FORWARD_MIN_FLOPS_PER_CORE {
            let neurons_per_chunk = (self.neurons as f64 / parallelism as f64).ceil() as usize;
            let grain = neurons_per_chunk * batch;
            out_data.par_chunks_mut(grain).enumerate().for_each(|(i, out)| {
                let start_neuron = neurons_per_chunk * i;
                let neuron_count = out.len() / batch;
                self.forward_mm(start_neuron, neuron_count, batch, &a_in_flat_t, out);
            });
        } else {
            self.forward_mm(0, self.neurons, batch, &a_in_flat_t, &mut out_data);
        }

        self.activation.forward_all(a_out_flat)
    }

    #[inline(always)]
    fn forward_mm(&self, start_neuron: usize, neuron_count: usize, batch: usize, a_in_flat_t: &Tensor<'_>, out: &mut [f64]) {
        let mut lhs_idx = self.fan_in * start_neuron;
        let mut rhs_idx = 0;
        let mut bias_idx = start_neuron;
        let mut out_idx = 0;
        let chunks = self.fan_in / LANES;
        let remainder = self.fan_in % LANES;
        for _ in 0..neuron_count {
            let bias = unsafe { *self.biases.data.get_unchecked(bias_idx) };
            for _ in 0..batch {
                let mut lhs_ptr = unsafe { self.weights.data.as_ptr().add(lhs_idx) };
                let mut rhs_ptr = unsafe { a_in_flat_t.data.as_ptr().add(rhs_idx) };
                let mut sum_simd: Simd<f64, LANES> = Simd::splat(0.0);
                for _ in 0..chunks {
                    let lhs_simd =
                        unsafe { std::ptr::read_unaligned(lhs_ptr as *const Simd<f64, LANES>) };
                    let rhs_simd =
                        unsafe { std::ptr::read_unaligned(rhs_ptr as *const Simd<f64, LANES>) };
                    sum_simd += lhs_simd * rhs_simd;
                    lhs_ptr = unsafe { lhs_ptr.add(LANES) };
                    rhs_ptr = unsafe { rhs_ptr.add(LANES) };
                }
                let mut sum: f64 = sum_simd.reduce_sum();
                lhs_idx += LANES * chunks;
                rhs_idx += LANES * chunks;
                for _ in 0..remainder {
                    let lhs = unsafe { *self.weights.data.get_unchecked(lhs_idx) };
                    let rhs = unsafe { *a_in_flat_t.data.get_unchecked(rhs_idx) };
                    sum += lhs * rhs;
                    lhs_idx += 1;
                    rhs_idx += 1; 
                }
                unsafe {
                    *out.get_unchecked_mut(out_idx) = sum + bias;
                }
                lhs_idx -= self.fan_in;
                out_idx += 1;
            }
            lhs_idx += self.fan_in;
            rhs_idx = 0;
            bias_idx += 1;
        }
    }

    pub fn backward<'i, 'o>(
        &mut self,
        c: f64,
        a_in: &Tensor<'i>,
        a_out: &Tensor<'i>,
        mut grad: Tensor<'i>,
    ) -> Tensor<'o> {
        // backwards activations
        grad = self.activation.backward_all(&a_out, grad);

        let batch = *a_in.shape.get(self.batch_idx).unwrap_or(&1);
        let a_in_flat = a_in
            .reshape(&[self.fan_in, batch])
            .unwrap();
        let parallelism = rayon::current_num_threads();
        let flops_per_core = (self.neurons * self.fan_in * batch) / parallelism;

        // pass 1: activations backwards (W^T dot grad) [fan_in, neurons] x [neurons, batch] -> [fan_in, batch]
        let mut a_grad = a_in_flat.cloned_view().materialize();
        // todo: fuse transpose + materialize of owned tensors since bounds will be fixed
        let weights_t = self.weights.transpose(&[1, 0]).unwrap().materialize();
        let grad_t = grad.transpose(&[1, 0]).unwrap().materialize();
        let out_data: &mut [f64] = a_grad.data.to_mut().as_mut_slice();

        if flops_per_core >= BACKWARD_PASS1_MIN_FLOPS_PER_CORE {
            let fan_in_per_chunk = (self.fan_in as f64 / parallelism as f64).ceil() as usize;
            let grain = fan_in_per_chunk * batch;
            out_data.par_chunks_mut(grain).enumerate().for_each(|(i, out)| {
                let fan_in_count = out.len() / batch;
                let start_fan_in = i * fan_in_per_chunk;
                Self::backwards_pass1_mm(start_fan_in, fan_in_count, batch, self.neurons, &weights_t.data, &grad_t.data, out);
            });
        } else {
            Self::backwards_pass1_mm(0, self.fan_in, batch, self.neurons, &weights_t.data, &grad_t.data, out_data);
        }
        // pass 2: weights backwards (grad dot a_in^T) [neurons, batch] dot [batch, fan_in] -> [neurons, fan_in] -> lhs_grad
        //         bias backwards (sum over batch of grad) -> bias_grad
        let a_in_flat_t = a_in_flat
            .transpose(&[1, 0])
            .unwrap()
            .materialize();
        let out_data = self.weights.data.to_mut().as_mut_slice();
        let bias_data = self.biases.data.to_mut().as_mut_slice();
        if flops_per_core >= BACKWARD_PASS2_MIN_FLOPS_PER_CORE {
            let neurons_per_chunk = (self.neurons as f64 / parallelism as f64).ceil() as usize;
            let neuron_grain = neurons_per_chunk * self.fan_in;
            let bias_grain = neurons_per_chunk;
            out_data.par_chunks_mut(neuron_grain).zip(bias_data.par_chunks_mut(bias_grain)).enumerate().for_each(|(i, (out, bias))| {
                let start_neuron = i * neurons_per_chunk;
                let neuron_count = out.len() / self.fan_in;
                Self::backwards_pass2_mm(start_neuron, neuron_count, batch, self.fan_in, c, &a_in_flat_t.data, &grad.data, bias, out);
            });
        } else {
            Self::backwards_pass2_mm(0, self.neurons, batch, self.fan_in, c, &a_in_flat_t.data, &grad.data, bias_data, out_data);
        }

        a_grad
    }

    fn backwards_pass1_mm(start_fan_in: usize, fan_in_count: usize, batch: usize, neurons: usize, weights_t: &[f64], grad_t: &[f64], out: &mut [f64]) {
        let mut lhs_idx = start_fan_in * neurons;
        let mut rhs_idx = 0;
        let mut out_idx = 0;
        let chunks = neurons / LANES;
        let remainder = neurons % LANES;
        for _ in 0..fan_in_count {
            for _ in 0..batch {
                let mut sum_simd: Simd<f64, LANES> = Simd::splat(0.0);
                // todo: cast ptrs once
                let mut lhs_ptr = unsafe { weights_t.as_ptr().add(lhs_idx) };
                let mut rhs_ptr = unsafe { grad_t.as_ptr().add(rhs_idx) };
                for _ in 0..chunks {
                    let lhs_simd =
                        unsafe { std::ptr::read_unaligned(lhs_ptr as *const Simd<f64, LANES>) };
                    let rhs_simd =
                        unsafe { std::ptr::read_unaligned(rhs_ptr as *const Simd<f64, LANES>) };
                    sum_simd += lhs_simd * rhs_simd;
                    lhs_ptr = unsafe { lhs_ptr.add(LANES) };
                    rhs_ptr = unsafe { rhs_ptr.add(LANES) };
                }
                let mut sum = sum_simd.reduce_sum();
                lhs_idx += LANES * chunks;
                rhs_idx += LANES * chunks;
                for _ in 0..remainder {
                    let lhs = unsafe { *weights_t.get_unchecked(lhs_idx) };
                    let rhs = unsafe { *grad_t.get_unchecked(rhs_idx) };
                    sum += lhs * rhs;
                    lhs_idx += 1;
                    rhs_idx += 1;
                }
                unsafe {
                    *out.get_unchecked_mut(out_idx) = sum;
                }
                lhs_idx -= neurons;
                out_idx += 1;                    
            }
            lhs_idx += neurons;
            rhs_idx = 0;
        }
    }

    #[inline(always)]
    fn backwards_pass2_mm(start_neuron: usize, neuron_count: usize, batch: usize, fan_in: usize, c: f64, a_in_flat_t: &[f64], grad: &[f64], bias: &mut [f64], out: &mut [f64]) {
        let c_simd = Simd::splat(c);
        let chunks = fan_in / LANES;
        let remainder = fan_in % LANES;
        let mut lhs_idx = start_neuron * batch;
        let mut rhs_idx = 0;
        let mut out_idx = 0;
        for n in 0..neuron_count {
            let bias = unsafe { bias.get_unchecked_mut(n) };
            let mut sum_g = 0.0;
            for _ in 0..batch {
                // access gradient (also accumulate for bias update)
                let g = unsafe { *grad.get_unchecked(lhs_idx) };
                sum_g += g;

                let g_simd = Simd::splat(g);
                let mut rhs_ptr = unsafe { a_in_flat_t.as_ptr().add(rhs_idx) };
                let mut out_ptr = unsafe { out.as_mut_ptr().add(out_idx) };
                for _ in 0..chunks {
                    let rhs_simd =
                        unsafe { std::ptr::read_unaligned(rhs_ptr as *const Simd<f64, LANES>) };
                    let out_simd =
                        unsafe { std::ptr::read_unaligned(out_ptr as *const Simd<f64, LANES>) };
                    let descended = out_simd - g_simd * rhs_simd * c_simd;
                    unsafe {
                        std::ptr::write_unaligned(out_ptr as *mut Simd<f64, LANES>, descended)
                    };
                    rhs_ptr = unsafe { rhs_ptr.add(LANES) };
                    out_ptr = unsafe { out_ptr.add(LANES) };
                }
                rhs_idx += LANES * chunks;
                out_idx += LANES * chunks;
                for _ in 0..remainder {
                    let rhs = unsafe { *a_in_flat_t.get_unchecked(rhs_idx) };
                    let out = unsafe { out.get_unchecked_mut(out_idx) };
                    *out -= c * g * rhs;
                    rhs_idx += 1;
                    out_idx += 1;
                }
                lhs_idx += 1;
                out_idx -= fan_in;
            }
            // backwards to bias too since we actually are already traversing over the correct axes while backpropagating weights 
            // (accumulated sum over batch to avoid unneeded writes / muls)
            *bias -= c * sum_g;
            
            rhs_idx = 0;
            out_idx += fan_in;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::cpu2;

    fn tensor_with_data(shape: Vec<usize>, values: &[f64]) -> Tensor<'static> {
        let mut tensor = Tensor::init(Fill { shape, with: 0.0 }).expect("tensor init");
        assert_eq!(tensor.data.len(), values.len());
        tensor.data.to_mut().clone_from_slice(values);
        tensor
    }

    fn ff_with_params(
        weights: &[f64],
        biases: &[f64],
        input_shape: Vec<usize>,
        activation: Activation,
    ) -> FeedForward<'static> {
        let neurons = biases.len();
        let fan_in = cpu2::length_of(&input_shape);
        assert_eq!(
            weights.len(),
            neurons * fan_in,
            "weights must be neurons x fan_in_input_shape"
        );

        let mut w = Tensor::init(Fill {
            shape: vec![neurons, fan_in],
            with: 0.0,
        })
        .unwrap();
        w.data.to_mut().clone_from_slice(weights);

        let mut b = Tensor::init(Fill {
            shape: vec![neurons],
            with: 0.0,
        })
        .unwrap();
        b.data.to_mut().clone_from_slice(biases);

        FeedForward {
            weights: w,
            biases: b,
            activation,
            neurons,
            fan_in: fan_in,
            batch_idx: input_shape.len()
        }
    }

    fn ff_zeroed(
        input_shape: Vec<usize>,
        neurons: usize,
        activation: Activation,
    ) -> FeedForward<'static> {
        let fan_in = cpu2::length_of(&input_shape);
        FeedForward {
            weights: Tensor::init(Fill {
                shape: vec![neurons, fan_in],
                with: 0.0,
            })
            .unwrap(),
            biases: Tensor::init(Fill {
                shape: vec![neurons],
                with: 0.0,
            })
            .unwrap(),
            activation,
            neurons,
            fan_in: fan_in,
            batch_idx: input_shape.len()
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: actual {} vs expected {}",
            actual.len(),
            expected.len()
        );
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "mismatch at index {}: expected {}, got {} (diff {})",
                idx,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    #[test]
    fn forward_matches_matmul_bias_then_activation() {
        let activation = Activation::Sigmoid;
        let ff = ff_with_params(&[0.5, -1.0, 1.5, 0.3], &[0.2, -0.1], vec![2], activation);
        let input = tensor_with_data(vec![2, 1], &[0.4, 0.8]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![2, 1]);
        let expected = vec![
            activation.forward(0.5 * 0.4 + -1.0 * 0.8 + 0.2),
            activation.forward(1.5 * 0.4 + 0.3 * 0.8 - 0.1),
        ];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_handles_explicit_batch_dim() {
        let activation = Activation::ReLU;
        let ff = ff_with_params(&[1.0, 2.0], &[0.0], vec![2], activation);
        // Shape [feature, batch]
        let input = tensor_with_data(vec![2, 3], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 3]);
        let expected = vec![21.0, 42.0, 63.0];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn backward_returns_input_grad_and_updates_params() {
        let activation = Activation::ReLU;
        let mut ff = ff_with_params(
            &[1.0, -1.0, 0.5, 0.5, 1.0, -0.5],
            &[3.0, 1.0],
            vec![3],
            activation,
        );
        let input = tensor_with_data(vec![3, 1], &[2.0, 3.0, 4.0]);
        let grad = tensor_with_data(vec![2, 1], &[0.2, -0.4]);

        let lr = 0.1;
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![3, 1]);
        let expected_input_grad = vec![0.0, -0.6, 0.3];
        assert_close(input_grad.data.as_ref(), &expected_input_grad, 1e-12);

        let expected_biases = vec![2.98, 1.04];
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);

        let expected_weights = vec![0.96, -1.06, 0.42, 0.58, 1.12, -0.34];
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
    }

    #[test]
    fn backward_small_batch_gates_inactive_neurons() {
        let activation = Activation::ReLU;
        let mut ff = ff_with_params(&[1.0, 2.0, -1.0, 1.0], &[0.0, -3.0], vec![2], activation);
        let input = tensor_with_data(vec![2, 1], &[1.0, 2.0]);
        let grad = tensor_with_data(vec![2, 1], &[0.5, 1.0]);

        let lr = 0.1;
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![2, 1]);
        let expected_weights = vec![0.95, 1.9, -1.0, 1.0];
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
        let expected_biases = vec![-0.05, -3.0];
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);
        let expected_input_grad = vec![0.5, 1.0];
        assert_close(input_grad.data.as_ref(), &expected_input_grad, 1e-12);
    }

    #[test]
    fn backward_large_batch_accumulates_gradients() {
        let activation = Activation::ReLU;
        let batch = 64;
        let mut ff = ff_with_params(&[1.0, 2.0, 0.5, 1.5], &[0.1, -0.2], vec![2], activation);

        let mut input = Tensor::init(Fill {
            shape: vec![2, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.to_mut();
            for b in 0..batch {
                data[0 * batch + b] = 1.0;
                data[1 * batch + b] = 2.0;
            }
        }

        let mut grad = Tensor::init(Fill {
            shape: vec![2, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.to_mut();
            for b in 0..batch {
                data[0 * batch + b] = 0.3;
                data[1 * batch + b] = -0.7;
            }
        }

        let lr = 0.01;
        let activations_out = ff.forward(input.cloned_view());
        let a_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(a_grad.shape, vec![2, batch]);
        let expected_weights = vec![0.808, 1.616, 0.948, 2.396];
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
        let expected_biases = vec![-0.092, 0.248];
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);

        for b in 0..batch {
            let idx0 = 0 * batch + b;
            let idx1 = 1 * batch + b;
            assert!(
                (a_grad.data[idx0] + 0.05).abs() <= 1e-12,
                "expected dL/dx0=-0.05 at batch {}, got {}",
                b,
                a_grad.data[idx0]
            );
            assert!(
                (a_grad.data[idx1] + 0.45).abs() <= 1e-12,
                "expected dL/dx1=-0.45 at batch {}, got {}",
                b,
                a_grad.data[idx1]
            );
        }
    }

    #[test]
    fn forward_large_dims_matches_expected_rows() {
        let activation = Activation::ReLU;
        let neurons = 8_096;
        let features = 512;
        let batch = 16;

        let mut ff = ff_zeroed(vec![features], neurons, activation);
        {
            let w = ff.weights.data.to_mut();
            w[0 * features + 0] = 2.0;
            w[1 * features + (features - 1)] = -1.0;
        }
        {
            let b = ff.biases.data.to_mut();
            b[0] = 0.5;
            b[1] = -0.5;
        }

        let mut input = Tensor::init(Fill {
            shape: vec![features, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.to_mut();
            for b in 0..batch {
                data[0 * batch + b] = 1.0; // feature 0
                data[(features - 1) * batch + b] = 2.0; // feature 511
            }
        }

        let out = ff.forward(input);
        assert_eq!(out.shape, vec![neurons, batch]);
        let out_data = out.data.as_ref();
        for b in 0..batch {
            let idx0 = 0 * batch + b;
            let idx1 = 1 * batch + b;
            let idx_last = (neurons - 1) * batch + b;
            assert!(
                (out_data[idx0] - 2.5).abs() <= 1e-12,
                "neuron 0 batch {} expected 2.5, got {}",
                b,
                out_data[idx0]
            );
            assert!(
                out_data[idx1].abs() <= 1e-12,
                "neuron 1 batch {} expected 0.0, got {}",
                b,
                out_data[idx1]
            );
            assert!(
                out_data[idx_last].abs() <= 1e-12,
                "neuron 8095 batch {} expected 0.0, got {}",
                b,
                out_data[idx_last]
            );
        }
    }

    #[test]
    fn backward_large_dims_updates_expected_weights_and_biases() {
        let activation = Activation::ReLU;
        let neurons = 8_096;
        let features = 512;
        let batch = 16;
        let mut ff = ff_zeroed(vec![features], neurons, activation);
        // Configure a few weights/biases so ReLU gates some channels off.
        {
            let w = ff.weights.data.to_mut();
            w[0 * features + 0] = 1.0; // neuron 0, feature 0 -> positive pre-activation
            w[1 * features + (features - 1)] = -1.0; // neuron 1, feature 511 -> negative pre-activation
            w[(neurons - 1) * features + 0] = 0.5; // last neuron, feature 0 -> positive pre-activation
        }

        // Input: feature 0 is 1.0, feature 511 is 2.0 for every batch.
        let mut input = Tensor::init(Fill {
            shape: vec![features, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.to_mut();
            for b in 0..batch {
                data[0 * batch + b] = 1.0;
                data[(features - 1) * batch + b] = 2.0;
            }
        }

        // Gradients: neuron 0 -> +1.0, neuron 1 -> -1.0, last neuron -> +0.5 for every batch.
        let mut grad = Tensor::init(Fill {
            shape: vec![neurons, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.to_mut();
            for b in 0..batch {
                data[0 * batch + b] = 1.0;
                data[1 * batch + b] = -1.0; // will be gated to zero by ReLU (negative pre-act)
                data[(neurons - 1) * batch + b] = 0.5;
            }
        }

        let lr = 0.1;
        let activations_out = ff.forward(input.cloned_view());
        let a_grad = ff.backward(lr, &input, &activations_out, grad);

        // Input gradient: only neurons with positive pre-activation (0 and last) contribute.
        assert_eq!(a_grad.shape, vec![features, batch]);
        for b in 0..batch {
            let idx0 = 0 * batch + b;
            let idx511 = (features - 1) * batch + b;
            assert!(
                (a_grad.data[idx0] - 1.25).abs() <= 1e-12,
                "expected dL/dx0=1.25 at batch {}, got {}",
                b,
                a_grad.data[idx0]
            );
            assert!(
                a_grad.data[idx511].abs() <= 1e-12,
                "expected dL/dx511=0 at batch {}, got {}",
                b,
                a_grad.data[idx511]
            );
        }

        // Bias updates: b0 -= lr * 16, b1 gated to zero, b_last -= lr * (0.5 * 16)
        let biases = ff.biases.data.as_ref();
        assert_close(&biases[0..3], &[-1.6, 0.0, 0.0], 1e-12);
        assert!(
            (biases[neurons - 1] + 0.8).abs() <= 1e-12,
            "expected last bias to be -0.8, got {}",
            biases[neurons - 1]
        );

        // Weight updates on selected entries:
        // w(0,0) starts at 1.0 and -= lr * sum(grad0 * x0) = 1.0 - 0.1 * 16 = -0.6
        // w(1,511) starts at -1.0 and stays unchanged (gated)
        // w(last,0) starts at 0.5 and -= lr * sum(grad_last * x0) = 0.5 - 0.1 * (0.5 * 16) = -0.3
        let weights = ff.weights.data.as_ref();
        assert!(
            (weights[0 * features + 0] + 0.6).abs() <= 1e-12,
            "expected w(0,0)=-0.6, got {}",
            weights[0 * features + 0]
        );
        assert!(
            (weights[1 * features + (features - 1)] + 1.0).abs() <= 1e-12,
            "expected w(1,511)=-1.0 (unchanged), got {}",
            weights[1 * features + (features - 1)]
        );
        assert!(
            (weights[(neurons - 1) * features + 0] + 0.3).abs() <= 1e-12,
            "expected w(last,0)=-0.3, got {}",
            weights[(neurons - 1) * features + 0]
        );
    }

    #[test]
    fn forward_non_multiple_of_lanes_dimensions_matches_reference() {
        let activation = Activation::ReLU;
        let features = 10; // 8-lane SIMD + 2-scalar remainder
        let neurons = 9; // 8-lane SIMD + 1-scalar remainder
        let batch = 3;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let biases: Vec<f64> = (0..neurons).map(|n| n as f64 * 0.05).collect();
        let ff = ff_with_params(&weights, &biases, vec![features], activation);

        let mut input = Tensor::init(Fill {
            shape: vec![features, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.to_mut();
            for f in 0..features {
                for b in 0..batch {
                    data[f * batch + b] = 1.0 + f as f64 * 0.2 + b as f64 * 0.1;
                }
            }
        }

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![neurons, batch]);
        let mut expected = vec![0.0; neurons * batch];
        for n in 0..neurons {
            for b in 0..batch {
                let mut acc = biases[n];
                for f in 0..features {
                    acc += weights[n * features + f] * (1.0 + f as f64 * 0.2 + b as f64 * 0.1);
                }
                expected[n * batch + b] = activation.forward(acc);
            }
        }
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn backward_non_multiple_of_lanes_dimensions_matches_reference() {
        let activation = Activation::ReLU;
        let features = 10; // exercises remainder path on fan_in input dimension
        let neurons = 9; // exercises remainder path on neuron dimension
        let batch = 2;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.05 * (i as f64 + 1.0))
            .collect();
        let biases: Vec<f64> = (0..neurons).map(|n| 0.1 + n as f64 * 0.02).collect();
        let mut ff = ff_with_params(&weights, &biases, vec![features], activation);

        let mut input = Tensor::init(Fill {
            shape: vec![features, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.to_mut();
            for f in 0..features {
                for b in 0..batch {
                    data[f * batch + b] = 1.0 + f as f64 * 0.3 + b as f64 * 0.05;
                }
            }
        }
        let mut grad = Tensor::init(Fill {
            shape: vec![neurons, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.to_mut();
            for n in 0..neurons {
                for b in 0..batch {
                    data[n * batch + b] = 0.1 + n as f64 * 0.02 + b as f64 * 0.01;
                }
            }
        }

        let lr = 0.01;
        let input_data = input.data.clone().into_owned();
        let grad_data = grad.data.clone().into_owned();
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![features, batch]);

        let mut expected_input_grad = vec![0.0; features * batch];
        for f in 0..features {
            for b in 0..batch {
                let mut sum = 0.0;
                for n in 0..neurons {
                    sum += weights[n * features + f] * grad_data[n * batch + b];
                }
                expected_input_grad[f * batch + b] = sum;
            }
        }

        let mut expected_biases = biases.clone();
        for n in 0..neurons {
            let mut delta = 0.0;
            for b in 0..batch {
                delta += grad_data[n * batch + b];
            }
            expected_biases[n] -= lr * delta;
        }

        let mut expected_weights = weights.clone();
        for n in 0..neurons {
            for f in 0..features {
                let mut delta = 0.0;
                for b in 0..batch {
                    delta += grad_data[n * batch + b] * input_data[f * batch + b];
                }
                expected_weights[n * features + f] -= lr * delta;
            }
        }

        assert_close(input_grad.data.as_ref(), &expected_input_grad, 1e-12);
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
    }

    #[test]
    fn forward_with_tanh_activation() {
        let activation = Activation::Tanh;
        let ff = ff_with_params(&[1.0, 0.5, -0.5, 1.0], &[0.1, -0.1], vec![2], activation);
        let input = tensor_with_data(vec![2, 1], &[0.5, 0.8]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![2, 1]);
        let expected = vec![
            activation.forward(1.0 * 0.5 + 0.5 * 0.8 + 0.1),
            activation.forward(-0.5 * 0.5 + 1.0 * 0.8 - 0.1),
        ];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_with_silu_activation() {
        let activation = Activation::SiLU;
        let ff = ff_with_params(&[2.0, -1.0, 0.5, 1.5], &[0.3, -0.2], vec![2], activation);
        let input = tensor_with_data(vec![2, 1], &[1.0, 2.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![2, 1]);
        // SiLU(x) = x * sigmoid(x)
        let pre0 = 2.0 * 1.0 + -1.0 * 2.0 + 0.3;  // 0.3
        let pre1 = 0.5 * 1.0 + 1.5 * 2.0 - 0.2;   // 3.3
        let expected = vec![
            activation.forward(pre0),
            activation.forward(pre1),
        ];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_dims_smaller_than_lanes() {
        // All dimensions smaller than SIMD lane width (8)
        let activation = Activation::ReLU;
        let features = 3;
        let neurons = 5;
        let batch = 2;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let biases: Vec<f64> = (0..neurons).map(|n| n as f64 * 0.05).collect();
        let ff = ff_with_params(&weights, &biases, vec![features], activation);

        let input = tensor_with_data(
            vec![features, batch],
            &[1.0, 2.0, 0.5, 1.5, -0.5, 0.5],
        );

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![neurons, batch]);
        // Compute expected values manually
        let mut expected = vec![0.0; neurons * batch];
        for n in 0..neurons {
            for b in 0..batch {
                let mut acc = biases[n];
                for f in 0..features {
                    acc += weights[n * features + f] * [1.0, 2.0, 0.5, 1.5, -0.5, 0.5][f * batch + b];
                }
                expected[n * batch + b] = activation.forward(acc);
            }
        }
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_single_neuron() {
        let activation = Activation::Sigmoid;
        let ff = ff_with_params(&[1.0, 2.0, 3.0], &[0.5], vec![3], activation);
        let input = tensor_with_data(vec![3, 1], &[0.1, 0.2, 0.3]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 1]);
        let expected = vec![activation.forward(1.0 * 0.1 + 2.0 * 0.2 + 3.0 * 0.3 + 0.5)];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_single_feature() {
        let activation = Activation::ReLU;
        let ff = ff_with_params(&[2.0, -1.0, 0.5], &[0.1, -0.5, 0.0], vec![1], activation);
        let input = tensor_with_data(vec![1, 1], &[3.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![3, 1]);
        let expected = vec![
            activation.forward(2.0 * 3.0 + 0.1),   // 6.1
            activation.forward(-1.0 * 3.0 - 0.5), // 0.0 (ReLU gates negative)
            activation.forward(0.5 * 3.0 + 0.0),  // 1.5
        ];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_single_batch() {
        let activation = Activation::ReLU;
        let ff = ff_with_params(&[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0], vec![2], activation);
        let input = tensor_with_data(vec![2, 1], &[1.0, 1.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![2, 1]);
        let expected = vec![3.0, 7.0];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_large_batch() {
        // Test with larger batch size to exercise parallel code paths
        let activation = Activation::ReLU;
        let features = 4;
        let batch = 32;
        let neurons = 2;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let biases = vec![0.0; neurons];
        let ff = ff_with_params(&weights, &biases, vec![features], activation);

        let mut input = Tensor::init(Fill {
            shape: vec![features, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.to_mut();
            for i in 0..data.len() {
                data[i] = 0.1 * (i as f64 + 1.0);
            }
        }

        let out = ff.forward(input.cloned_view());

        assert_eq!(out.shape, vec![neurons, batch]);

        // Verify values
        let input_data = input.data.as_ref();
        for n in 0..neurons {
            for b in 0..batch {
                let mut acc = biases[n];
                for f in 0..features {
                    acc += weights[n * features + f] * input_data[f * batch + b];
                }
                let expected = activation.forward(acc);
                let actual = out.data[n * batch + b];
                assert!(
                    (actual - expected).abs() <= 1e-12,
                    "mismatch at neuron {} batch {}: expected {}, got {}",
                    n, b, expected, actual
                );
            }
        }
    }

    #[test]
    fn backward_with_zero_gradients() {
        let activation = Activation::ReLU;
        let mut ff = ff_with_params(&[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2], vec![2], activation);
        let original_weights = ff.weights.data.clone().into_owned();
        let original_biases = ff.biases.data.clone().into_owned();

        let input = tensor_with_data(vec![2, 1], &[1.0, 1.0]);
        let grad = tensor_with_data(vec![2, 1], &[0.0, 0.0]);

        let lr = 0.1;
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![2, 1]);
        // With zero gradients, weights and biases should remain unchanged
        assert_close(ff.weights.data.as_ref(), &original_weights, 1e-12);
        assert_close(ff.biases.data.as_ref(), &original_biases, 1e-12);
        // Input gradient should also be zero
        assert_close(input_grad.data.as_ref(), &[0.0, 0.0], 1e-12);
    }

    #[test]
    fn backward_with_zero_input() {
        // Use ReLU with positive biases so pre-activations are positive (derivative = 1)
        let activation = Activation::ReLU;
        let mut ff = ff_with_params(&[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2], vec![2], activation);
        let original_weights = ff.weights.data.clone().into_owned();

        let input = tensor_with_data(vec![2, 1], &[0.0, 0.0]);
        let grad = tensor_with_data(vec![2, 1], &[1.0, 1.0]);

        let lr = 0.1;
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![2, 1]);
        // With zero input, weights should remain unchanged (gradient * input = gradient * 0 = 0)
        assert_close(ff.weights.data.as_ref(), &original_weights, 1e-12);

        // Biases: b -= lr * grad (ReLU derivative = 1 for positive pre-activations)
        let expected_biases = vec![0.1 - 0.1 * 1.0, 0.2 - 0.1 * 1.0];
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);

        // Input gradient = W^T * grad
        let expected_input_grad = vec![1.0 + 3.0, 2.0 + 4.0]; // [4.0, 6.0]
        assert_close(input_grad.data.as_ref(), &expected_input_grad, 1e-12);
    }

    #[test]
    fn backward_dims_smaller_than_lanes() {
        // Use ReLU with positive weights/inputs/biases so all pre-activations are positive
        // This makes ReLU act like identity (derivative = 1)
        let activation = Activation::ReLU;
        let features = 3;
        let neurons = 5;
        let batch = 2;

        // All positive weights
        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        // All positive biases
        let biases: Vec<f64> = (0..neurons).map(|n| 1.0 + n as f64 * 0.05).collect();
        let mut ff = ff_with_params(&weights, &biases, vec![features], activation);

        // All positive inputs ensure positive pre-activations with positive weights/biases
        let input = tensor_with_data(
            vec![features, batch],
            &[1.0, 2.0, 0.5, 1.5, 0.5, 0.5],
        );
        let grad = tensor_with_data(
            vec![neurons, batch],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        );

        let lr = 0.01;
        let input_data = input.data.clone().into_owned();
        let grad_data = grad.data.clone().into_owned();
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![features, batch]);
        // With all positive pre-activations, ReLU derivative = 1, so grad passes through unchanged
        let mut expected_input_grad = vec![0.0; features * batch];
        for f in 0..features {
            for b in 0..batch {
                for n in 0..neurons {
                    expected_input_grad[f * batch + b] +=
                        weights[n * features + f] * grad_data[n * batch + b];
                }
            }
        }

        let mut expected_biases = biases.clone();
        for n in 0..neurons {
            for b in 0..batch {
                expected_biases[n] -= lr * grad_data[n * batch + b];
            }
        }

        let mut expected_weights = weights.clone();
        for n in 0..neurons {
            for f in 0..features {
                for b in 0..batch {
                    expected_weights[n * features + f] -=
                        lr * grad_data[n * batch + b] * input_data[f * batch + b];
                }
            }
        }

        assert_close(input_grad.data.as_ref(), &expected_input_grad, 1e-12);
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
    }

    #[test]
    fn backward_single_neuron_single_feature() {
        // ReLU with positive pre-activation (2.0*3.0 + 0.5 = 6.5 > 0) acts like identity
        let activation = Activation::ReLU;
        let mut ff = ff_with_params(&[2.0], &[0.5], vec![1], activation);

        let input = tensor_with_data(vec![1, 1], &[3.0]);
        let grad = tensor_with_data(vec![1, 1], &[0.5]);

        let lr = 0.1;
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![1, 1]);
        // input_grad = W^T * grad = 2.0 * 0.5 = 1.0
        assert_close(input_grad.data.as_ref(), &[1.0], 1e-12);
        // bias -= lr * grad = 0.5 - 0.1 * 0.5 = 0.45
        assert_close(ff.biases.data.as_ref(), &[0.45], 1e-12);
        // weight -= lr * grad * input = 2.0 - 0.1 * 0.5 * 3.0 = 1.85
        assert_close(ff.weights.data.as_ref(), &[1.85], 1e-12);
    }

    #[test]
    fn backward_large_batch() {
        // Test backward with larger batch size to exercise different code paths
        let activation = Activation::ReLU;
        let features = 4;
        let batch = 32;
        let neurons = 2;

        // Positive weights
        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        // Positive biases ensure positive pre-activations
        let biases = vec![1.0, 1.0];
        let mut ff = ff_with_params(&weights, &biases, vec![features], activation);

        let mut input = Tensor::init(Fill {
            shape: vec![features, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.to_mut();
            // Positive inputs
            for i in 0..data.len() {
                data[i] = 0.05 * (i as f64 + 1.0);
            }
        }

        let mut grad = Tensor::init(Fill {
            shape: vec![neurons, batch],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.to_mut();
            for i in 0..data.len() {
                data[i] = 0.1 * (i as f64 + 1.0);
            }
        }

        let lr = 0.01;
        let input_data = input.data.clone().into_owned();
        let grad_data = grad.data.clone().into_owned();
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        // Output shape should match input shape
        assert_eq!(input_grad.shape, vec![features, batch]);

        // With positive pre-activations, ReLU derivative = 1
        let mut expected_input_grad = vec![0.0; features * batch];
        for f in 0..features {
            for b in 0..batch {
                for n in 0..neurons {
                    expected_input_grad[f * batch + b] +=
                        weights[n * features + f] * grad_data[n * batch + b];
                }
            }
        }
        assert_close(input_grad.data.as_ref(), &expected_input_grad, 1e-12);

        // Compute expected biases
        let mut expected_biases = biases.clone();
        for n in 0..neurons {
            for b in 0..batch {
                expected_biases[n] -= lr * grad_data[n * batch + b];
            }
        }
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);

        // Compute expected weights
        let mut expected_weights = weights.clone();
        for n in 0..neurons {
            for f in 0..features {
                for b in 0..batch {
                    expected_weights[n * features + f] -=
                        lr * grad_data[n * batch + b] * input_data[f * batch + b];
                }
            }
        }
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
    }

    #[test]
    fn forward_negative_pre_activations_with_relu() {
        // Specifically test ReLU gating behavior
        let activation = Activation::ReLU;
        let ff = ff_with_params(
            &[-1.0, -1.0, 1.0, 1.0], // First neuron will be negative, second positive
            &[-5.0, 0.0],
            vec![2],
            activation,
        );
        let input = tensor_with_data(vec![2, 1], &[1.0, 1.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![2, 1]);
        // Neuron 0: -1*1 + -1*1 - 5 = -7 -> ReLU -> 0
        // Neuron 1: 1*1 + 1*1 + 0 = 2 -> ReLU -> 2
        assert_close(out.data.as_ref(), &[0.0, 2.0], 1e-12);
    }

    #[test]
    fn backward_relu_gates_gradient_for_negative_preactivations() {
        let activation = Activation::ReLU;
        let mut ff = ff_with_params(
            &[-1.0, -1.0, 1.0, 1.0],
            &[-5.0, 0.0],
            vec![2],
            activation,
        );

        let input = tensor_with_data(vec![2, 1], &[1.0, 1.0]);
        let grad = tensor_with_data(vec![2, 1], &[1.0, 1.0]);

        let lr = 0.1;
        let activations_out = ff.forward(input.cloned_view());
        let input_grad = ff.backward(lr, &input, &activations_out, grad);

        assert_eq!(input_grad.shape, vec![2, 1]);
        // Neuron 0 has negative pre-activation, so its gradient is gated to 0
        // Neuron 1 passes gradient through
        // Input grad = W^T * gated_grad
        // Only neuron 1 contributes: [1.0, 1.0]
        assert_close(input_grad.data.as_ref(), &[1.0, 1.0], 1e-12);

        // Bias 0 unchanged (gated), bias 1 updated
        assert_close(ff.biases.data.as_ref(), &[-5.0, -0.1], 1e-12);

        // Weights for neuron 0 unchanged, neuron 1 updated
        let expected_weights = vec![-1.0, -1.0, 0.9, 0.9];
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
    }
}
