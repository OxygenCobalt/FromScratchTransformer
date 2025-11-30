use std::simd::{Simd, num::SimdFloat};

use rand_distr::{Normal, Distribution};
use crate::{ml::activation::Activation, tensor::{cpu2::{self, FillUninit, CPUTensor, Fill, Generate}}};

const LANES: usize = 8; // number of SIMD lanes

pub struct FeedForward<T> {
    weights: T,
    biases: T,
    activation: Activation,
    // might be better to keep on stack rather than reading weights.shape[0] over and over
    // todo: evaluate if this is better and maybe try to move other things on-stack
    neurons: usize,
    // avoid recomputing these every forward/backward pass
    flattened_input_shape: usize,
    flattened_input_ndim: usize,
}

impl FeedForward<CPUTensor<'_>> {
    pub fn new(input_shape: Vec<usize>, neurons: usize, activation: Activation) -> Self {
        let flattened = cpu2::length_of(&input_shape);
        let xavier = Normal::new(0.0, 2.0 / (flattened + neurons) as f64).unwrap();
        Self {
            weights: CPUTensor::init(Generate {
                shape: vec![neurons, flattened],
                with: || xavier.sample(&mut rand::rng()),
            })
            .unwrap(),
            biases: CPUTensor::init(Fill {
                shape: vec![neurons],
                with: 0.0,
            })
            .unwrap(),
            activation,
            neurons,
            flattened_input_shape: flattened,
            flattened_input_ndim: input_shape.len(),
        }
    }

    pub fn forward<'i, 'o>(&self, activations_in: CPUTensor<'i>) -> CPUTensor<'o> {
        // flatten all trailing dims (batch etc)
        let mut out_shape = vec![self.neurons];
        out_shape.extend_from_slice(&activations_in.shape[self.flattened_input_ndim..]);
        let trailer = activations_in.shape[self.flattened_input_ndim..].iter().product();
        let flat_activations_in = activations_in.reshape(&[self.flattened_input_shape, trailer]).unwrap();
        let mut flat_activations_out = CPUTensor::init(unsafe { FillUninit::new(vec![self.neurons, trailer])}).unwrap();

        // forward pass: Wx + b -> [neurons, flattened] x [flattened, trailer] -> [neurons, trailer] + [neurons]
        let flat_activations_in_t = flat_activations_in.transpose(&[1, 0]).unwrap().materialize();
        let mut lhs_idx = 0;
        let mut rhs_idx = 0;
        let mut bias_idx = 0;
        let mut out_idx = 0;
        // explicit slice definitions to signal to the compiler about aliasing
        let out_data = flat_activations_out.data.to_mut().as_mut_slice();
        for _ in 0..self.neurons {
            let bias=  unsafe { *self.biases.data.get_unchecked(bias_idx) };
            for _ in 0..trailer {
                let mut sum: f64 = 0.0;
                for _ in 0..self.flattened_input_shape {
                    let lhs = unsafe { *self.weights.data.get_unchecked(lhs_idx) };
                    let rhs = unsafe { *flat_activations_in_t.data.get_unchecked(rhs_idx) };
                    sum += lhs * rhs;
                    lhs_idx += self.weights.stride[1];
                    rhs_idx += flat_activations_in_t.stride[1];
                }
                unsafe {
                    *out_data.get_unchecked_mut(out_idx) = sum + bias;
                }
                lhs_idx -= self.weights.stride[1] * self.flattened_input_shape;
                rhs_idx -= flat_activations_in_t.stride[1] * self.flattened_input_shape;
                rhs_idx += flat_activations_in_t.stride[0];
                out_idx += flat_activations_out.stride[1];
            }
            lhs_idx += self.weights.stride[0];
            rhs_idx = 0;
            bias_idx += self.biases.stride[0];
            out_idx -= flat_activations_out.stride[1] * trailer;
            out_idx += flat_activations_out.stride[0];
        }

        self.activation.forward_all(flat_activations_out.into_reshape(&out_shape).unwrap())
    }

    pub fn backward<'i, 'o>(&mut self, c: f64, activations_in: CPUTensor<'i>, mut grad: CPUTensor<'i>) -> CPUTensor<'o> {
        // pass 1: backwards gradient
        // todo: fuse activations when feasible
        grad = self.activation.backward_all(&activations_in, grad);

        // flatten out extra dims (batch etc)
        let original_shape = activations_in.shape.clone();
        let trailer: usize = activations_in.shape[self.flattened_input_ndim..].iter().product();
        let flat_grad = grad.reshape(&[self.neurons, trailer]).unwrap();
        let flat_activations_in = activations_in.reshape(&[self.flattened_input_shape, trailer]).unwrap();
        
        // pass 2: backwards biases
        let mut bias_idx = 0;
        let mut grad_idx = 0;
        let bias_data = self.biases.data.to_mut().as_mut_slice();
        for _ in 0..self.neurons {
            for _ in 0..trailer {
                let bias = unsafe {
                    bias_data.get_unchecked_mut(bias_idx)
                };
                let grad = unsafe {
                    flat_grad.data.get_unchecked(grad_idx)
                };
                *bias -= c * *grad;
                grad_idx += flat_grad.stride[1];
            }
            bias_idx += self.biases.stride[0];
            grad_idx -= flat_grad.stride[1] * trailer;
            grad_idx += flat_grad.stride[0];
        }

        // pass 3: activations backwards (W^T dot grad) [flattened, neurons] x [neurons, trailer] -> [flattened, trailer]
        // todo: fuse with pass 4 since their bounds are the same just in a different order
        // todo: evaluate if fusing with bias backwards with a branch is better than two distinct passes
        let mut a_grad = flat_activations_in.cloned_view();
        let weights_t = self.weights.transpose(&[1, 0]).unwrap().materialize();
        let grad_t = flat_grad.transpose(&[1, 0]).unwrap().materialize();
        let mut lhs_idx = 0;
        let mut rhs_idx = 0;
        let mut out_idx = 0;
        let out_data: &mut [f64] = a_grad.data.to_mut().as_mut_slice();
        for _ in 0..self.flattened_input_shape {
            for _ in 0..trailer {
                let chunks = self.neurons / LANES;
                let remainder = self.neurons % LANES;
                let mut sum_simd: Simd<f64, LANES> = Simd::splat(0.0);
                let mut lhs_ptr = unsafe { weights_t.data.as_ptr().add(lhs_idx) };
                let mut rhs_ptr = unsafe { grad_t.data.as_ptr().add(rhs_idx) };
                for _ in 0..chunks {
                    let lhs_simd = unsafe { std::ptr::read_unaligned(lhs_ptr as *const Simd<f64, LANES>) };
                    let rhs_simd = unsafe { std::ptr::read_unaligned(rhs_ptr as *const Simd<f64, LANES>) };
                    sum_simd += lhs_simd * rhs_simd;
                    lhs_ptr = unsafe { lhs_ptr.add(LANES) };
                    rhs_ptr = unsafe { rhs_ptr.add(LANES) };
                    lhs_idx += weights_t.stride[1] * LANES;
                    rhs_idx += grad_t.stride[1] * LANES;
                }
                let mut sum = sum_simd.reduce_sum();
                for _ in (self.neurons - remainder)..self.neurons {
                    let lhs = unsafe { *weights_t.data.get_unchecked(lhs_idx) };
                    let rhs = unsafe { *grad_t.data.get_unchecked(rhs_idx) };
                    sum += lhs * rhs;
                    lhs_idx += weights_t.stride[1];
                    rhs_idx += grad_t.stride[1];
                }
                unsafe {
                    *out_data.get_unchecked_mut(out_idx) = sum;
                }
                lhs_idx -= weights_t.stride[1] * self.neurons;
                rhs_idx -= grad_t.stride[1] * self.neurons;
                rhs_idx += grad_t.stride[0];
                out_idx += a_grad.stride[1];
            }
            lhs_idx += weights_t.stride[0];
            rhs_idx = 0;
            out_idx -= a_grad.stride[1] * trailer;
            out_idx += a_grad.stride[0];
        }

        // pass 4: weights backwards (grad dot a_in^T) [neurons, trailer] dot [trailer, flattened] -> [neurons, flattened] -> lhs_grad
        let activations_in_t = flat_activations_in.transpose(&[1, 0]).unwrap();
        let mut lhs_idx = 0;
        let mut rhs_idx = 0;
        let mut out_idx = 0;
        let out_data = self.weights.data.to_mut().as_mut_slice();
        for _ in 0..self.neurons {
            for _ in 0..self.flattened_input_shape {
                let mut sum: f64 = 0.0;
                for _ in 0..trailer {
                    let lhs = unsafe { *flat_grad.data.get_unchecked(lhs_idx) };
                    let rhs = unsafe { *activations_in_t.data.get_unchecked(rhs_idx) };
                    sum += lhs * rhs;
                    lhs_idx += flat_grad.stride[1];
                    rhs_idx += activations_in_t.stride[0];
                }
                unsafe {
                    let out = out_data.get_unchecked_mut(out_idx);
                    *out -= c * sum;
                }
                lhs_idx -= flat_grad.stride[1] * trailer;
                rhs_idx -= activations_in_t.stride[0] * trailer;
                rhs_idx += activations_in_t.stride[1];
                out_idx += self.weights.stride[1];
            }
            rhs_idx = 0;
            lhs_idx += flat_grad.stride[0];
            out_idx -= self.weights.stride[1] * self.flattened_input_shape;
            out_idx += self.weights.stride[0];
        }
        
        a_grad.into_reshape(&original_shape).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::cpu2;

    fn tensor_with_data(shape: Vec<usize>, values: &[f64]) -> CPUTensor<'static> {
        let mut tensor = CPUTensor::init(Fill { shape, with: 0.0 }).expect("tensor init");
        assert_eq!(tensor.data.len(), values.len());
        tensor.data.to_mut().clone_from_slice(values);
        tensor
    }

    fn ff_with_params(
        weights: &[f64],
        biases: &[f64],
        input_shape: Vec<usize>,
        activation: Activation,
    ) -> FeedForward<CPUTensor<'static>> {
        let neurons = biases.len();
        let flattened = cpu2::length_of(&input_shape);
        assert_eq!(
            weights.len(),
            neurons * flattened,
            "weights must be neurons x flattened_input_shape"
        );

        let mut w = CPUTensor::init(Fill {
            shape: vec![neurons, flattened],
            with: 0.0,
        })
        .unwrap();
        w.data.to_mut().clone_from_slice(weights);

        let mut b = CPUTensor::init(Fill {
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
            flattened_input_shape: flattened,
            flattened_input_ndim: input_shape.len(),
        }
    }

    fn ff_zeroed(
        input_shape: Vec<usize>,
        neurons: usize,
        activation: Activation,
    ) -> FeedForward<CPUTensor<'static>> {
        let flattened = cpu2::length_of(&input_shape);
        FeedForward {
            weights: CPUTensor::init(Fill {
                shape: vec![neurons, flattened],
                with: 0.0,
            })
            .unwrap(),
            biases: CPUTensor::init(Fill {
                shape: vec![neurons],
                with: 0.0,
            })
            .unwrap(),
            activation,
            neurons,
            flattened_input_shape: flattened,
            flattened_input_ndim: input_shape.len(),
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
        let ff = ff_with_params(
            &[0.5, -1.0, 1.5, 0.3],
            &[0.2, -0.1],
            vec![2],
            activation,
        );
        let input = tensor_with_data(vec![2], &[0.4, 0.8]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![2]);
        let expected = vec![
            activation.forward(0.5 * 0.4 + -1.0 * 0.8 + 0.2),
            activation.forward(1.5 * 0.4 + 0.3 * 0.8 - 0.1),
        ];
        assert_close(out.data.as_ref(), &expected, 1e-12);
    }

    #[test]
    fn forward_flattens_trailing_batch_dims() {
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
        let input = tensor_with_data(vec![3], &[2.0, 3.0, 4.0]);
        let grad = tensor_with_data(vec![2], &[0.2, -0.4]);

        let lr = 0.1;
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![3]);
        let expected_input_grad = vec![0.0, -0.6, 0.3];
        assert_close(input_grad.data.as_ref(), &expected_input_grad, 1e-12);

        let expected_biases = vec![2.98, 1.04];
        assert_close(ff.biases.data.as_ref(), &expected_biases, 1e-12);

        let expected_weights = vec![0.96, -1.06, 0.42, 0.58, 1.12, -0.34];
        assert_close(ff.weights.data.as_ref(), &expected_weights, 1e-12);
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

        let mut input =
            CPUTensor::init(Fill { shape: vec![features, batch], with: 0.0 }).unwrap();
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
        let mut input =
            CPUTensor::init(Fill { shape: vec![features, batch], with: 0.0 }).unwrap();
        {
            let data = input.data.to_mut();
            for b in 0..batch {
                data[0 * batch + b] = 1.0;
                data[(features - 1) * batch + b] = 2.0;
            }
        }

        // Gradients: neuron 0 -> +1.0, neuron 1 -> -1.0, last neuron -> +0.5 for every batch.
        let mut grad = CPUTensor::init(Fill {
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
        let a_grad = ff.backward(lr, input, grad);

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
}
