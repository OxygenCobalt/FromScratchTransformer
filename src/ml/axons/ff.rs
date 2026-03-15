use std::{
    ops::Add,
    simd::{Simd, num::SimdFloat},
    slice,
};

use crate::tensor::cpu2::{self, Fill, FillUninit, Generate, Tensor, TensorView};
use rand_distr::{Distribution, Normal};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::io::{self, Read, Write};

const LANES: usize = 8; // number of SIMD lanes
const FORWARD_MIN_FLOPS_PER_CORE: usize = 512_000 / 24; // measured on bench suite. adjusted to apply to any core count
// Backward pass thresholds derived from benching/benchmarks.csv (multithread vs initial).
// Backward total crossover ~1.18e7 FLOPs; split across two passes ~= 5.9e6 FLOPs.
const BACKWARD_PASS1_MIN_FLOPS_PER_CORE: usize = 6_000_000 / 24;
const BACKWARD_PASS2_MIN_FLOPS_PER_CORE: usize = 6_000_000 / 24;

pub struct FeedForward {
    weights: Tensor,
    biases: Tensor,
    // avoid recomputing these every forward/backward pass
    fan_in: usize,
    // might be better to keep on stack rather than reading weights.shape[0] over and over
    // todo: evaluate if this is better and maybe try to move other things on-stack
    neurons: usize,
    batch_idx: usize,
}

impl FeedForward {
    pub fn new(input_shape: Vec<usize>, neurons: usize) -> Self {
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
            fan_in,
            neurons,
            batch_idx: input_shape.len(),
        }
    }

    pub fn forward(&self, a_in: Tensor) -> Tensor {
        // forward pass: Wx + b -> [neurons, fan_in] x [fan_in, batch] -> [neurons, batch] + [neurons]
        let batch = a_in.shape[0];
        let a_in_flat = a_in.r(&[batch, self.fan_in]).unwrap();
        let mut a_out_flat =
            Tensor::init(unsafe { FillUninit::new(vec![batch, self.neurons]) }).unwrap();
        // explicit slice definitions to signal to the compiler about aliasing
        let a_out_flat_data = a_out_flat.data.as_mut_slice();
        let parallelism = rayon::current_num_threads();
        let flops_per_core = (self.neurons * self.fan_in * batch) / parallelism;
        if flops_per_core >= FORWARD_MIN_FLOPS_PER_CORE {
            let neurons_per_chunk = (self.neurons as f64 / parallelism as f64).ceil() as usize;
            let a_out_flat_addy = a_out_flat_data.as_mut_ptr() as usize;
            (0..self.neurons)
                .step_by(neurons_per_chunk)
                .collect::<Vec<usize>>()
                .into_par_iter()
                .for_each(|start_neuron| {
                    let neuron_count = neurons_per_chunk.min(self.neurons - start_neuron);
                    let out_ptr = a_out_flat_addy as *mut f64;
                    self.forward_mm(start_neuron, neuron_count, batch, &a_in_flat, out_ptr);
                });
        } else {
            self.forward_mm(
                0,
                self.neurons,
                batch,
                &a_in_flat,
                a_out_flat_data.as_mut_ptr(),
            );
        }
        a_out_flat
    }

    #[inline(always)]
    fn forward_mm(
        &self,
        start_neuron: usize,
        neuron_count: usize,
        batch: usize,
        a_in: &TensorView<'_>,
        a_out: *mut f64,
    ) {
        let mut weights_idx = self.fan_in * start_neuron; // shape = [neurons, fan_in], stride = [fan_in, 1]
        let mut a_in_idx = 0; // shape = [batch, fan_in], stride = [fan_in, 1]
        let mut bias_idx = start_neuron; // shape = [neurons], stride = [1]
        let mut a_out_idx = start_neuron; // shape = [batch, neuron], stride = [neurons, 1]
        let chunks = self.fan_in / LANES;
        let remainder = self.fan_in % LANES;
        for _ in 0..batch {
            for _ in 0..neuron_count {
                let bias = unsafe { *self.biases.data.get_unchecked(bias_idx) };
                let mut weights_ptr = unsafe { self.weights.data.as_ptr().add(weights_idx) };
                let mut a_in_ptr = unsafe { a_in.tensor.data.as_ptr().add(a_in_idx) };
                let mut sum_simd: Simd<f64, LANES> = Simd::splat(0.0);
                for _ in 0..chunks {
                    let weights_simd =
                        unsafe { std::ptr::read_unaligned(weights_ptr as *const Simd<f64, LANES>) };
                    let a_in_simd =
                        unsafe { std::ptr::read_unaligned(a_in_ptr as *const Simd<f64, LANES>) };
                    sum_simd += weights_simd * a_in_simd;
                    weights_ptr = unsafe { weights_ptr.add(LANES) };
                    a_in_ptr = unsafe { a_in_ptr.add(LANES) };
                }
                let mut sum: f64 = sum_simd.reduce_sum();
                weights_idx += LANES * chunks;
                a_in_idx += LANES * chunks;
                for _ in 0..remainder {
                    unsafe {
                        sum += *self.weights.data.get_unchecked(weights_idx)
                            * *a_in.tensor.data.get_unchecked(a_in_idx)
                    };
                    weights_idx += 1;
                    a_in_idx += 1;
                }
                unsafe {
                    *a_out.add(a_out_idx) = sum + bias;
                }
                a_in_idx -= self.fan_in;
                a_out_idx += 1;
                bias_idx += 1;
            }
            a_in_idx += self.fan_in;
            weights_idx = self.fan_in * start_neuron;
            bias_idx = start_neuron;
            a_out_idx -= neuron_count;
            a_out_idx += self.neurons;
        }
    }

    pub fn backward(&mut self, c: f64, a_in: Tensor, grad: Tensor) -> Tensor {
        let batch = a_in.shape[0];
        let parallelism = rayon::current_num_threads();
        let flops_per_core = (self.neurons * self.fan_in * batch) / parallelism;

        // pass 1: activations backwards (W^T dot grad) [fan_in, neurons] x [neurons, batch] -> [fan_in, batch]
        let mut a_grad = Tensor::init(Fill::null(vec![batch, self.fan_in])).unwrap();
        let a_grad_ptr = a_grad.data.as_mut_slice().as_mut_ptr();
        if flops_per_core >= BACKWARD_PASS1_MIN_FLOPS_PER_CORE {
            let fan_in_per_chunk = (self.fan_in as f64 / parallelism as f64).ceil() as usize;
            let a_grad_addy = a_grad_ptr as usize;
            (0..self.fan_in)
                .step_by(fan_in_per_chunk)
                .collect::<Vec<usize>>()
                .into_par_iter()
                .for_each(|start_fan_in| {
                    let fan_in_count = fan_in_per_chunk.min(self.fan_in - start_fan_in);
                    let a_grad_ptr = a_grad_addy as *mut f64;
                    Self::backwards_pass1_mm(
                        start_fan_in,
                        fan_in_count,
                        batch,
                        self.neurons,
                        self.fan_in,
                        &self.weights.data,
                        &grad.data,
                        a_grad_ptr,
                    );
                });
        } else {
            Self::backwards_pass1_mm(
                0,
                self.fan_in,
                batch,
                self.neurons,
                self.fan_in,
                &self.weights.data,
                &grad.data,
                a_grad_ptr,
            );
        }
        // pass 2: weights backwards (grad^T dot a_in) [neurons, batch] dot [batch, fan_in] -> [neurons, fan_in] -> weights
        //         bias backwards (sum over batch of grad) -> bias_grad
        let weights_data = self.weights.data.as_mut_slice();
        let bias_data = self.biases.data.as_mut_slice();
        if flops_per_core >= BACKWARD_PASS2_MIN_FLOPS_PER_CORE {
            // TODO: advanced backwards pass2 tiling now that we can be noncontiguous
            let neurons_per_chunk = (self.neurons as f64 / parallelism as f64).ceil() as usize;
            let neuron_grain = neurons_per_chunk * self.fan_in;
            let bias_grain = neurons_per_chunk;
            weights_data
                .par_chunks_mut(neuron_grain)
                .zip(bias_data.par_chunks_mut(bias_grain))
                .enumerate()
                .for_each(|(i, (out, bias))| {
                    let start_neuron = i * neurons_per_chunk;
                    let neuron_count = out.len() / self.fan_in;
                    Self::backwards_pass2_mm(
                        start_neuron,
                        neuron_count,
                        batch,
                        self.fan_in,
                        self.neurons,
                        c,
                        &a_in.data,
                        &grad.data,
                        bias,
                        out,
                    );
                });
        } else {
            Self::backwards_pass2_mm(
                0,
                self.neurons,
                batch,
                self.fan_in,
                self.neurons,
                c,
                &a_in.data,
                &grad.data,
                bias_data,
                weights_data,
            );
        }

        a_grad
    }

    #[inline(always)]
    fn backwards_pass1_mm(
        start_fan_in: usize,
        fan_in_count: usize,
        batch: usize,
        neurons: usize,
        fan_in: usize,
        weights: &[f64],  // [neurons, fan_in]
        grad: &[f64],     // [batch, neurons]
        a_grad: *mut f64, // [batch, fan_in]
    ) {
        let mut weights_idx = start_fan_in;
        let mut grad_idx = 0;
        let mut a_grad_idx = start_fan_in;
        let chunks = fan_in_count / LANES;
        let remainder = fan_in_count % LANES;
        for _ in 0..neurons {
            for _ in 0..batch {
                let g = unsafe { *grad.get_unchecked(grad_idx) };
                let g_simd: Simd<f64, LANES> = Simd::splat(g);
                let mut weights_ptr = unsafe { weights.as_ptr().add(weights_idx) };
                let mut a_grad_ptr = unsafe { a_grad.add(a_grad_idx) };
                for _ in 0..chunks {
                    let weights_simd =
                        unsafe { std::ptr::read_unaligned(weights_ptr as *const Simd<f64, LANES>) };
                    let a_grad_simd =
                        unsafe { std::ptr::read_unaligned(a_grad_ptr as *const Simd<f64, LANES>) };
                    let dot = a_grad_simd + g_simd * weights_simd;
                    unsafe { std::ptr::write_unaligned(a_grad_ptr as *mut Simd<f64, LANES>, dot) };
                    weights_ptr = unsafe { weights_ptr.add(LANES) };
                    a_grad_ptr = unsafe { a_grad_ptr.add(LANES) };
                }
                // advance weights across fan_in
                weights_idx += LANES * chunks;
                // advance a_grad across fan_in
                a_grad_idx += LANES * chunks;
                for _ in 0..remainder {
                    unsafe { *a_grad.add(a_grad_idx) += g * weights[weights_idx] }
                    // advance weights across fan_in
                    weights_idx += 1;
                    // advance a_grad across fan_in
                    a_grad_idx += 1;
                }
                // rewind weights across fan_in
                weights_idx -= fan_in_count;
                // rewind a_grad across fan_in
                a_grad_idx -= fan_in_count;
                // advance a_grad across batch
                a_grad_idx += fan_in;
                // advance grad across batch
                grad_idx += neurons;
            }
            // adv. weights across neurons
            weights_idx += fan_in;
            // rewind grad across batch
            grad_idx -= batch * neurons;
            // adv. grad across neurons
            grad_idx += 1;
            // rewind a_grad across batch
            a_grad_idx -= batch * fan_in;
        }
    }

    #[inline(always)]
    fn backwards_pass2_mm(
        start_neuron: usize,
        neuron_count: usize,
        batch: usize,
        fan_in: usize,
        neurons: usize,
        c: f64,
        a_in: &[f64],        // [batch, fan_in]
        grad: &[f64],        // [batch, neurons]
        bias: &mut [f64],    // [neurons]
        weights: &mut [f64], // [neurons, fan_in]
    ) {
        let c_simd = Simd::splat(c);
        let chunks = fan_in / LANES;
        let remainder = fan_in % LANES;
        let mut grad_idx = start_neuron;
        let mut a_in_idx = 0;
        let mut weights_idx = 0;
        for n in 0..neuron_count {
            let bias = unsafe { bias.get_unchecked_mut(n) };
            let mut sum_g = 0.0;
            for _ in 0..batch {
                // access gradient (also accumulate for bias update)
                let g = unsafe { *grad.get_unchecked(grad_idx) };
                sum_g += g;

                let g_simd = Simd::splat(g);
                let mut a_in_ptr = unsafe { a_in.as_ptr().add(a_in_idx) };
                let mut weights_ptr = unsafe { weights.as_mut_ptr().add(weights_idx) };
                for _ in 0..chunks {
                    let a_in_simd =
                        unsafe { std::ptr::read_unaligned(a_in_ptr as *const Simd<f64, LANES>) };
                    let weights_simd =
                        unsafe { std::ptr::read_unaligned(weights_ptr as *const Simd<f64, LANES>) };
                    let descended = weights_simd - g_simd * a_in_simd * c_simd;
                    unsafe {
                        std::ptr::write_unaligned(weights_ptr as *mut Simd<f64, LANES>, descended)
                    };
                    a_in_ptr = unsafe { a_in_ptr.add(LANES) };
                    weights_ptr = unsafe { weights_ptr.add(LANES) };
                }
                a_in_idx += LANES * chunks;
                weights_idx += LANES * chunks;
                for _ in 0..remainder {
                    let a_in = unsafe { *a_in.get_unchecked(a_in_idx) };
                    let weight = unsafe { weights.get_unchecked_mut(weights_idx) };
                    *weight -= c * g * a_in;
                    a_in_idx += 1;
                    weights_idx += 1;
                }
                grad_idx += neurons;
                weights_idx -= fan_in;
            }
            // backwards to bias too since we actually are already traversing over the correct axes while backpropagating weights
            // (accumulated sum over batch to avoid unneeded writes / muls)
            *bias -= c * sum_g;
            grad_idx -= neurons * batch;
            grad_idx += 1;
            a_in_idx = 0;
            weights_idx += fan_in;
        }
    }

    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read_exact(&mut signature)?;
        if &signature != b"FeedFrwd" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid feedforward signature",
            ));
        }
        let mut finb = [0u8; 8];
        read.read_exact(&mut finb)?;
        let fan_in = usize::from_le_bytes(finb);
        let mut fisb = [0u8; 8];
        read.read_exact(&mut fisb)?;
        let neurons = usize::from_le_bytes(fisb);
        let mut bib = [0u8; 8];
        read.read_exact(&mut bib)?;
        let batch_idx = usize::from_le_bytes(bib);
        let weights = Tensor::read(read)?;
        let biases = Tensor::read(read)?;
        Ok(Self {
            weights,
            biases,
            fan_in,
            neurons,
            batch_idx,
        })
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"FeedFrwd")?;
        write.write_all(&self.fan_in.to_le_bytes())?;
        write.write_all(&self.neurons.to_le_bytes())?;
        write.write_all(&self.batch_idx.to_le_bytes())?;
        self.weights.write(write)?;
        self.biases.write(write)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::cpu2;

    fn tensor_with_data(shape: Vec<usize>, values: &[f64]) -> Tensor {
        let mut tensor = Tensor::init(Fill { shape, with: 0.0 }).expect("tensor init");
        assert_eq!(tensor.data.len(), values.len());
        tensor.data.as_mut_slice().clone_from_slice(values);
        tensor
    }

    fn ff_with_params(weights: &[f64], biases: &[f64], input_shape: Vec<usize>) -> FeedForward {
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
        w.data.as_mut_slice().clone_from_slice(weights);

        let mut b = Tensor::init(Fill {
            shape: vec![neurons],
            with: 0.0,
        })
        .unwrap();
        b.data.as_mut_slice().clone_from_slice(biases);

        FeedForward {
            weights: w,
            biases: b,
            neurons,
            fan_in: fan_in,
            batch_idx: 0,
        }
    }

    fn ff_zeroed(input_shape: Vec<usize>, neurons: usize) -> FeedForward {
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
            neurons,
            fan_in: fan_in,
            batch_idx: 0,
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
    fn forward_matches_matmul_plus_bias() {
        let ff = ff_with_params(&[0.5, -1.0, 1.5, 0.3], &[0.2, -0.1], vec![2]);
        let input = tensor_with_data(vec![1, 2], &[0.4, 0.8]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 2]);
        let expected = vec![-0.4, 0.74];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_handles_explicit_batch_dim() {
        let ff = ff_with_params(&[1.0, 2.0], &[0.0], vec![2]);
        // Shape [batch, feature]
        let input = tensor_with_data(vec![3, 2], &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![3, 1]);
        let expected = vec![21.0, 42.0, 63.0];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn backward_returns_input_grad_and_updates_params() {
        let mut ff = ff_with_params(&[1.0, -1.0, 0.5, 0.5, 1.0, -0.5], &[3.0, 1.0], vec![3]);
        let input = tensor_with_data(vec![1, 3], &[2.0, 3.0, 4.0]);
        let grad = tensor_with_data(vec![1, 2], &[0.2, -0.4]);

        let lr = 0.1;
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![1, 3]);
        let expected_input_grad = vec![0.0, -0.6, 0.3];
        assert_close(input_grad.data.as_slice(), &expected_input_grad, 1e-12);

        let expected_biases = vec![2.98, 1.04];
        assert_close(ff.biases.data.as_slice(), &expected_biases, 1e-12);

        let expected_weights = vec![0.96, -1.06, 0.42, 0.58, 1.12, -0.34];
        assert_close(ff.weights.data.as_slice(), &expected_weights, 1e-12);
    }

    #[test]
    fn backward_small_batch_updates_all_neurons() {
        let mut ff = ff_with_params(&[1.0, 2.0, -1.0, 1.0], &[0.0, -3.0], vec![2]);
        let input = tensor_with_data(vec![1, 2], &[1.0, 2.0]);
        let grad = tensor_with_data(vec![1, 2], &[0.5, 1.0]);

        let lr = 0.1;
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![1, 2]);
        let expected_weights = vec![0.95, 1.9, -1.1, 0.8];
        assert_close(ff.weights.data.as_slice(), &expected_weights, 1e-12);
        let expected_biases = vec![-0.05, -3.1];
        assert_close(ff.biases.data.as_slice(), &expected_biases, 1e-12);
        let expected_input_grad = vec![-0.5, 2.0];
        assert_close(input_grad.data.as_slice(), &expected_input_grad, 1e-12);
    }

    #[test]
    fn backward_large_batch_accumulates_gradients() {
        let batch = 64;
        let mut ff = ff_with_params(&[1.0, 2.0, 0.5, 1.5], &[0.1, -0.2], vec![2]);

        let mut input = Tensor::init(Fill {
            shape: vec![batch, 2],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.as_mut_slice();
            for b in 0..batch {
                data[b * 2 + 0] = 1.0;
                data[b * 2 + 1] = 2.0;
            }
        }

        let mut grad = Tensor::init(Fill {
            shape: vec![batch, 2],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.as_mut_slice();
            for b in 0..batch {
                data[b * 2 + 0] = 0.3;
                data[b * 2 + 1] = -0.7;
            }
        }

        let lr = 0.01;
        let a_grad = ff.backward(lr, input, grad);

        assert_eq!(a_grad.shape, vec![batch, 2]);
        let expected_weights = vec![0.808, 1.616, 0.948, 2.396];
        assert_close(ff.weights.data.as_slice(), &expected_weights, 1e-12);
        let expected_biases = vec![-0.092, 0.248];
        assert_close(ff.biases.data.as_slice(), &expected_biases, 1e-12);

        for b in 0..batch {
            let idx0 = b * 2 + 0;
            let idx1 = b * 2 + 1;
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
    fn backward_repeated_calls_keep_weight_layout_consistent() {
        let mut ff = ff_with_params(&[1.0, 2.0, 0.5, 1.5], &[0.1, -0.2], vec![2]);
        let input = tensor_with_data(vec![1, 2], &[1.0, 2.0]);
        let grad = tensor_with_data(vec![1, 2], &[0.3, -0.7]);
        let lr = 0.01;

        let first_input_grad = ff.backward(lr, input.clone(), grad.clone());
        let second_input_grad = ff.backward(lr, input, grad);

        assert_close(first_input_grad.data.as_slice(), &[-0.05, -0.45], 1e-12);
        assert_close(
            second_input_grad.data.as_slice(),
            &[-0.0558, -0.4616],
            1e-12,
        );
        assert_close(ff.biases.data.as_slice(), &[0.094, -0.186], 1e-12);
        assert_close(
            ff.weights.data.as_slice(),
            &[0.994, 1.988, 0.514, 1.528],
            1e-12,
        );
    }

    #[test]
    fn forward_large_dims_matches_expected_rows() {
        let neurons = 8_096;
        let features = 512;
        let batch = 16;

        let mut ff = ff_zeroed(vec![features], neurons);
        {
            let w = ff.weights.data.as_mut_slice();
            w[0 * features + 0] = 2.0;
            w[1 * features + (features - 1)] = -1.0;
        }
        {
            let b = ff.biases.data.as_mut_slice();
            b[0] = 0.5;
            b[1] = -0.5;
        }

        let mut input = Tensor::init(Fill {
            shape: vec![batch, features],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.as_mut_slice();
            for b in 0..batch {
                data[b * features + 0] = 1.0; // feature 0
                data[b * features + (features - 1)] = 2.0; // feature 511
            }
        }

        let out = ff.forward(input);
        assert_eq!(out.shape, vec![batch, neurons]);
        let out_data = out.data.as_slice();
        for b in 0..batch {
            let idx0 = b * neurons + 0;
            let idx1 = b * neurons + 1;
            let idx_last = b * neurons + (neurons - 1);
            assert!(
                (out_data[idx0] - 2.5).abs() <= 1e-12,
                "neuron 0 batch {} expected 2.5, got {}",
                b,
                out_data[idx0]
            );
            assert!(
                (out_data[idx1] + 2.5).abs() <= 1e-12,
                "neuron 1 batch {} expected -2.5, got {}",
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
        let neurons = 8_096;
        let features = 512;
        let batch = 16;
        let mut ff = ff_zeroed(vec![features], neurons);
        // Configure a few sparse weights and probe selected updates.
        {
            let w = ff.weights.data.as_mut_slice();
            w[0 * features + 0] = 1.0;
            w[1 * features + (features - 1)] = -1.0;
            w[(neurons - 1) * features + 0] = 0.5;
        }

        // Input: feature 0 is 1.0, feature 511 is 2.0 for every batch.
        let mut input = Tensor::init(Fill {
            shape: vec![batch, features],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.as_mut_slice();
            for b in 0..batch {
                data[b * features + 0] = 1.0;
                data[b * features + (features - 1)] = 2.0;
            }
        }

        // Gradients: neuron 0 -> +1.0, neuron 1 -> -1.0, last neuron -> +0.5 for every batch.
        let mut grad = Tensor::init(Fill {
            shape: vec![batch, neurons],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.as_mut_slice();
            for b in 0..batch {
                data[b * neurons + 0] = 1.0;
                data[b * neurons + 1] = -1.0;
                data[b * neurons + (neurons - 1)] = 0.5;
            }
        }

        let lr = 0.1;
        let a_grad = ff.backward(lr, input, grad);

        // Input gradient is W^T * grad.
        assert_eq!(a_grad.shape, vec![batch, features]);
        for b in 0..batch {
            let idx0 = b * features + 0;
            let idx511 = b * features + (features - 1);
            assert!(
                (a_grad.data[idx0] - 1.25).abs() <= 1e-12,
                "expected dL/dx0=1.25 at batch {}, got {}",
                b,
                a_grad.data[idx0]
            );
            assert!(
                (a_grad.data[idx511] - 1.0).abs() <= 1e-12,
                "expected dL/dx511=1 at batch {}, got {}",
                b,
                a_grad.data[idx511]
            );
        }

        // Bias updates: b -= lr * sum_batch(grad).
        let biases = ff.biases.data.as_slice();
        assert_close(&biases[0..3], &[-1.6, 1.6, 0.0], 1e-12);
        assert!(
            (biases[neurons - 1] + 0.8).abs() <= 1e-12,
            "expected last bias to be -0.8, got {}",
            biases[neurons - 1]
        );

        // Weight updates on selected entries:
        // w(0,0) starts at 1.0 and -= lr * sum(grad0 * x0) = 1.0 - 0.1 * 16 = -0.6
        // w(1,511) starts at -1.0 and -= lr * sum(grad1 * x511) = -1.0 - 0.1 * (-32) = 2.2
        // w(last,0) starts at 0.5 and -= lr * sum(grad_last * x0) = 0.5 - 0.1 * (0.5 * 16) = -0.3
        let weights = ff.weights.data.as_slice();
        assert!(
            (weights[0 * features + 0] + 0.6).abs() <= 1e-12,
            "expected w(0,0)=-0.6, got {}",
            weights[0 * features + 0]
        );
        assert!(
            (weights[1 * features + (features - 1)] - 2.2).abs() <= 1e-12,
            "expected w(1,511)=2.2, got {}",
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
        let features = 10; // 8-lane SIMD + 2-scalar remainder
        let neurons = 9; // 8-lane SIMD + 1-scalar remainder
        let batch = 3;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let biases: Vec<f64> = (0..neurons).map(|n| n as f64 * 0.05).collect();
        let ff = ff_with_params(&weights, &biases, vec![features]);

        let mut input = Tensor::init(Fill {
            shape: vec![batch, features],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.as_mut_slice();
            for b in 0..batch {
                for f in 0..features {
                    data[b * features + f] = 1.0 + f as f64 * 0.2 + b as f64 * 0.1;
                }
            }
        }

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![batch, neurons]);
        let mut expected = vec![0.0; batch * neurons];
        for b in 0..batch {
            for n in 0..neurons {
                let mut acc = biases[n];
                for f in 0..features {
                    acc += weights[n * features + f] * (1.0 + f as f64 * 0.2 + b as f64 * 0.1);
                }
                expected[b * neurons + n] = acc;
            }
        }
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn backward_non_multiple_of_lanes_dimensions_matches_reference() {
        let features = 10; // exercises remainder path on fan_in input dimension
        let neurons = 9; // exercises remainder path on neuron dimension
        let batch = 2;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.05 * (i as f64 + 1.0))
            .collect();
        let biases: Vec<f64> = (0..neurons).map(|n| 0.1 + n as f64 * 0.02).collect();
        let mut ff = ff_with_params(&weights, &biases, vec![features]);

        let mut input = Tensor::init(Fill {
            shape: vec![batch, features],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.as_mut_slice();
            for b in 0..batch {
                for f in 0..features {
                    data[b * features + f] = 1.0 + f as f64 * 0.3 + b as f64 * 0.05;
                }
            }
        }
        let mut grad = Tensor::init(Fill {
            shape: vec![batch, neurons],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.as_mut_slice();
            for b in 0..batch {
                for n in 0..neurons {
                    data[b * neurons + n] = 0.1 + n as f64 * 0.02 + b as f64 * 0.01;
                }
            }
        }

        let lr = 0.01;
        let input_data = input.data.clone();
        let grad_data = grad.data.clone();
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![batch, features]);

        let mut expected_input_grad = vec![0.0; batch * features];
        for b in 0..batch {
            for f in 0..features {
                let mut sum = 0.0;
                for n in 0..neurons {
                    sum += weights[n * features + f] * grad_data[b * neurons + n];
                }
                expected_input_grad[b * features + f] = sum;
            }
        }

        let mut expected_biases = biases.clone();
        for n in 0..neurons {
            let mut delta = 0.0;
            for b in 0..batch {
                delta += grad_data[b * neurons + n];
            }
            expected_biases[n] -= lr * delta;
        }

        let mut expected_weights = weights.clone();
        for n in 0..neurons {
            for f in 0..features {
                let mut delta = 0.0;
                for b in 0..batch {
                    delta += grad_data[b * neurons + n] * input_data[b * features + f];
                }
                expected_weights[n * features + f] -= lr * delta;
            }
        }

        assert_close(input_grad.data.as_slice(), &expected_input_grad, 1e-12);
        assert_close(ff.biases.data.as_slice(), &expected_biases, 1e-12);
        assert_close(ff.weights.data.as_slice(), &expected_weights, 1e-12);
    }

    #[test]
    fn forward_linear_values_case_a() {
        let ff = ff_with_params(&[1.0, 0.5, -0.5, 1.0], &[0.1, -0.1], vec![2]);
        let input = tensor_with_data(vec![1, 2], &[0.5, 0.8]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 2]);
        let expected = vec![1.0, 0.45];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_linear_values_case_b() {
        let ff = ff_with_params(&[2.0, -1.0, 0.5, 1.5], &[0.3, -0.2], vec![2]);
        let input = tensor_with_data(vec![1, 2], &[1.0, 2.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 2]);
        let pre0 = 2.0 * 1.0 + -1.0 * 2.0 + 0.3; // 0.3
        let pre1 = 0.5 * 1.0 + 1.5 * 2.0 - 0.2; // 3.3
        let expected = vec![pre0, pre1];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_dims_smaller_than_lanes() {
        // All dimensions smaller than SIMD lane width (8)
        let features = 3;
        let neurons = 5;
        let batch = 2;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let biases: Vec<f64> = (0..neurons).map(|n| n as f64 * 0.05).collect();
        let ff = ff_with_params(&weights, &biases, vec![features]);

        // batch-first: [batch, features] = [2, 3]
        // batch0=[1.0, 0.5, -0.5], batch1=[2.0, 1.5, 0.5]
        let input = tensor_with_data(vec![batch, features], &[1.0, 0.5, -0.5, 2.0, 1.5, 0.5]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![batch, neurons]);
        let input_data = [1.0, 0.5, -0.5, 2.0, 1.5, 0.5];
        let mut expected = vec![0.0; batch * neurons];
        for b in 0..batch {
            for n in 0..neurons {
                let mut acc = biases[n];
                for f in 0..features {
                    acc += weights[n * features + f] * input_data[b * features + f];
                }
                expected[b * neurons + n] = acc;
            }
        }
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_single_neuron() {
        let ff = ff_with_params(&[1.0, 2.0, 3.0], &[0.5], vec![3]);
        let input = tensor_with_data(vec![1, 3], &[0.1, 0.2, 0.3]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 1]);
        let expected = vec![1.9];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_single_feature() {
        let ff = ff_with_params(&[2.0, -1.0, 0.5], &[0.1, -0.5, 0.0], vec![1]);
        let input = tensor_with_data(vec![1, 1], &[3.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 3]);
        let expected = vec![6.1, -3.5, 1.5];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_single_batch() {
        let ff = ff_with_params(&[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0], vec![2]);
        let input = tensor_with_data(vec![1, 2], &[1.0, 1.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 2]);
        let expected = vec![3.0, 7.0];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_large_batch() {
        // Test with larger batch size to exercise parallel code paths
        let features = 4;
        let batch = 32;
        let neurons = 2;

        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        let biases = vec![0.0; neurons];
        let ff = ff_with_params(&weights, &biases, vec![features]);

        let mut input = Tensor::init(Fill {
            shape: vec![batch, features],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.as_mut_slice();
            for i in 0..data.len() {
                data[i] = 0.1 * (i as f64 + 1.0);
            }
        }

        let out = ff.forward(input.clone());

        assert_eq!(out.shape, vec![batch, neurons]);

        // Verify values
        let input_data = input.data.as_slice();
        for b in 0..batch {
            for n in 0..neurons {
                let mut acc = biases[n];
                for f in 0..features {
                    acc += weights[n * features + f] * input_data[b * features + f];
                }
                let expected = acc;
                let actual = out.data[b * neurons + n];
                assert!(
                    (actual - expected).abs() <= 1e-12,
                    "mismatch at neuron {} batch {}: expected {}, got {}",
                    n,
                    b,
                    expected,
                    actual
                );
            }
        }
    }

    #[test]
    fn backward_with_zero_gradients() {
        let mut ff = ff_with_params(&[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2], vec![2]);
        let original_weights = ff.weights.data.clone();
        let original_biases = ff.biases.data.clone();

        let input = tensor_with_data(vec![1, 2], &[1.0, 1.0]);
        let grad = tensor_with_data(vec![1, 2], &[0.0, 0.0]);

        let lr = 0.1;
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![1, 2]);
        // With zero gradients, weights and biases should remain unchanged
        assert_close(ff.weights.data.as_slice(), &original_weights, 1e-12);
        assert_close(ff.biases.data.as_slice(), &original_biases, 1e-12);
        // Input gradient should also be zero
        assert_close(input_grad.data.as_slice(), &[0.0, 0.0], 1e-12);
    }

    #[test]
    fn backward_with_zero_input() {
        // Use ReLU with positive biases so pre-activations are positive (derivative = 1)
        let mut ff = ff_with_params(&[1.0, 2.0, 3.0, 4.0], &[0.1, 0.2], vec![2]);
        let original_weights = ff.weights.data.clone();

        let input = tensor_with_data(vec![1, 2], &[0.0, 0.0]);
        let grad = tensor_with_data(vec![1, 2], &[1.0, 1.0]);

        let lr = 0.1;
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![1, 2]);
        // With zero input, weights should remain unchanged (gradient * input = gradient * 0 = 0)
        assert_close(ff.weights.data.as_slice(), &original_weights, 1e-12);

        // Biases: b -= lr * grad (ReLU derivative = 1 for positive pre-activations)
        let expected_biases = vec![0.1 - 0.1 * 1.0, 0.2 - 0.1 * 1.0];
        assert_close(ff.biases.data.as_slice(), &expected_biases, 1e-12);

        // Input gradient = W^T * grad
        let expected_input_grad = vec![1.0 + 3.0, 2.0 + 4.0]; // [4.0, 6.0]
        assert_close(input_grad.data.as_slice(), &expected_input_grad, 1e-12);
    }

    #[test]
    fn backward_dims_smaller_than_lanes() {
        // Use ReLU with positive weights/inputs/biases so all pre-activations are positive
        // This makes ReLU act like identity (derivative = 1)
        let features = 3;
        let neurons = 5;
        let batch = 2;

        // All positive weights
        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        // All positive biases
        let biases: Vec<f64> = (0..neurons).map(|n| 1.0 + n as f64 * 0.05).collect();
        let mut ff = ff_with_params(&weights, &biases, vec![features]);

        // batch-first: [batch, features] = [2, 3]
        // batch0=[1.0, 0.5, 0.5], batch1=[2.0, 1.5, 0.5]
        let input = tensor_with_data(vec![batch, features], &[1.0, 0.5, 0.5, 2.0, 1.5, 0.5]);
        // batch-first: [batch, neurons] = [2, 5]
        // batch0=[0.1, 0.3, 0.5, 0.7, 0.9], batch1=[0.2, 0.4, 0.6, 0.8, 1.0]
        let grad = tensor_with_data(
            vec![batch, neurons],
            &[0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0],
        );

        let lr = 0.01;
        let input_data = input.data.clone();
        let grad_data = grad.data.clone();
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![batch, features]);
        let mut expected_input_grad = vec![0.0; batch * features];
        for b in 0..batch {
            for f in 0..features {
                for n in 0..neurons {
                    expected_input_grad[b * features + f] +=
                        weights[n * features + f] * grad_data[b * neurons + n];
                }
            }
        }

        let mut expected_biases = biases.clone();
        for n in 0..neurons {
            for b in 0..batch {
                expected_biases[n] -= lr * grad_data[b * neurons + n];
            }
        }

        let mut expected_weights = weights.clone();
        for n in 0..neurons {
            for f in 0..features {
                for b in 0..batch {
                    expected_weights[n * features + f] -=
                        lr * grad_data[b * neurons + n] * input_data[b * features + f];
                }
            }
        }

        assert_close(input_grad.data.as_slice(), &expected_input_grad, 1e-12);
        assert_close(ff.biases.data.as_slice(), &expected_biases, 1e-12);
        assert_close(ff.weights.data.as_slice(), &expected_weights, 1e-12);
    }

    #[test]
    fn backward_single_neuron_single_feature() {
        let mut ff = ff_with_params(&[2.0], &[0.5], vec![1]);

        let input = tensor_with_data(vec![1, 1], &[3.0]);
        let grad = tensor_with_data(vec![1, 1], &[0.5]);

        let lr = 0.1;
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![1, 1]);
        // input_grad = W^T * grad = 2.0 * 0.5 = 1.0
        assert_close(input_grad.data.as_slice(), &[1.0], 1e-12);
        // bias -= lr * grad = 0.5 - 0.1 * 0.5 = 0.45
        assert_close(ff.biases.data.as_slice(), &[0.45], 1e-12);
        // weight -= lr * grad * input = 2.0 - 0.1 * 0.5 * 3.0 = 1.85
        assert_close(ff.weights.data.as_slice(), &[1.85], 1e-12);
    }

    #[test]
    fn backward_large_batch() {
        // Test backward with larger batch size to exercise different code paths
        let features = 4;
        let batch = 32;
        let neurons = 2;

        // Positive weights
        let weights: Vec<f64> = (0..neurons * features)
            .map(|i| 0.1 * (i as f64 + 1.0))
            .collect();
        // Positive biases ensure positive pre-activations
        let biases = vec![1.0, 1.0];
        let mut ff = ff_with_params(&weights, &biases, vec![features]);

        let mut input = Tensor::init(Fill {
            shape: vec![batch, features],
            with: 0.0,
        })
        .unwrap();
        {
            let data = input.data.as_mut_slice();
            // Positive inputs
            for i in 0..data.len() {
                data[i] = 0.05 * (i as f64 + 1.0);
            }
        }

        let mut grad = Tensor::init(Fill {
            shape: vec![batch, neurons],
            with: 0.0,
        })
        .unwrap();
        {
            let data = grad.data.as_mut_slice();
            for i in 0..data.len() {
                data[i] = 0.1 * (i as f64 + 1.0);
            }
        }

        let lr = 0.01;
        let input_data = input.data.clone();
        let grad_data = grad.data.clone();
        let input_grad = ff.backward(lr, input, grad);

        // Output shape should match input shape
        assert_eq!(input_grad.shape, vec![batch, features]);

        let mut expected_input_grad = vec![0.0; batch * features];
        for b in 0..batch {
            for f in 0..features {
                for n in 0..neurons {
                    expected_input_grad[b * features + f] +=
                        weights[n * features + f] * grad_data[b * neurons + n];
                }
            }
        }
        assert_close(input_grad.data.as_slice(), &expected_input_grad, 1e-12);

        // Compute expected biases
        let mut expected_biases = biases.clone();
        for n in 0..neurons {
            for b in 0..batch {
                expected_biases[n] -= lr * grad_data[b * neurons + n];
            }
        }
        assert_close(ff.biases.data.as_slice(), &expected_biases, 1e-12);

        // Compute expected weights
        let mut expected_weights = weights.clone();
        for n in 0..neurons {
            for f in 0..features {
                for b in 0..batch {
                    expected_weights[n * features + f] -=
                        lr * grad_data[b * neurons + n] * input_data[b * features + f];
                }
            }
        }
        assert_close(ff.weights.data.as_slice(), &expected_weights, 1e-12);
    }

    #[test]
    fn forward_allows_negative_outputs() {
        let ff = ff_with_params(
            &[-1.0, -1.0, 1.0, 1.0], // First neuron will be negative, second positive
            &[-5.0, 0.0],
            vec![2],
        );
        let input = tensor_with_data(vec![1, 2], &[1.0, 1.0]);

        let out = ff.forward(input);

        assert_eq!(out.shape, vec![1, 2]);
        assert_close(out.data.as_slice(), &[-7.0, 2.0], 1e-12);
    }

    #[test]
    fn backward_does_not_gate_gradients_by_preactivation() {
        let mut ff = ff_with_params(&[-1.0, -1.0, 1.0, 1.0], &[-5.0, 0.0], vec![2]);

        let input = tensor_with_data(vec![1, 2], &[1.0, 1.0]);
        let grad = tensor_with_data(vec![1, 2], &[1.0, 1.0]);

        let lr = 0.1;
        let input_grad = ff.backward(lr, input, grad);

        assert_eq!(input_grad.shape, vec![1, 2]);
        assert_close(input_grad.data.as_slice(), &[0.0, 0.0], 1e-12);

        assert_close(ff.biases.data.as_slice(), &[-5.1, -0.1], 1e-12);

        let expected_weights = vec![-1.1, -1.1, 0.9, 0.9];
        assert_close(ff.weights.data.as_slice(), &expected_weights, 1e-12);
    }
}
