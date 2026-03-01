use std::ops::Neg;

use crate::{
    dataset::{Example, Test},
    ml::nn2::{NeuralNetwork, Reporting},
    tensor::cpu2::{Fill, Scalar, Tensor},
};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    MSE,
    LogLikelihood,
    Accuracy(AccuracyOf),
}

pub struct Losses {
    pub loss: Tensor,
    pub prime: Tensor,
}

impl Loss {
    pub fn run(&self, batch_activations: &Tensor, output: &Tensor) -> Losses {
        match self {
            Loss::MSE => {
                let loss_data: Vec<f64> = batch_activations
                    .data
                    .iter()
                    .zip(output.data.iter())
                    .map(|(a, o)| (a - o).powi(2))
                    .collect();
                let prime_data: Vec<f64> = batch_activations
                    .data
                    .iter()
                    .zip(output.data.iter())
                    .map(|(a, o)| 2.0 * (a - o))
                    .collect();
                let loss = Tensor {
                    shape: batch_activations.shape.clone(),
                    stride: batch_activations.stride.clone(),
                    data: loss_data,
                };
                let prime: Tensor = Tensor {
                    shape: batch_activations.shape.clone(),
                    stride: batch_activations.stride.clone(),
                    data: prime_data,
                };
                Losses { loss, prime }
            }
            Loss::LogLikelihood => {
                let classes = batch_activations.shape[0];
                let batches = batch_activations.shape[1];
                let mut log_lik = Tensor::init(Fill::null(vec![batches])).unwrap();
                let mut log_lik_prime = Tensor::init(Fill::null(vec![classes, batches])).unwrap();
                let mut batch_activations_idx = 0;
                for batch in 0..batches {
                    let mut argmax_idx = batch_activations_idx;
                    let mut argmax = batch_activations_idx;
                    let mut max = f64::MIN;
                    for _ in 0..classes {
                        let x = output.data[argmax_idx];
                        if x > max {
                            argmax = argmax_idx;
                            max = x;
                        }
                        argmax_idx += output.stride[0];
                    }
                    log_lik.data[batch] = batch_activations.data[argmax].ln().neg();
                    log_lik_prime.data[argmax] = -1.0 / batch_activations.data[argmax];
                    // batch_activations_idx -= batch_activations.stride[0] * classes;
                    batch_activations_idx += batch_activations.stride[1];
                }
                Losses {
                    loss: log_lik,
                    prime: log_lik_prime,
                }
            }
            Loss::Accuracy(of) => {
                let loss = if of.accurate(batch_activations, output) {
                    Tensor::init(Scalar(1.0))
                } else {
                    Tensor::init(Scalar(0.0))
                }
                .unwrap();
                Losses {
                    loss,
                    // todo: hacky workaround fixfixfix
                    prime: Tensor::init(Scalar(0.0)).unwrap(),
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccuracyOf {
    Argmax,
}

impl AccuracyOf {
    pub fn accurate(&self, batch_activations: &Tensor, output: &Tensor) -> bool {
        match self {
            AccuracyOf::Argmax => {
                // todo: remove this bad argmax impl if you want to use this as an actual loss fn
                fn flat_argmax(tensor: &Tensor) -> usize {
                    tensor
                        .data
                        .iter()
                        .enumerate()
                        .max_by(|(_, x), (_, y)| {
                            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap()
                        .0
                }
                let predicted = flat_argmax(batch_activations);
                let actual = flat_argmax(output);
                predicted == actual
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::cpu2::Fill;

    fn tensor_with_data(shape: Vec<usize>, values: &[f64]) -> Tensor {
        let mut t = Tensor::init(Fill::null(shape)).unwrap();
        assert_eq!(t.data.len(), values.len(), "shape/data length mismatch");
        t.data.copy_from_slice(values);
        t
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "index {i}: expected {e}, got {a} (diff {})",
                (a - e).abs()
            );
        }
    }

    // ── MSE tests ──

    #[test]
    fn mse_identical_inputs_gives_zero_loss() {
        let activations = tensor_with_data(vec![3], &[1.0, 2.0, 3.0]);
        let output = tensor_with_data(vec![3], &[1.0, 2.0, 3.0]);
        let result = Loss::MSE.run(&activations, &output);
        assert_close(&result.loss.data, &[0.0, 0.0, 0.0], 1e-12);
        assert_close(&result.prime.data, &[0.0, 0.0, 0.0], 1e-12);
    }

    #[test]
    fn mse_loss_values() {
        // activations = [1, 4], output = [2, 1]
        // loss = [(1-2)^2, (4-1)^2] = [1, 9]
        // prime = [2*(1-2), 2*(4-1)] = [-2, 6]
        let activations = tensor_with_data(vec![2], &[1.0, 4.0]);
        let output = tensor_with_data(vec![2], &[2.0, 1.0]);
        let result = Loss::MSE.run(&activations, &output);
        assert_close(&result.loss.data, &[1.0, 9.0], 1e-12);
        assert_close(&result.prime.data, &[-2.0, 6.0], 1e-12);
    }

    #[test]
    fn mse_preserves_shape_and_stride() {
        let activations = tensor_with_data(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = tensor_with_data(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = Loss::MSE.run(&activations, &output);
        assert_eq!(result.loss.shape, vec![2, 3]);
        assert_eq!(result.loss.stride, activations.stride);
        assert_eq!(result.prime.shape, vec![2, 3]);
        assert_eq!(result.prime.stride, activations.stride);
    }

    #[test]
    fn mse_2d_loss_values() {
        // shape [2, 2], activations = [[1, 2], [3, 4]], output = [[0, 0], [0, 0]]
        // loss element-wise: [1, 4, 9, 16]
        // prime element-wise: [2, 4, 6, 8]
        let activations = tensor_with_data(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let output = tensor_with_data(vec![2, 2], &[0.0, 0.0, 0.0, 0.0]);
        let result = Loss::MSE.run(&activations, &output);
        assert_close(&result.loss.data, &[1.0, 4.0, 9.0, 16.0], 1e-12);
        assert_close(&result.prime.data, &[2.0, 4.0, 6.0, 8.0], 1e-12);
    }

    #[test]
    fn mse_negative_differences() {
        // activations < output: prime should be negative
        let activations = tensor_with_data(vec![2], &[0.0, 0.0]);
        let output = tensor_with_data(vec![2], &[3.0, 5.0]);
        let result = Loss::MSE.run(&activations, &output);
        assert_close(&result.loss.data, &[9.0, 25.0], 1e-12);
        assert_close(&result.prime.data, &[-6.0, -10.0], 1e-12);
    }

    // ── LogLikelihood tests ──
    // loss = -ln(activation_at_target_class)
    // prime = -1/activation_at_target_class (at target index), 0 elsewhere

    #[test]
    fn log_likelihood_single_batch_picks_correct_class() {
        // shape [3, 1]: 3 classes, 1 batch
        // output (one-hot): class 1 is the target -> [0, 1, 0]
        let activations = tensor_with_data(vec![3, 1], &[0.1, 0.7, 0.2]);
        let output = tensor_with_data(vec![3, 1], &[0.0, 1.0, 0.0]);
        let result = Loss::LogLikelihood.run(&activations, &output);
        // loss = -ln(0.7)
        assert_eq!(result.loss.shape, vec![1]);
        assert_close(&result.loss.data, &[-(0.7_f64.ln())], 1e-12);
        // prime = -1/0.7 at class 1, 0 elsewhere
        assert_eq!(result.prime.shape, vec![3, 1]);
        assert_close(&result.prime.data, &[0.0, -1.0 / 0.7, 0.0], 1e-12);
    }

    #[test]
    fn log_likelihood_multi_batch_picks_per_batch_class() {
        // shape [3, 2]: 3 classes, 2 batches
        // stride = [2, 1]
        // Data layout (row-major): [c0b0, c0b1, c1b0, c1b1, c2b0, c2b1]
        //
        // output: batch 0 target = class 2, batch 1 target = class 0
        //   class 0: [0, 1]
        //   class 1: [0, 0]
        //   class 2: [1, 0]
        // flat: [0, 1, 0, 0, 1, 0]
        let activations = tensor_with_data(vec![3, 2], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let output = tensor_with_data(vec![3, 2], &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);

        let result = Loss::LogLikelihood.run(&activations, &output);

        // batch 0: target class = 2, activation at (2,0) = 0.5, loss = -ln(0.5)
        // batch 1: target class = 0, activation at (0,1) = 0.2, loss = -ln(0.2)
        assert_eq!(result.loss.shape, vec![2]);
        assert_close(
            &result.loss.data,
            &[-(0.5_f64.ln()), -(0.2_f64.ln())],
            1e-12,
        );

        // prime: -1/x at target class for each batch, 0 elsewhere
        // (0,1) = -1/0.2 = -5.0, (2,0) = -1/0.5 = -2.0
        // flat: [0, -5, 0, 0, -2, 0]
        assert_eq!(result.prime.shape, vec![3, 2]);
        assert_close(
            &result.prime.data,
            &[0.0, -1.0 / 0.2, 0.0, 0.0, -1.0 / 0.5, 0.0],
            1e-12,
        );
    }

    #[test]
    fn log_likelihood_prime_indexing_is_correct() {
        // shape [4, 1]: 4 classes, 1 batch
        // output one-hot at class 3: [0, 0, 0, 1]
        let activations = tensor_with_data(vec![4, 1], &[0.1, 0.2, 0.3, 0.9]);
        let output = tensor_with_data(vec![4, 1], &[0.0, 0.0, 0.0, 1.0]);
        let result = Loss::LogLikelihood.run(&activations, &output);

        // loss = -ln(0.9)
        assert_close(&result.loss.data, &[-(0.9_f64.ln())], 1e-12);
        // prime = [0, 0, 0, -1/0.9]
        assert_eq!(result.prime.data.len(), 4);
        assert_close(&result.prime.data, &[0.0, 0.0, 0.0, -1.0 / 0.9], 1e-12);
    }

    #[test]
    fn log_likelihood_perfect_confidence() {
        // activation = 1.0 at target class → loss = -ln(1) = 0, prime = -1/1 = -1
        let activations = tensor_with_data(vec![2, 1], &[0.0, 1.0]);
        let output = tensor_with_data(vec![2, 1], &[0.0, 1.0]);
        let result = Loss::LogLikelihood.run(&activations, &output);
        assert_close(&result.loss.data, &[0.0], 1e-12);
        assert_close(&result.prime.data, &[0.0, -1.0], 1e-12);
    }

    #[test]
    fn log_likelihood_low_confidence_gives_high_loss() {
        // activation = 0.01 at target → loss = -ln(0.01) ≈ 4.605
        let activations = tensor_with_data(vec![2, 1], &[0.99, 0.01]);
        let output = tensor_with_data(vec![2, 1], &[0.0, 1.0]);
        let result = Loss::LogLikelihood.run(&activations, &output);
        assert_close(&result.loss.data, &[-(0.01_f64.ln())], 1e-12);
        // prime = -1/0.01 = -100 (large gradient pushes hard)
        assert_close(&result.prime.data, &[0.0, -100.0], 1e-12);
    }

    // ── Accuracy tests ──

    #[test]
    fn accuracy_argmax_correct_prediction() {
        // Both have their max at the same flat index
        let activations = tensor_with_data(vec![3], &[0.1, 0.9, 0.2]);
        let output = tensor_with_data(vec![3], &[0.0, 1.0, 0.0]);
        let result = Loss::Accuracy(AccuracyOf::Argmax).run(&activations, &output);
        assert_close(&result.loss.data, &[1.0], 1e-12);
    }

    #[test]
    fn accuracy_argmax_wrong_prediction() {
        // activations max at index 2, output max at index 0
        let activations = tensor_with_data(vec![3], &[0.1, 0.2, 0.9]);
        let output = tensor_with_data(vec![3], &[1.0, 0.0, 0.0]);
        let result = Loss::Accuracy(AccuracyOf::Argmax).run(&activations, &output);
        assert_close(&result.loss.data, &[0.0], 1e-12);
    }

    #[test]
    fn accuracy_argmax_with_negative_values() {
        // Max should still be found correctly with negative values
        let activations = tensor_with_data(vec![3], &[-5.0, -1.0, -3.0]);
        let output = tensor_with_data(vec![3], &[-10.0, -2.0, -20.0]);
        // Both have argmax at index 1
        let result = Loss::Accuracy(AccuracyOf::Argmax).run(&activations, &output);
        assert_close(&result.loss.data, &[1.0], 1e-12);
    }

    #[test]
    fn accuracy_prime_is_always_zero() {
        let activations = tensor_with_data(vec![3], &[0.1, 0.9, 0.2]);
        let output = tensor_with_data(vec![3], &[0.0, 1.0, 0.0]);
        let result = Loss::Accuracy(AccuracyOf::Argmax).run(&activations, &output);
        // As noted in the code, prime is a hacky workaround that returns 0
        assert_close(&result.prime.data, &[0.0], 1e-12);
    }
}

pub struct LossesOn<'a, E> {
    test: &'a Test<E>,
    losses: &'a [Loss],
}

impl<'a, E: Example<Tensor>> LossesOn<'a, E> {
    pub fn new(test: &'a Test<E>, losses: &'a [Loss]) -> Self {
        Self { test, losses }
    }
}

impl<'a, E: Example<Tensor>> Reporting for LossesOn<'a, E> {
    fn report(&self, nn: &NeuralNetwork, epoch: Option<u64>) -> std::io::Result<()> {
        let eval_bar = ProgressBar::new(self.test.len() as u64)
            .with_style(
                ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}]")
                    .unwrap()
                    .progress_chars("=> "),
            )
            .with_prefix(
                format![
                    "eval@epoch {}",
                    epoch
                        .map(|e| (e + 1).to_string())
                        .unwrap_or("init".to_string())
                ]
                .blue()
                .to_string(),
            );
        let mut avg_losses = vec![0.0; self.losses.len()];
        for example in self.test.iter() {
            let activations = nn.test(&example.input());
            for (i, loss) in self.losses.iter().enumerate() {
                let loss_value = loss.run(&activations, &example.output()).loss;
                avg_losses[i] += loss_value.data[0];
            }
            eval_bar.inc(1);
        }
        eval_bar.finish();
        let n = self.test.len() as f64;
        for (i, loss) in self.losses.iter().enumerate() {
            println!(
                "{}: epoch {}: avg. {} = {:.3}",
                "losses_on".purple(),
                epoch
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "init".to_string()),
                match loss {
                    Loss::MSE => "mse",
                    Loss::LogLikelihood => "loglik",
                    Loss::Accuracy(acc) => match acc {
                        AccuracyOf::Argmax => "accuracy(argmax)",
                    },
                },
                avg_losses[i] / n
            );
        }
        Ok(())
    }
}
