use std::borrow::Cow;

use crate::{
    dataset::{Example, Test},
    ml::nn2::{NeuralNetwork, Reporting},
    tensor::cpu2::{Tensor, Scalar},
};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    MSE,
    Accuracy(AccuracyOf),
}

pub struct Losses<'a> {
    pub loss: Tensor<'a>,
    pub prime: Tensor<'a>
}

impl Loss {
    pub fn run<'o>(&self, batch_activations: &Tensor, output: &Tensor) -> Losses<'o> {
        match self {
            Loss::MSE => {
                let loss_data: Vec<f64> = batch_activations.data.iter().zip(output.data.iter()).map(|(a, o)| (a - o).powi(2)).collect();
                let prime_data: Vec<f64> = batch_activations.data.iter().zip(output.data.iter()).zip(&loss_data).map(|((a, o), l)| 2.0 * (a - o)).collect();
                let loss = Tensor { shape: batch_activations.shape.clone(), stride: batch_activations.stride.clone(), data: Cow::Owned(loss_data) };
                let prime: Tensor<'_> = Tensor { shape: batch_activations.shape.clone(), stride: batch_activations.stride.clone(), data: Cow::Owned(prime_data) };
                Losses {
                    loss,
                    prime
                }
            },
            Loss::Accuracy(of) => {
                let loss = if of.accurate(batch_activations, output) {
                    Tensor::init(Scalar(1.0))
                } else {
                    Tensor::init(Scalar(0.0))
                }.unwrap();
                Losses {
                    loss,
                    // todo: hacky workaround fixfixfix
                    prime: Tensor::init(Scalar(0.0)).unwrap()
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
                    tensor.data
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

pub struct LossesOn<'a, E> {
    test: &'a Test<E>,
    losses: &'a [Loss],
}

impl<'a, E: Example<crate::tensor::cpu::CPUTensor>> LossesOn<'a, E> {
    pub fn new(test: &'a Test<E>, losses: &'a [Loss]) -> Self {
        Self {
            test,
            losses,
        }
    }
}

impl<'a, E: Example<crate::tensor::cpu::CPUTensor>> Reporting for LossesOn<'a, E> {
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
            let activations = nn.test(&example.input().to_cpu2());
            for (i, loss) in self.losses.iter().enumerate() {
                let loss_value = loss.run(&activations, &example.output().to_cpu2()).loss;
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
