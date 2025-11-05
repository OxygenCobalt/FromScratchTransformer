use colored::Colorize;
use crate::{dataset::TestSet, nn::Reporting, tensor::Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    MSE,
    LogLikelihood,
    Accuracy(AccuracyOf)
}

impl Loss {
    pub fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
        match self {
            Loss::MSE => batch_activations.sub(output).unwrap().pow(2).sum(),
            Loss::LogLikelihood => batch_activations.at_argmax(&output).unwrap().ln().neg(),
            Loss::Accuracy(acc) => {
                if acc.accurate(batch_activations, output) {
                    T::scalar(1.0)
                } else {
                    T::scalar(0.0)
                }
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccuracyOf {
    Argmax,
}

impl AccuracyOf {
    pub fn accurate<T: Tensor>(&self, batch_activations: &T, output: &T) -> bool {
        match self {
            AccuracyOf::Argmax => {
                fn flat_argmax<T: Tensor>(tensor: &T) -> usize {
                    tensor
                        .iter()
                        .enumerate()
                        .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap()
                        .0
                }
                let predicted = flat_argmax(batch_activations);
                let actual = flat_argmax(output);
                predicted == actual
            },
        }
    }
}

pub struct LossesOn<'a, S: TestSet>(pub &'a S, pub Vec<Loss>);

impl<'a, T: Tensor, S: TestSet> Reporting<T> for LossesOn<'a, S> {
    fn report(&self, nn: &crate::nn::NeuralNetwork<T>, epoch: Option<u64>) -> std::io::Result<()> {
        let mut avg_losses = vec![0.0; self.1.len()];
        let test = self.0.test();
        for example in test.iter::<T>() {
            let activations = nn.test(&example.input);
            for (i, loss) in self.1.iter().enumerate() {
                let loss_value = loss.loss(&activations, &example.output);
                avg_losses[i] += *loss_value.get(&[]).unwrap()
            }
        }
        let n = test.len() as f64;
        for (i, loss) in self.1.iter().enumerate() {
            println!(
                "{}: epoch {}: avg. {} = {:.3}",
                "losses_on".green(),
                epoch.map(|e| e.to_string()).unwrap_or_else(|| "init".to_string()),
                match loss {
                    Loss::MSE => "mse",
                    Loss::LogLikelihood => "log-lik",
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

// pub trait Loss: Send + Sync {
//     fn tag() -> String;
//     fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T;
// }

// pub struct MSE;

// impl Loss for MSE {
//     fn tag() -> String {
//         "mse".to_string()
//     }

//     fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
//         batch_activations.sub(output).unwrap().pow(2).sum()
//     }
// }

// pub struct LogLikelihood;

// impl Loss for LogLikelihood {
//     fn tag() -> String {
//         "log-lik".to_string()
//     }

//     fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
//         batch_activations.at_argmax(&output).unwrap().ln().neg()
//     }
// }

// pub struct Accuracy<F: AccuracyFunction>(F);

// impl<F: AccuracyFunction> Loss for Accuracy<F> {
//     fn tag() -> String {
//         "accuracy(".to_string() + &F::tag() + ")"
//     }
//     fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
//         if self.0.accuracy(batch_activations, output) {
//             T::scalar(1.0)
//         } else {
//             T::scalar(0.0)
//         }
//     }
// }


// pub struct Argmax;

// impl AccuracyFunction for Argmax {
//     fn tag() -> String {
//         "argmax".to_string()
//     }
    
//     fn accuracy<T: Tensor>(&self, batch_activations: &T, output: &T) -> bool {
//         fn flat_argmax<T: Tensor>(tensor: &T) -> usize {
//             tensor
//                 .iter()
//                 .enumerate()
//                 .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
//                 .unwrap()
//                 .0
//         }
//         let predicted = flat_argmax(batch_activations);
//         let actual = flat_argmax(output);
//         predicted == actual
//     }
// }

// pub trait Losses<L: Loss> {
//     fn 
// }