use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::io::{self, Read, Write};
use rand::seq::SliceRandom;

use crate::tensor::{Generate, SharpTensor, Tensor, TensorIO, Th};

use super::{
    activation::Activation,
    autograd::Autograd,
    loss::Loss,
};

pub struct NeuralNetwork<T: Tensor> {
    axons: Vec<Axon<T>>,
}

impl <T: Tensor> NeuralNetwork<T> {
    pub fn new(layers: &[Layer]) -> Self {
        if layers.len() < 2 {
            panic!("not enough layers");
        }
        let mut axons = vec![];
        for i in 0..(layers.len() - 1) {
            axons.push(layers[i].axon(&layers[i + 1]));
        }

        Self { axons }
    }

    pub fn test(
        &self,
        test: &Test<T>,
        loss: &impl Loss,
    ) -> TestResults<T> {
        let mut sum = 0.0;
        let mut results = Vec::new();
        for example in &test.examples {
            let activations = self.evaluate(&example.input);
            sum += loss.loss(&activations, &example.output).get(&[]).unwrap();
            results.push(TestResult { example: example.clone(), activations });
        }
        let loss = sum / test.examples.len() as f64;
        return TestResults {
            avg_loss: loss,
            results,
        };
    }

    pub fn evaluate(&self, input: &T) -> T {
        let mut current = input.clone();
        for axon in &self.axons {
            current = axon.forward(&current)
        }
        current
    }
}

impl <T: SharpTensor> NeuralNetwork<T> {
    pub fn train(
        &mut self,
        train: &impl TrainingSetup<T>,
        hyperparams: Hyperparams,
        loss: &impl Loss,
    ) -> io::Result<()> {
        println!("{}: epochs = {} / batch size = {} / learning rate = {}", 
            "train".cyan(),
            hyperparams.epochs,
            hyperparams.batch_size,
            hyperparams.learning_rate);
        train.report(None, loss, self)?;
        for epoch in 0..hyperparams.epochs {
            let batches = train.train().batch(hyperparams.batch_size);
            let sgd_bar = ProgressBar::new(batches.len() as u64)
                .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] / avg batch loss = {msg}")
                                .unwrap()
                                .progress_chars("=> "))
                .with_prefix(format!["epoch {}", epoch + 1].blue().to_string());
            let mut total_loss = 0.0;
            for (i, batch) in batches.into_iter().enumerate() {
                let auto_axons: Vec<Axon<Autograd<T>>> =
                    self.axons.iter_mut().map(|a| a.train()).collect();
                let mut current = Autograd::new(batch.input);
                for axon in &auto_axons {
                    current = axon.forward(&current)
                }
                let loss = loss.loss(&current, &Autograd::new(batch.output));
                total_loss += loss.get(&[]).unwrap();
                loss.backward();
                // this drops the entire computation graph allowing us to move the weight/bias
                // gradients out w/o a clone
                std::mem::drop(loss);
                std::mem::drop(current);

                let c = hyperparams.learning_rate / hyperparams.batch_size as f64;
                for (axon, auto_axon) in self.axons.iter_mut().zip(auto_axons.into_iter()) {
                    axon.commit(auto_axon, c);
                }
                sgd_bar.inc(1);
                sgd_bar.set_message(format!["{:.3}", total_loss / (i + 1) as f64]);
            }
            sgd_bar.finish();
            train.report(Some(epoch), loss, self)?;
        }
        Ok(())
    }
}

impl <T: TensorIO> NeuralNetwork<T> {
    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read(&mut signature)?;
        if &signature != b"NeuralNt" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid neuralnet signature",
            ));
        }
        let mut nb = [0u8; 8];
        read.read(&mut nb)?;
        let axon_count = usize::from_le_bytes(nb);
        let mut axons = Vec::with_capacity(axon_count);
        for _ in 0..axon_count {
            axons.push(Axon::read(read)?)
        }
        Ok(Self { axons })
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write(b"NeuralNt")?;
        write.write_all(&self.axons.len().to_le_bytes())?;
        for axon in &self.axons {
            axon.write(write)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Example<T: Tensor> {
    pub input: T,
    pub output: T,
}

#[derive(Clone)]
pub struct Train<T: Tensor> {
    pub examples: Vec<Example<T>>,
}

impl <T: Tensor> Train<T> {
    fn batch(&self, size: usize) -> Vec<Example<T>> {
        let mut examples = self.examples.clone();
        examples.shuffle(&mut rand::rng());
        self.examples
            .chunks(size)
            .map(|examples| {
                let input = T::tensor(Th::C(
                    examples
                        .iter()
                        .map(|e| Th::R(e.input.iter().cloned().collect()))
                        .collect::<Vec<Th>>(),
                ))
                .unwrap();
                let output = T::tensor(Th::C(
                    examples
                        .iter()
                        .map(|e| Th::R(e.output.iter().cloned().collect()))
                        .collect::<Vec<Th>>(),
                ))
                .unwrap();
                Example { input, output }
            })
            .collect()
    }
}

pub trait TrainingSetup<T: Tensor> {
    fn train(&self) -> &Train<T>;
    fn report(&self, epoch: Option<u64>, loss: &impl Loss, nn: &NeuralNetwork<T>) -> io::Result<()>;
}

pub struct NoReporting<'a, T: Tensor>(pub &'a Train<T>);

impl <'a, T: Tensor> TrainingSetup<T> for NoReporting<'a, T> {
    fn train(&self) -> &Train<T> {
        &self.0
    }

    fn report(&self, _: Option<u64>, _: &impl Loss, _: &NeuralNetwork<T>) -> io::Result<()> {
        Ok(())
    }
}

pub struct Checkpoint<'a, T: TensorIO, R: TrainingSetup<T>>(pub &'a R, pub PathBuf, pub PhantomData<T>);

impl <'a,  T: TensorIO, R: TrainingSetup<T>> TrainingSetup<T> for Checkpoint<'a, T, R> {
    fn train(&self) -> &Train<T> {
        self.0.train()
    }

    fn report(&self, epoch: Option<u64>, loss: &impl Loss, nn: &NeuralNetwork<T>) -> io::Result<()> {
        self.0.report(epoch,  loss, nn)?;
        let path = self.1.join(Path::new(&format!["{}.nn", epoch.map(|e| (e + 1).to_string()).unwrap_or_else(|| "init".to_string())]));
        println!("{}: writing nn to {}", "checkpoint".red(), path.display().to_string());
        let mut file = File::create(path)?;
        nn.write(&mut file)?;
        Ok(())
    }
}

pub struct Hyperparams {
    pub epochs: u64,
    pub batch_size: usize,
    pub learning_rate: f64
}

pub struct Test<T: Tensor> {
    pub examples: Vec<Example<T>>,
}

pub struct TestResults<T: Tensor> {
    pub avg_loss: f64,
    pub results: Vec<TestResult<T>>,
}

pub struct TestResult<T: Tensor> {
    pub example: Example<T>,
    pub activations: T
}

pub static NORMAL: LazyLock<Normal<f64>> =
    std::sync::LazyLock::new(|| Normal::new(0.0, 1.0).unwrap());

pub enum Layer {
    Dense {
        neurons: usize,
        activation: Activation,
    },
    Dropout {
        neurons: usize,
        rate: f64,
        activation: Activation,
    },
}

impl Layer {
    fn axon<T: Tensor>(&self, last: &Layer) -> Axon<T> {
        match self {
            Self::Dense {
                neurons,
                activation,
            } => Axon::Dense {
                ff: FeedForward::new(last.activation_shape()[0], *neurons, *activation),
            },
            Self::Dropout {
                neurons,
                rate,
                activation,
            } => Axon::Dropout {
                ff: FeedForward::new(last.activation_shape()[0], *neurons, *activation),
                rate: *rate,
            },
        }
    }

    fn activation_shape(&self) -> Vec<usize> {
        match self {
            Self::Dense { neurons, .. } => vec![*neurons],
            Self::Dropout { neurons, .. } => vec![*neurons],
        }
    }
}

struct FeedForward<T: Tensor> {
    weights: T,
    biases: T,
    activation: Activation,
}

impl<T: Tensor> FeedForward<T> {
    fn new(last: usize, next: usize, activation: Activation) -> Self {
        Self {
            weights: T::tensor(Generate {
                shape: vec![last, next],
                with: || NORMAL.sample(&mut rand::rng()),
            })
            .unwrap(),
            biases: T::tensor(Generate {
                shape: vec![last],
                with: || NORMAL.sample(&mut rand::rng()),
            })
            .unwrap(),
            activation,
        }
    }

    fn forward(&self, activations: &T) -> T {
        self.activation.activate(
            self.weights
                .dot(&activations, 1)
                .unwrap()
                .add(&self.biases)
                .unwrap(),
        )
    }
}

impl<T: TensorIO> FeedForward<T> {
    fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read(&mut signature)?;
        if &signature != b"FeedFrwd" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid feedforward signature",
            ));
        }
        let mut activation_id = [0u8; 8];
        read.read(&mut activation_id)?;
        let activation = Activation::from_id(&activation_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "invalid feedforward signature"))?;
        let weights = T::read(read)?;
        let biases = T::read(read)?;
        Ok(Self {
            activation,
            weights,
            biases,
        })
    }

    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"FeedFrwd")?;
        write.write_all(self.activation.id())?;
        self.weights.write(write)?;
        self.biases.write(write)
    }
}

enum Axon<T: Tensor> {
    Dense { ff: FeedForward<T> },
    Dropout { ff: FeedForward<T>, rate: f64 },
}

impl<T: Tensor> Axon<T> {
    pub fn forward(&self, activations: &T) -> T {
        match self {
            Self::Dense { ff } => ff.forward(activations),
            Self::Dropout { ff, .. } => ff.forward(activations),
        }
    }
}

impl<T: SharpTensor> Axon<T> {
    fn train<'a>(&'a mut self) -> Axon<Autograd<T>> {
        match self {
            Self::Dense { ff } => Axon::Dense {
                ff: FeedForward {
                    weights: Autograd::new(ff.weights.clone()),
                    biases: Autograd::new(ff.biases.clone()),
                    activation: ff.activation,
                },
            },
            Self::Dropout { ff, rate } => {
                let mut dropped_weights = ff.weights.clone();
                for row in 0..dropped_weights.shape()[1] {
                    if rand::random_range(0.0..1.0) > *rate {
                        continue;
                    }
                    for col in 0..dropped_weights.shape()[0] {
                        *dropped_weights.get_mut(&[col, row]).unwrap() = 0.0;
                    }
                }
                Axon::Dense {
                    ff: FeedForward {
                        weights: Autograd::new(dropped_weights),
                        biases: Autograd::new(ff.biases.clone()),
                        activation: ff.activation,
                    },
                }
            }
        }
    }

    pub fn commit(&mut self, axon: Axon<Autograd<T>>, c: f64) -> Option<()> {
        let scale = T::scalar(c);
        match (self, axon) {
            (Self::Dense { ff }, Axon::<Autograd<T>>::Dense { ff: autoff }) => {
                ff.weights = ff
                    .weights
                    .sub(&autoff.weights.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
                ff.biases = ff
                    .biases
                    .sub(&autoff.biases.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
            }
            (Self::Dropout { ff, .. }, Axon::<Autograd<T>>::Dropout { ff: autoff, .. }) => {
                ff.weights = ff
                    .weights
                    .sub(&autoff.weights.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
                ff.biases = ff
                    .biases
                    .sub(&autoff.biases.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
            }
            _ => return None,
        }
        Some(())
    }
}

impl<T: TensorIO> Axon<T> {
    fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut id = [0u8; 8];
        read.read(&mut id)?;
        match &id {
            b"AxonDnse" => Ok(Self::Dense {
                ff: FeedForward::read(read)?,
            }),
            b"AxonDrop" => {
                let mut rateb = [0u8; 8];
                read.read(&mut rateb)?;
                Ok(Self::Dropout {
                    ff: FeedForward::read(read)?,
                    rate: f64::from_le_bytes(rateb),
                })
            }
            _ => Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid axon signature",
            )),
        }
    }

    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Dense { ff } => {
                write.write_all(b"AxonDnse")?;
                ff.write(write)
            }
            Self::Dropout { ff, rate } => {
                write.write_all(b"AxonDrop")?;
                write.write_all(&rate.to_le_bytes())?;
                ff.write(write)
            }
        }
    }
}
