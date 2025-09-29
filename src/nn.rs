use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::io::{self, Read, Write};
use rand::seq::SliceRandom;

use crate::tensor::{Field, Generate, Tensor, TensorIO, TensorMut, Th};

use super::{
    activation::Activation,
    autograd::Autograd,
    loss::Loss,
};

pub struct NeuralNetwork<T: Tensor> {
    axons: Vec<Axon<T>>
}

impl <T: Tensor> NeuralNetwork<T> {
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

impl <T: TensorMut> NeuralNetwork<T> {
    pub fn train(
        setup: &impl Setup<T>,
        reporting: &impl Reporting<T>,
        training_set: &impl TrainingSet<T>,
        hyperparams: &Hyperparams,
        loss: &impl Loss,
    ) -> io::Result<Self> {
        println!("{}: epochs = {} / batch size = {} / learning rate = {}", 
            "train".cyan(),
            hyperparams.epochs,
            hyperparams.batch_size,
            hyperparams.learning_rate);
        let mut init = setup.setup()?;
        let start = init.at_epoch.unwrap_or(0);
        if start > hyperparams.epochs {
            return Err(io::Error::new(io::ErrorKind::Other, "invalid setup, epoch exceeds training parameters"))
        }
        println!("{}: starting w/nn @ {}", "train".cyan(), init.at_epoch.map(|e| format!["epoch {}", e + 1]).unwrap_or_else(|| "init".to_string()));
        reporting.report(&init.nn, loss, None)?;
        for epoch in init.at_epoch.unwrap_or(0)..hyperparams.epochs {
            let batches = training_set.get()?.batch(hyperparams.batch_size);
            let sgd_bar = ProgressBar::new(batches.len() as u64)
                .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] / avg batch loss = {msg}")
                                .unwrap()
                                .progress_chars("=> "))
                .with_prefix(format!["epoch {}", epoch + 1].blue().to_string());
            let mut total_loss = 0.0;
            for (i, batch) in batches.into_iter().enumerate() {
                let auto_axons: Vec<Axon<Autograd<T>>> =
                    init.nn.axons.iter_mut().map(|a| a.train()).collect();
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
                for (axon, auto_axon) in init.nn.axons.iter_mut().zip(auto_axons.into_iter()) {
                    axon.commit(auto_axon, c);
                }
                sgd_bar.inc(1);
                sgd_bar.set_message(format!["{:.3}", total_loss / (i + 1) as f64]);
            }
            sgd_bar.finish();
            reporting.report(&init.nn, loss, Some(epoch))?;
        }
        Ok(init.nn)
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

pub trait Setup<T: Tensor> {
    fn setup(&self) -> io::Result<Init<T>>;
}

pub trait Reporting<T: Tensor> {
    fn report(&self, nn: &NeuralNetwork<T>, loss: &impl Loss, epoch: Option<u64>) -> io::Result<()>;
}

pub trait TrainingSet<T: Tensor> {
    fn get(&self) -> io::Result<Train<T>>;
}

pub struct Init<T: Tensor> {
    nn: NeuralNetwork<T>,
    at_epoch: Option<u64>
}


pub struct Layers(Vec<Layer>);

impl Layers {
    pub fn new(layers: Vec<Layer>) -> Option<Self> {
        if layers.len() < 2 {
            return None;
        }
        Some(Self(layers))
    }
}

impl <T: Tensor> Setup<T> for Layers {
    fn setup(&self) -> io::Result<Init<T>> {
        let mut axons = vec![];
        for i in 0..(self.0.len() - 1) {
            axons.push(self.0[i].axon(&self.0[i + 1]));
        }
        Ok(Init {
            nn: NeuralNetwork { axons },
            at_epoch: None
        })
    }
}

pub struct Checkpoint<'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> {
    setup: &'a S,
    reporting: &'a R,
    path: &'a Path,
    data: PhantomData<T>
}

impl <'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> Checkpoint<'a, T, S, R> {
    pub fn new(setup: &'a S, reporting: &'a R, path: &'a Path) -> Self {
        Self {
            setup,
            reporting,
            path,
            data: PhantomData
        }
    }

    fn checkpoint_path(&self, epoch: Option<u64>) -> PathBuf {
        self.path.join(Path::new(&format!["{}.nn", epoch.map(|e| (e + 1).to_string()).unwrap_or_else(|| "init".to_string())]))
    }
}

impl <'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> Setup<T> for Checkpoint<'a, T, S, R> {
    fn setup(&self) -> io::Result<Init<T>> {
        fn open<T: TensorIO>(path: PathBuf) -> io::Result<NeuralNetwork<T>> {
            let mut file = File::open(path)?;
            let nn = NeuralNetwork::read(&mut file)?;
            Ok(nn)
        }
        let amount = self.path.read_dir()?.count();
        for epoch in (0..amount).map(|i| Some(i as u64)).rev().chain(std::iter::once(None)) {
            let path = self.checkpoint_path(epoch);
            if let Ok(nn) = open(path) {
                println!("{}: located checkpointed nn at epoch {}", "checkpoint".red(), epoch.map(|e| (e + 1).to_string()).unwrap_or_else(|| "init".to_string()));
                return Ok(Init {
                    nn,
                    at_epoch: epoch
                })
            }
        }
        self.setup.setup()
    }
}

impl <'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> Reporting<T> for Checkpoint<'a, T, S, R> {
    fn report(&self, nn: &NeuralNetwork<T>, loss: &impl Loss, epoch: Option<u64>) -> io::Result<()> {
        self.reporting.report(nn, loss, epoch)?;
        let path = self.checkpoint_path(epoch);
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

static NORMAL: LazyLock<Normal<f64>> =
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
    Conv2D {
        input_size: usize,
        field: Field,
        filters: usize,
        activation: Activation
    }
}
pub enum Pooling {
    Max
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
            Self::Conv2D { field, filters, activation, .. } => {
                Axon::Conv2D { conv: Conv2D::new(*field, *filters, *activation) }
            }
        }
    }

    fn activation_shape(&self) -> Vec<usize> {
        match self {
            Self::Dense { neurons, .. } => vec![*neurons],
            Self::Dropout { neurons, .. } => vec![*neurons],
            // todo: validate locations_on during layer creation, also stop calling it twice
            Self::Conv2D  { input_size, field, filters, .. } => vec![*filters, field.locations_on(*input_size).unwrap(), field.locations_on(*input_size).unwrap()]
        }
    }
}

enum Axon<T: Tensor> {
    Dense { ff: FeedForward<T> },
    Dropout { ff: FeedForward<T>, rate: f64 },
    Conv2D { conv: Conv2D<T> },

}

impl<T: Tensor> Axon<T> {
    pub fn forward(&self, activations: &T) -> T {
        match self {
            Self::Dense { ff } => ff.forward(activations),
            Self::Dropout { ff, .. } => ff.forward(activations),
            Self::Conv2D { conv } => conv.forward(activations)
        }
    }
}

impl<T: TensorMut> Axon<T> {
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
            },
            Self::Conv2D { conv } => {
                Axon::Conv2D { conv: Conv2D { filters: Autograd::new(conv.filters.clone()), field: conv.field, activation: conv.activation } }
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
            },
            Self::Conv2D { conv } => {
                todo!()
            }
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

struct Conv2D<T: Tensor> {
    filters: T,
    field: Field,
    activation: Activation
}

impl <T: Tensor> Conv2D<T> {
    fn new(field: Field, filters: usize, activation: Activation) -> Self {
        // dont have anything to make column-wise tensors just yet, instead hack around
        // w/a transpose lol
        Self {
            filters: T::tensor(Generate { shape: vec![field.size, field.size, filters], with: || NORMAL.sample(&mut rand::rng()) }).unwrap(),
            field,
            activation
        }
    }
    fn forward(&self, activations: &T) -> T {
        activations.conv2d(&self.filters, self.field.stride).unwrap()
    }
}