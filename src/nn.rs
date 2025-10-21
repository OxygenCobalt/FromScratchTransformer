use atomic_float::AtomicF64;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::fs::File;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::LazyLock;

use crate::tensor::{Autograd, CPUTensor, DifferentiableTensor, Field, Fill, Generate, Tensor, TensorIO, TensorMut, Tt};

use super::{activation::Activation, loss::Loss};

pub struct NeuralNetwork<T: Tensor> {
    axons: Vec<Axon<T>>,
}

impl<T: Tensor> NeuralNetwork<T> {
    pub fn test(&self, test: &Test<T>, loss: &impl Loss) -> TestResults<T> {
        let mut sum = 0.0;
        let mut results = Vec::new();
        for example in &test.examples {
            let activations = self.evaluate(&example.input);
            sum += loss.loss(&activations, &example.output).get(&[]).unwrap();
            results.push(TestResult {
                example: example.clone(),
                activations,
            });
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
            current = axon.forward(current)
        }
        current
    }
}

impl<T: TensorMut + DifferentiableTensor> NeuralNetwork<T> {
    pub fn train(
        setup: &impl Setup<T>,
        reporting: &impl Reporting<T>,
        training_set: &impl TrainingSet<T>,
        hyperparams: &Hyperparams,
        loss: &impl Loss,
    ) -> io::Result<Self> {
        println!(
            "{}: epochs = {} / batch size = {} / learning rate = {}",
            "train".cyan(),
            hyperparams.epochs,
            hyperparams.batch_size,
            hyperparams.learning_rate
        );
        let mut init = setup.setup()?;
        let start = init.at_epoch.unwrap_or(0);
        if start > hyperparams.epochs {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid setup, epoch exceeds training parameters",
            ));
        }
        println!(
            "{}: starting w/nn @ {}",
            "train".cyan(),
            init.at_epoch
                .map(|e| format!["epoch {}", e + 1])
                .unwrap_or_else(|| "init".to_string())
        );
        if let None = init.at_epoch {
            reporting.report(&init.nn, loss, None)?;
        }
        for epoch in init.at_epoch.unwrap_or(0)..hyperparams.epochs {
            let batches = training_set.get()?.batch(hyperparams.batch_size);
            let sgd_bar = ProgressBar::new(batches.len() as u64)
                .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] / avg batch loss = {msg}")
                                .unwrap()
                                .progress_chars("=> "))
                .with_prefix(format!["epoch {}", epoch + 1].blue().to_string());
            let mut total_loss = 0.0;
            for (i, batch) in batches.into_iter().enumerate() {
                let auto_axons: Vec<Axon<T::Autograd>> =
                    init.nn.axons.iter_mut().map(|a| a.train()).collect();
                let mut current = batch.input.autograd();
                for axon in &auto_axons {
                    current = axon.forward(current)
                }
                let loss = loss.loss(&current, &batch.output.autograd());
                total_loss += loss.iter().sum::<f64>() / *loss.shape().first().unwrap_or(&1) as f64;
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

impl<T: TensorMut + DifferentiableTensor + Send + Sync> NeuralNetwork<T>
where
    T::Autograd: Send,
{
    pub fn par_train(
        setup: &impl Setup<T>,
        reporting: &impl Reporting<T>,
        training_set: &impl TrainingSet<T>,
        hyperparams: &Hyperparams,
        loss: &(impl Loss + Sync),
    ) -> io::Result<Self> {
        println!(
            "{}: epochs = {} / batch size = {} / learning rate = {}",
            "train".cyan(),
            hyperparams.epochs,
            hyperparams.batch_size,
            hyperparams.learning_rate
        );
        let mut init = setup.setup()?;
        let start = init.at_epoch.unwrap_or(0);
        if start > hyperparams.epochs {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid setup, epoch exceeds training parameters",
            ));
        }
        println!(
            "{}: starting w/nn @ {}",
            "train".cyan(),
            init.at_epoch
                .map(|e| format!["epoch {}", e + 1])
                .unwrap_or_else(|| "init".to_string())
        );
        if let None = init.at_epoch {
            reporting.report(&init.nn, loss, None)?;
        }
        for epoch in init.at_epoch.unwrap_or(0)..hyperparams.epochs {
            let training_set = training_set.get()?;
            let batches = training_set.par_batch(hyperparams.batch_size);
            let sgd_bar = ProgressBar::new(batches.len() as u64)
                .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] / avg batch loss = {msg}")
                                .unwrap()
                                .progress_chars("=> "))
                .with_prefix(format!["epoch {}", epoch + 1].blue().to_string());
            let total_loss = AtomicF64::new(0.0);
            for (i, batch) in batches.into_iter().enumerate() {
                let c = hyperparams.learning_rate / hyperparams.batch_size as f64;
                let all_auto_axons: Vec<_> = batch.map(|example| {
                    let auto_axons: Vec<Axon<T::Autograd>> =
                        init.nn.axons.iter().map(|a| a.train()).collect();
                    let mut current = example.input.autograd();
                    for axon in &auto_axons {
                        current = axon.forward(current);
                    }
                    let loss = loss.loss(&current, &example.output.autograd());
                    total_loss.fetch_add(loss.iter().sum::<f64>() / *loss.shape().first().unwrap_or(&1) as f64, Ordering::Relaxed);
                    loss.backward();
                    // this drops the entire computation graph allowing us to move the weight/bias
                    // gradients out w/o a clone
                    std::mem::drop(loss);
                    std::mem::drop(current);
                    auto_axons
                }).collect();

                for auto_axon in all_auto_axons {
                    for (axon, auto_axon) in init.nn.axons.iter_mut().zip(auto_axon.into_iter()) {
                        axon.commit(auto_axon, c);
                    }
                }
                sgd_bar.inc(1);
                sgd_bar.set_message(format!["{:.3}", total_loss.load(Ordering::Relaxed) / (i + 1) as f64]);
            }
            sgd_bar.finish();
            reporting.report(&init.nn, loss, Some(epoch))?;
        }
        Ok(init.nn)
    }
}

impl<T: TensorIO> NeuralNetwork<T> {
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

impl<T: Tensor> Train<T> {
    fn batch(&self, size: usize) -> Vec<Example<T>> {
        let mut examples = self.examples.clone();
        examples.shuffle(&mut rand::rng());
        examples
            .chunks(size)
            .map(|examples| {
                let input = T::tensor(Tt(examples
                    .iter()
                    .map(|e| e.input.clone())
                    .collect::<Vec<T>>()))
                .unwrap();
                let output = T::tensor(Tt(examples
                    .iter()
                    .map(|e| e.output.clone())
                    .collect::<Vec<T>>()))
                .unwrap();
                Example { input, output }
            })
            .collect()
    }
}

impl <T: Tensor + Send> Train<T> {
    fn par_batch(&self, size: usize) -> Vec<impl ParallelIterator<Item=Example<T>>> {
        let mut examples = self.examples.clone();
        examples.shuffle(&mut rand::rng());
        examples.chunks(size).map(|chunk| {
            chunk.to_vec().into_par_iter()
        }).collect()
    }
}

pub trait Setup<T: Tensor> {
    fn setup(&self) -> io::Result<Init<T>>;
}

pub trait Reporting<T: Tensor> {
    fn report(&self, nn: &NeuralNetwork<T>, loss: &impl Loss, epoch: Option<u64>)
    -> io::Result<()>;
}

pub trait TrainingSet<T: Tensor> {
    fn get(&self) -> io::Result<Train<T>>;
}

pub struct Init<T: Tensor> {
    nn: NeuralNetwork<T>,
    at_epoch: Option<u64>,
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

impl<T: Tensor> Setup<T> for Layers {
    fn setup(&self) -> io::Result<Init<T>> {
        let mut axons = vec![];
        for i in 0..self.0.len() {
            axons.push(self.0[i].axon(if i > 0 { Some(&self.0[i - 1]) } else { None }));
        }
        Ok(Init {
            nn: NeuralNetwork { axons },
            at_epoch: None,
        })
    }
}

pub struct Checkpoint<'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> {
    setup: &'a S,
    reporting: &'a R,
    path: &'a Path,
    data: PhantomData<T>,
}

impl<'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> Checkpoint<'a, T, S, R> {
    pub fn new(setup: &'a S, reporting: &'a R, path: &'a Path) -> Self {
        Self {
            setup,
            reporting,
            path,
            data: PhantomData,
        }
    }

    fn checkpoint_path(&self, epoch: Option<u64>) -> PathBuf {
        self.path.join(Path::new(&format![
            "{}.nn",
            epoch
                .map(|e| (e + 1).to_string())
                .unwrap_or_else(|| "init".to_string())
        ]))
    }
}

impl<'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> Setup<T> for Checkpoint<'a, T, S, R> {
    fn setup(&self) -> io::Result<Init<T>> {
        fn open<T: TensorIO>(path: PathBuf) -> io::Result<NeuralNetwork<T>> {
            let mut file = File::open(path)?;
            let nn = NeuralNetwork::read(&mut file)?;
            Ok(nn)
        }
        let amount = self.path.read_dir()?.count();
        for epoch in (0..amount)
            .map(|i| Some(i as u64))
            .rev()
            .chain(std::iter::once(None))
        {
            let path = self.checkpoint_path(epoch);
            if let Ok(nn) = open(path) {
                println!(
                    "{}: located checkpointed nn at epoch {}",
                    "checkpoint".red(),
                    epoch
                        .map(|e| (e + 1).to_string())
                        .unwrap_or_else(|| "init".to_string())
                );
                return Ok(Init {
                    nn,
                    at_epoch: epoch,
                });
            }
        }
        self.setup.setup()
    }
}

impl<'a, T: TensorIO, S: Setup<T>, R: Reporting<T>> Reporting<T> for Checkpoint<'a, T, S, R> {
    fn report(
        &self,
        nn: &NeuralNetwork<T>,
        loss: &impl Loss,
        epoch: Option<u64>,
    ) -> io::Result<()> {
        self.reporting.report(nn, loss, epoch)?;
        let path = self.checkpoint_path(epoch);
        println!(
            "{}: writing nn to {}",
            "checkpoint".red(),
            path.display().to_string()
        );
        let mut file = File::create(path)?;
        nn.write(&mut file)?;
        Ok(())
    }
}

pub struct Hyperparams {
    pub epochs: u64,
    pub batch_size: usize,
    pub learning_rate: f64,
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
    pub activations: T,
}

pub enum Layer {
    Dense {
        input_shape: Option<Vec<usize>>,
        neurons: usize,
        activation: Activation,
    },
    Dropout {
        input_shape: Option<Vec<usize>>,
        neurons: usize,
        rate: f64,
        activation: Activation,
    },
    Conv2D {
        input_size: usize,
        field: Field,
        filters: usize,
        activation: Activation,
    },
    Pool2D {
        input_size: usize,
        field: Field,
        filters: usize,
    },
}

impl Layer {
    fn axon<T: Tensor>(&self, last: Option<&Layer>) -> Axon<T> {
        match self {
            Self::Dense {
                input_shape,
                neurons,
                activation,
            } => Axon::Dense {
                ff: FeedForward::new(last.map(|l| l.activation_shape()).or(input_shape.clone()).unwrap(), *neurons, *activation),
            },
            Self::Dropout {
                input_shape,
                neurons,
                rate,
                activation,
            } => Axon::Dropout {
                ff: FeedForward::new(last.map(|l| l.activation_shape()).or(input_shape.clone()).unwrap(), *neurons, *activation),
                rate: *rate,
            },
            Self::Conv2D {
                field,
                filters,
                activation,
                input_size
            } => Axon::Conv2D {
                conv: Conv2D::new(*field, *filters, *activation, *input_size),
            },
            Self::Pool2D { field, .. } => Axon::Pool2D {
                pool: Pool2D::new(*field),
            },
        }
    }

    fn activation_shape(&self) -> Vec<usize> {
        match self {
            Self::Dense { neurons, .. } => vec![*neurons],
            Self::Dropout { neurons, .. } => vec![*neurons],
            // todo: validate locations_on during layer creation, also stop calling it twice
            Self::Conv2D {
                input_size,
                field,
                filters,
                ..
            } => vec![
                *filters,
                field.locations_on(*input_size).unwrap(),
                field.locations_on(*input_size).unwrap(),
            ],
            Self::Pool2D {
                input_size,
                field,
                filters,
            } => {
                vec![
                    *filters,
                    field.locations_on(*input_size).unwrap(),
                    field.locations_on(*input_size).unwrap(),
                ]
            }
        }
    }
}

enum Axon<T: Tensor> {
    Dense { ff: FeedForward<T> },
    Dropout { ff: FeedForward<T>, rate: f64 },
    Conv2D { conv: Conv2D<T> },
    Pool2D { pool: Pool2D<T> },
}

impl<T: Tensor> Axon<T> {
    pub fn forward(&self, activations: T) -> T {
        match self {
            Self::Dense { ff } => ff.forward(activations),
            Self::Dropout { ff, .. } => ff.forward(activations),
            Self::Conv2D { conv } => conv.forward(&activations),
            Self::Pool2D { pool } => pool.forward(activations),
        }
    }
}

impl<T: DifferentiableTensor + TensorMut> Axon<T> {
    fn train<'a>(&self) -> Axon<T::Autograd> {
        match self {
            Self::Dense { ff } => Axon::Dense {
                ff: FeedForward {
                    weights: ff.weights.clone().autograd(),
                    biases: ff.biases.clone().autograd(),
                    activation: ff.activation,
                    flattened_input_ndim: ff.flattened_input_ndim,
                    flattened_input_shape: ff.flattened_input_shape,
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
                        weights: dropped_weights.autograd(),
                        biases: ff.biases.clone().autograd(),
                        activation: ff.activation,
                        flattened_input_ndim: ff.flattened_input_ndim,
                        flattened_input_shape: ff.flattened_input_shape,
                    },
                }
            }
            Self::Conv2D { conv } => Axon::Conv2D {
                conv: Conv2D {
                    weights: conv.weights.clone().autograd(),
                    biases: conv.biases.clone().autograd(),
                    field: conv.field,
                    activation: conv.activation,
                },
            },
            Self::Pool2D { pool } => Axon::Pool2D {
                pool: Pool2D {
                    field: pool.field,
                    phantom: PhantomData,
                },
            },
        }
    }

    pub fn commit(&mut self, axon: Axon<T::Autograd>, c: f64) -> Option<()> {
        let scale = T::scalar(c);
        match (self, axon) {
            (Self::Dense { ff }, Axon::<T::Autograd>::Dense { ff: autoff }) => {
                ff.weights = ff
                    .weights
                    .sub(&autoff.weights.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
                ff.biases = ff
                    .biases
                    .sub(&autoff.biases.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
            }
            (Self::Dropout { ff, .. }, Axon::<T::Autograd>::Dropout { ff: autoff, .. }) => {
                ff.weights = ff
                    .weights
                    .sub(&autoff.weights.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
                ff.biases = ff
                    .biases
                    .sub(&autoff.biases.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
            }
            (Self::Conv2D { conv }, Axon::<T::Autograd>::Conv2D { conv: autoconv }) => {
                conv.weights = conv
                    .weights
                    .sub(&autoconv.weights.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
                conv.biases = conv
                    .biases
                    .sub(&autoconv.biases.into_grad().unwrap().mul(&scale).unwrap())
                    .unwrap();
            },
            (Self::Pool2D { pool }, Axon::<T::Autograd>::Pool2D { pool: autopool }) => {

            }
            _ => return None,
        }
        Some(())
    }
}

impl<T: TensorIO> Axon<T> {
    fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut id = [0u8; 8];
        read.read_exact(&mut id)?;
        match &id {
            b"AxonDnse" => Ok(Self::Dense {
                ff: FeedForward::read(read)?,
            }),
            b"AxonDrop" => {
                let mut rateb = [0u8; 8];
                read.read_exact(&mut rateb)?;
                Ok(Self::Dropout {
                    ff: FeedForward::read(read)?,
                    rate: f64::from_le_bytes(rateb),
                })
            }
            b"AxonCv2D" => Ok(Self::Conv2D {
                conv: Conv2D::read(read)?,
            }),
            b"AxonPl2D" => Ok(Self::Pool2D {
                pool: Pool2D::read(read)?,
            }),
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
            Self::Conv2D { conv } => {
                write.write_all(b"AxonCv2D")?;
                conv.write(write)
            }
            Self::Pool2D { pool } => {
                write.write_all(b"AxonPl2D")?;
                pool.write(write)
            }
        }
    }
}

struct FeedForward<T: Tensor> {
    weights: T,
    biases: T,
    activation: Activation,
    flattened_input_shape: usize,
    flattened_input_ndim: usize,
}

impl<T: Tensor> FeedForward<T> {
    fn new(input_shape: Vec<usize>, neurons: usize, activation: Activation) -> Self {
        let flattened = CPUTensor::len(&input_shape);
        let xavier = Normal::new(0.0, 2.0 / (flattened + neurons) as f64).unwrap();
        Self {
            weights: T::tensor(Generate {
                shape: vec![neurons, flattened],
                with: || xavier.sample(&mut rand::rng()),
            })
            .unwrap(),
            biases: T::tensor(Fill {
                shape: vec![neurons],
                with: 0.0
            })
            .unwrap(),
            activation,
            flattened_input_shape: flattened,
            flattened_input_ndim: input_shape.len(),
        }
    }

    fn forward(&self, activations: T) -> T {
        // we ignore any additional batching parameters that would be appended to the activation shape
        let mut flattened_shape = vec![self.flattened_input_shape];
        flattened_shape.extend_from_slice(&activations.shape()[self.flattened_input_ndim..]);
        // dbg!(activations.shape());
        self.activation.activate(
            self.weights
                .dot(
                    &activations.reshape(&flattened_shape).unwrap(), 1)
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
        let mut finb = [0u8; 8];
        read.read(&mut finb)?;
        let flattened_input_ndim = usize::from_le_bytes(finb);
        let mut fisb = [0u8; 8];
        let flattened_input_shape = usize::from_le_bytes(fisb);
        read.read(&mut fisb)?;
        let weights = T::read(read)?;
        let biases = T::read(read)?;
        Ok(Self {
            weights,
            biases,
            activation,
            flattened_input_ndim,
            flattened_input_shape
        })
    }

    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"FeedFrwd")?;
        write.write_all(self.activation.id())?;
        write.write_all(&self.flattened_input_ndim.to_le_bytes())?;
        write.write_all(&self.flattened_input_shape.to_le_bytes())?;
        self.weights.write(write)?;
        self.biases.write(write)
    }
}

struct Conv2D<T: Tensor> {
    weights: T,
    biases: T,
    field: Field,
    activation: Activation,
}

impl<T: Tensor> Conv2D<T> {
    fn new(field: Field, filters: usize, activation: Activation, input_size: usize) -> Self {
        let he = Normal::new(0.0, 2.0 / (field.size * field.size) as f64).unwrap();
        let weights = T::tensor(Generate {
            shape: vec![filters, field.size * field.size],
            with: || he.sample(&mut rand::rng()),
        }).unwrap();
        let locs = field
            .locations_on(input_size)
            .unwrap();
        let biases = T::tensor(Generate {
            shape: vec![locs, locs, filters],
            with: || 0.0
        }).unwrap();
        Self {
            weights,
            biases,
            field,
            activation,
        }
    }

    fn forward(&self, activations: &T) -> T {
        let locs = self
            .field
            .locations_on(*activations.shape().first().unwrap())
            .unwrap();
        let colified = activations.colify(self.field).unwrap();
        let convovled = self.weights.dot(&colified, 1).unwrap();
        let shape: Vec<usize> = [locs, locs, self.weights.shape()[0]]
            .into_iter()
            .chain(activations.shape()[2..].iter().cloned())
            .collect();
        let mut axes: Vec<usize> = (0..convovled.ndim()).collect();
        axes.swap(0, 1);
        let fixed = convovled.transpose(&axes).unwrap();
        let shaped = fixed.reshape(&shape).unwrap();
        self.activation.activate(shaped.add(&self.biases).unwrap())
    }
}

impl <T: TensorIO> Conv2D<T> {
    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        self.field.write(write)?;
        write.write_all(self.activation.id())?;
        self.weights.write(write)?;
        self.biases.write(write)?;
        Ok(())
    }
}

struct Pool2D<T: Tensor> {
    field: Field,
    phantom: PhantomData<T>,
}

impl<T: Tensor> Pool2D<T> {
    fn new(field: Field) -> Self {
        Self {
            field,
            phantom: PhantomData,
        }
    }
}

impl<T: Tensor> Pool2D<T> {
    fn forward(&self, activations: T) -> T {
        let locs = self
            .field
            .locations_on(*activations.shape().first().unwrap())
            .unwrap();

        let colified = activations.colify(self.field).unwrap();
        let maxed = colified.colmax().unwrap();
        let shape: Vec<usize> = [locs, locs]
            .into_iter()
            .chain(activations.shape()[2..].iter().cloned())
            .collect();

        maxed.reshape(&shape).unwrap()
    }
}

impl <T: TensorIO> Pool2D<T> {
    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        self.field.write(write)?;
        Ok(())
    }
}