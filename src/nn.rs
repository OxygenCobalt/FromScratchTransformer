use std::{fs::File, io::{self, Read, Write}};
use std::sync::LazyLock;
use rand_distr::{Distribution, Normal};

use crate::tensor::{CPUTensor, Generate, SharpTensor, Tensor, TensorIO};

use super::{
    loss::Loss, activation::Activation,
    autograd::{Autograd}, dataset::{Example, Test, Train}
};

pub struct NeuralNetwork {
    axons: Vec<Axon<CPUTensor>>
}

impl NeuralNetwork {
    pub fn new(layers: &[Layer]) -> Self {
        if layers.len() < 2 {
            panic!("not enough layers");
        }
        let mut axons = vec![];
        for i in 0..(layers.len() - 1) {
            axons.push(layers[i].axon(&layers[i + 1]));
        }

        Self {
            axons
        }
    }

    pub fn train(
        &mut self,
        train: &Train,
        epochs: u64,
        batch_size: usize,
        learning_rate: f64,
        loss: &impl Loss,
        reporting: Option<impl Reporting>,
        checkpoint_to: Option<&str>
    ) -> io::Result<()> {
        if let Some(ref rep) = reporting {
            rep.report(None, self.test(rep.data(), rep, loss));
        }
        if let Some(path) = checkpoint_to {
            let nn_path = path.to_owned() + "/init.nn";
            let mut file = File::create(nn_path)?;
            self.write(&mut file)?;
        }
        for epoch in 0..epochs {
            for (i, batch) in train.batch(batch_size).into_iter().enumerate() {
                let auto_axons: Vec<Axon<Autograd<CPUTensor>>> = self.axons.iter_mut().map(|a| a.train()).collect();
                let mut current = Autograd::new(batch.input);
                for axon in &auto_axons {
                    current = axon.forward(&current)
                }
                loss.loss(&current, &Autograd::new(batch.output)).backward();
                // this drops the entire computation graph allowing us to move the weight/bias
                // gradients out w/o a clone
                std::mem::drop(current);
                
                let c = learning_rate / batch_size as f64;
                for (axon, auto_axon) in self.axons.iter_mut().zip(auto_axons.into_iter()) {
                    axon.commit(auto_axon, c);
                }
            }
            if let Some(ref rep) = reporting {
                rep.report(Some(epoch), self.test(rep.data(), rep, loss));
            }
            if let Some(path) = checkpoint_to {
                let nn_path = path.to_owned() + &format!["/{}.nn", epoch + 1];
                let mut file = File::create(nn_path)?;
                self.write(&mut file)?;
            }
        }
        Ok(())
    }

    pub fn test(&self, test: &Test, success_criteria: &impl SuccessCriteria, loss: &impl Loss) -> TestResult {
        let mut sum = 0.0;
        let mut successes = 0;
        for example in &test.examples {
            let result = self.evaluate(&example.input);
            if success_criteria.is_success(example, &result) {
                successes += 1;
            }
            sum += loss.loss(&example.output, &result).get(&[]).unwrap();
        }
        let loss = sum / test.examples.len() as f64;
        return TestResult {
            avg_loss: loss,
            successes
        }
    }

    pub fn evaluate(&self, input: &CPUTensor) -> CPUTensor {
        let mut current = input.clone();
        for axon in &self.axons {
            current = axon.forward(&current)
        }
        current
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write(b"NeuralNt")?;
        write.write_all(&self.axons.len().to_le_bytes())?;
        for axon in &self.axons {
            axon.write(write)?;
        }
        Ok(())
    }

    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read(&mut signature)?;
        if &signature != b"NeuralNt" {
            return Err(io::Error::new(io::ErrorKind::Other, "invalid neuralnet signature"));
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
}

pub struct TestResult {
    pub avg_loss: f64,
    pub successes: u64,
}

pub trait SuccessCriteria {
    fn is_success(&self, example: &Example, output: &impl SharpTensor) -> bool;
}

pub trait Reporting: SuccessCriteria {
    fn data(&self) -> &Test;
    fn report(&self, epoch: Option<u64>, result: TestResult);
}

pub static NORMAL: LazyLock<Normal<f64>> = std::sync::LazyLock::new(|| Normal::new(0.0, 1.0).unwrap());

pub enum Layer {
    Dense { neurons: usize, activation: Activation },
    Dropout { neurons: usize, rate: f64, activation: Activation }
}

impl Layer {
    fn axon<T: Tensor>(&self, last: &Layer) -> Axon<T> {
        match self {
            Self::Dense { neurons, activation } => Axon::Dense { 
                ff: FeedForward::new(last.activation_shape()[0], *neurons, *activation) 
            },
            Self::Dropout { neurons, rate, activation } => Axon::Dropout { 
                ff: FeedForward::new(last.activation_shape()[0], *neurons, *activation), 
                rate: *rate 
            },
        }
    }

    fn activation_shape(&self) -> Vec<usize> {
        match self {
            Self::Dense { neurons, .. } => vec![*neurons],
            Self::Dropout { neurons, .. } => vec![*neurons]
        }
    }
}

struct FeedForward<T: Tensor> {
    weights: T,
    biases: T,
    activation: Activation
}

impl <T: Tensor> FeedForward<T> {
    fn new(last: usize, next: usize, activation: Activation) -> Self {
        Self {
                weights: T::tensor(Generate { shape: vec![last, next], with: || NORMAL.sample(&mut rand::rng())}).unwrap(),
                biases: T::tensor(Generate { shape: vec![last], with: || NORMAL.sample(&mut rand::rng())}).unwrap(),
                activation
        }
    }

    fn forward(&self, activations: &T) -> T {
        self.activation.activate(self.weights.clone()
            .dot(&activations, 1).unwrap()
            .add(&self.biases).unwrap())
    }
}

impl <T: TensorIO> FeedForward<T> {
    fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read(&mut signature)?;
        if &signature != b"FeedFrwd" {
            return Err(io::Error::new(io::ErrorKind::Other, "invalid feedforward signature"))
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
            biases
        })
    }

    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"FeedFrwd")?;
        write.write_all(self.activation.id())?;
        self.weights.write(write)?;
        self.biases.write(write)
    }
}

enum Axon<T: Tensor>  {
    Dense { ff: FeedForward<T> },
    Dropout { ff: FeedForward<T>, rate: f64 }
}

impl <T: Tensor> Axon<T> {
    pub fn forward(&self, activations: &T) -> T {
        match self {
            Self::Dense { ff } => ff.forward(activations),
            Self::Dropout { ff, .. } => ff.forward(activations)
        }
    }
}

impl <T: SharpTensor> Axon<T> {
    fn train<'a>(&'a mut self) -> Axon<Autograd<T>> {
        match self {
            Self::Dense { ff } => {
                Axon::Dense { ff: FeedForward { 
                    weights: Autograd::new(ff.weights.clone()),
                    biases: Autograd::new(ff.biases.clone()),
                    activation: ff.activation
                }}
            },
            Self::Dropout { ff, rate } =>  {
                let mut dropped_weights = ff.weights.clone();
                for row in 0..dropped_weights.shape()[1] {
                    if rand::random_range(0.0..1.0) > *rate {
                        continue
                    }
                    for col in 0..dropped_weights.shape()[0] {
                        *dropped_weights.get_mut(&[col, row]).unwrap() = 0.0;
                    }
                }
                Axon::Dense { ff: FeedForward { 
                    weights: Autograd::new(dropped_weights),
                    biases: Autograd::new(ff.biases.clone()),
                    activation: ff.activation
                }}
            }
        }
    }

    pub fn commit(&mut self, axon: Axon<Autograd<T>>, c: f64) -> Option<()> {
        let scale = T::scalar(c);
        match (self, axon) {
            (Self::Dense { ff  }, Axon::<Autograd<T>>::Dense { ff: autoff }) => {
                ff.weights = ff.weights.clone().sub(&autoff.weights.into_grad().unwrap().mul(&scale).unwrap()).unwrap();
                ff.biases = ff.biases.clone().sub(&autoff.biases.into_grad().unwrap().mul(&scale).unwrap()).unwrap();
            },
            (Self::Dropout { ff, ..  }, Axon::<Autograd<T>>::Dropout { ff: autoff, .. }) => {
                ff.weights = ff.weights.clone().sub(&autoff.weights.into_grad().unwrap().mul(&scale).unwrap()).unwrap();
                ff.biases = ff.biases.clone().sub(&autoff.biases.into_grad().unwrap().mul(&scale).unwrap()).unwrap();
            },
            _ => return None
        }
        Some(())
    }
}

impl <T: TensorIO> Axon<T> {
    fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut id = [0u8; 8];
        read.read(&mut id)?;
        match &id {
            b"AxonDnse" => {
                Ok(Self::Dense { ff: FeedForward::read(read)? })
            }
            b"AxonDrop" => {
                let mut rateb = [0u8; 8];
                read.read(&mut rateb)?;
                Ok(Self::Dropout { ff: FeedForward::read(read)?, rate: f64::from_le_bytes(rateb) })
            }
            _ => Err(io::Error::new(io::ErrorKind::Other, "invalid axon signature"))
        }
    }

    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Dense { ff } => {
                write.write_all(b"AxonDnse")?;
                ff.write(write)
            },
            Self::Dropout { ff, rate } => {
                write.write_all(b"AxonDrop")?;
                write.write_all(&rate.to_le_bytes())?;
                ff.write(write)
            }
        }
    }
}