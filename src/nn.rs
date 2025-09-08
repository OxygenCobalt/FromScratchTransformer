use std::{fs::File, io::{self, Read, Write}};

use rand::{seq::IndexedRandom, Rng};

use crate::tensor::Tensor;

use super::{
    loss::Loss, activation::Activation,
    autograd::{Autograd}, dataset::{Example, Test, Train}, matrix::{Matrix, Shape}
};

pub struct NeuralNetwork {
    axons: Vec<Axon2>
}

impl NeuralNetwork {
    pub fn new(layers: &[Layer2]) -> Self {
        if layers.len() < 2 {
            panic!("not enough layers");
        }
        let mut axons = vec![];
        for i in 0..(layers.len() - 1) {
            // let layer = &layers[i];
            // let next_layer = &layers[i + 1];
            // let in_size = layer.neurons;
            // let out_size = next_layer.neurons;
            // let axon = Axon {
            //     weights: Matrix::normal(Shape { m: out_size, n: in_size }),
            //     biases: Matrix::normal(Shape::vector(out_size)),
            //     activation_fn: layer.activation_fn
            // };
            // axons.push(axon);
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
            for batch in train.batch(batch_size) {
                // struct AutoAxon {
                //     weights: Autograd,
                //     biases: Autograd,
                //     activation_fn: Activation
                // }

                // let auto_axons: Vec<AutoAxon> = self.axons.iter().map(|a| AutoAxon {
                //     weights: Autograd::new(a.weights.clone()),
                //     biases: Autograd::new(Matrix::columns(vec![&a.biases; batch_size].as_slice())), // need to apply it to all of the batch at once
                //     activation_fn: a.activation_fn
                // }).collect();

                let auto_axons: Vec<AxonMut<'_>> = self.axons.iter_mut().map(|a| a.train(batch_size)).collect();

                let mut current = Autograd::new(batch.input);

                for axon in &auto_axons {
                    // let weighted = axon.weights.dot(&current).add(&axon.biases);
                    // let activation = weighted.execute_with(axon.activation_fn);
                    current = axon.forward(&current)
                }
                // eval loss and backpropagate back to weights/biases
                current.execute_with(loss.op(batch.output)).backward();
                // this drops the entire computation graph allowing us to move the weight/bias
                // gradients out w/o a clone
                std::mem::drop(current);
                
                let scale_by = learning_rate / batch_size as f64;
                for axon in auto_axons {
                    axon.commit(scale_by);
                }

                // for (axon, auto_axon) in self.axons.iter_mut().zip(auto_axons.into_iter()) {
                //     axon.weights.sub_assign(&auto_axon.weights.into_grad().unwrap().scale(scale_by));
                //     axon.biases.sub_assign(&auto_axon.biases.into_grad().unwrap().nsum().scale(scale_by));
                // }
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
            sum += loss.loss(&example.output, &result);
        }
        let loss = sum / test.examples.len() as f64;
        return TestResult {
            avg_loss: loss,
            successes
        }
    }

    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for axon in &self.axons {
            current = axon.forward(&current)
        }
        current
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        // write.write(b"NeuralNt")?;
        // write.write_all(&self.axons.len().to_le_bytes())?;
        // for axon in &self.axons {
        //     write.write_all(b"Axon\0\0\0\0")?;
        //     write.write_all(axon.activation_fn.id())?;
        //     axon.weights.write(write)?;
        //     axon.biases.write(write)?;
        // }
        Ok(())
    }

    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        // let mut signature = [0u8; 8];
        // read.read(&mut signature)?;
        // if &signature != b"NeuralNt" {
        //     return Err(io::Error::new(io::ErrorKind::Other, "invalid neuralnet signature"));
        // }
        // let mut nb = [0u8; 8];
        // read.read(&mut nb)?;
        // let axon_count = usize::from_le_bytes(nb);
        // let mut axons = Vec::with_capacity(axon_count);
        // for _ in 0..axon_count {
        //     let mut signature = [0u8; 8];
        //     read.read(&mut signature)?;
        //     if &signature != b"Axon\0\0\0\0" {
        //         return Err(io::Error::new(io::ErrorKind::Other, "invalid axon signature"));
        //     }
        //     let mut fnb = [0u8; 8];
        //     read.read(&mut fnb)?;
        //     let activation_fn = Activation::from_id(&fnb)
        //         .ok_or(io::Error::new(io::ErrorKind::Other, "invalid activation function"))?;
        //     let weights = Matrix::read(read)?;
        //     let biases  = Matrix::read(read)?;
        //     axons.push(Axon {
        //         weights,
        //         biases,
        //         activation_fn
        //     });
        // }
        // Ok(Self { axons })
        todo!()
    }
}

pub struct TestResult {
    pub avg_loss: f64,
    pub successes: u64,
}

pub trait SuccessCriteria {
    fn is_success(&self, example: &Example, output: &Matrix) -> bool;
}

pub trait Reporting: SuccessCriteria {
    fn data(&self) -> &Test;
    fn report(&self, epoch: Option<u64>, result: TestResult);
}

pub struct Layer {
    pub neurons: usize,
    pub activation_fn: Activation
}

struct Axon {
    weights: Matrix,
    biases: Matrix,
    activation_fn: Activation
}

pub enum Layer2 {
    Dense { neurons: usize, activation: Activation },
    Dropout { neurons: usize, rate: f64, activation: Activation }
}

impl Layer2 {
    fn axon(&self, last: &Layer2) -> Axon2 {
        match self {
            Self::Dense { neurons, activation } => Axon2::Dense {
                weights: Matrix::normal(Shape { m: last.activation_shape().m, n: *neurons }),
                biases: Matrix::normal(Shape::vector(last.activation_shape().m)),
                activation: *activation
            },
            Self::Dropout { neurons, rate, activation } => Axon2::Dropout {
                weights: Matrix::normal(Shape { m: last.activation_shape().m, n: *neurons }),
                biases: Matrix::normal(Shape::vector(last.activation_shape().m)),
                rate: *rate,
                activation: *activation
            }
        }
    }


    fn activation_shape(&self) -> Shape {
        match self {
            Self::Dense { neurons, .. } => Shape::vector(*neurons),
            Self::Dropout { neurons, .. } => Shape::vector(*neurons),
        }
    }
}

enum Axon2  {
    Dense { weights: Matrix, biases: Matrix, activation: Activation },
    Dropout { weights: Matrix, biases: Matrix, rate: f64, activation: Activation }
}

impl Axon2 {
    pub fn forward(&self, activations: &Matrix) -> Matrix {
        match self {
            Self::Dense { weights, biases, activation, .. } => 
            activation.activate(weights.clone().dot(&activations).add(&biases)),
            Self::Dropout { weights, biases, activation, .. } => 
            activation.activate(weights.clone().dot(&activations).add(&biases))
        }
    }

    fn train<'a>(&'a mut self, batch_size: usize) -> AxonMut<'a> {
        match self {
            Self::Dense { weights, biases, activation } => {
                AxonMut::Dense { auto_weights: Autograd::new(weights.clone()), auto_biases: Autograd::new(Matrix::columns(vec![biases as &Matrix; batch_size].as_slice())), weights, biases,  activation: *activation }
            },
            Self::Dropout { weights, biases, rate, activation } => {
                let mut samples: Vec<usize> = (0..biases.shape().m).into_iter().collect();
                let drop: Vec<usize> = IndexedRandom::choose_multiple(samples.as_mut_slice(), &mut rand::rng(), (biases.shape().m as f64 * *rate) as usize).cloned().collect();
                AxonMut::Dropout { auto_weights: Autograd::new(weights.clone()), drop, auto_biases: Autograd::new(Matrix::columns(vec![biases as &Matrix; batch_size].as_slice())), weights, biases, rate: *rate, activation: *activation }
            }
        }
    }
}

enum AxonMut<'a> {
    Dense { weights: &'a mut Matrix, biases: &'a mut Matrix, auto_weights: Autograd, auto_biases: Autograd, activation: Activation },
    Dropout { weights: &'a mut Matrix, biases: &'a mut Matrix, auto_weights: Autograd, auto_biases: Autograd, drop: Vec<usize>, rate: f64, activation: Activation }
}

impl  <'a> AxonMut<'a> {
    pub fn forward(&self, activations: &Autograd) -> Autograd {
        match self {
            Self::Dense { auto_weights, auto_biases, activation, .. } => 
            auto_weights.dot(&activations).add(&auto_biases).execute_with(*activation),
            Self::Dropout { auto_weights, auto_biases, activation, drop, rate, .. } => 
            auto_weights.dot(&activations).add(&auto_biases).execute_with(*activation).hacky_dropout(drop).scale(1.0 / (1.0 - rate))
        }
    }

    pub fn commit(self, c: f64) {
        match self {
            Self::Dense { weights, biases, auto_weights, auto_biases, .. } => {
                    weights.sub_assign(&auto_weights.into_grad().unwrap().scale(c));
                    biases.sub_assign(&auto_biases.into_grad().unwrap().nsum().scale(c));
                }
            Self::Dropout { weights, biases, auto_weights, auto_biases, .. } => {
                    weights.sub_assign(&auto_weights.into_grad().unwrap().scale(c));
                    biases.sub_assign(&auto_biases.into_grad().unwrap().nsum().scale(c));
                }
        }
    }
}

struct DenseParams<T: Tensor> {
    weights: T,
    biases: T,
    activation: Activation
}

impl <T: Tensor> DenseParams<T> {
    fn forward(&self, activations: T) -> T {
        self.activation.activate2(self.weights.clone().dot(&activations, (&[1], &[0])).unwrap())
    }
}