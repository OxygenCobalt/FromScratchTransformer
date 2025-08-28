use std::{fs::File, io::{self, Read, Write}, marker::PhantomData};

use crate::{
    dataset::{Example, Test, Train},
    matrix::{Matrix, Shape},
};

pub struct NeuralNetwork<L: Loss> {
    layers: Vec<HiddenLayer>,
    _loss: PhantomData<L>,
}

impl<L: Loss> NeuralNetwork<L> {
    pub fn new(layers: &[Layer]) -> Self {
        if layers.len() < 2 {
            panic!("not enough layers");
        }
        let mut hidden_layers = vec![];
        for i in 0..(layers.len() - 1) {
            let layer = &layers[i];
            let next_layer = &layers[i + 1];
            let in_size = layer.neurons;
            let out_size = next_layer.neurons;
            let layer= HiddenLayer {
                weights: Matrix::noisy(Shape { m: out_size, n: in_size,  }),
                biases: Matrix::noisy(Shape::vector(out_size)),
                activation_fn: layer.activation_fn
            };
            hidden_layers.push(layer);
        }

        Self {
            layers: hidden_layers,
            _loss: PhantomData,
        }
    }

    pub fn train(&mut self, train: &mut Train, epochs: u64, batch_size: usize, learning_rate: f64, reporting: Option<impl Reporting>, checkpoint_to: Option<&str>) -> io::Result<()> {
        if let Some(ref rep) = reporting {
            rep.report(None, self.test(rep.data(), rep));
        }
        if let Some(path) = checkpoint_to {
            let nn_path = path.to_owned() + "/init.nn";
            let mut file = File::create(nn_path)?;
            self.write(&mut file)?;
        }
        for epoch in 0..epochs {
            for batch in train.batch(batch_size) {
                // TODO: Switch to matrix slicing across a 3d matrix, requires tensors :(
                let mut sum_nablas: Vec<NablaLoss> = Vec::with_capacity(self.layers.len());
                for example in batch {
                    let mut nablas = self.backprop(example);
                    if sum_nablas.is_empty() {
                        sum_nablas.append(&mut nablas);
                        continue;
                    }
                    for (sum_nabla, nabla) in sum_nablas.iter_mut().zip(nablas) {
                        sum_nabla.weights.add_assign(&nabla.weights);
                        sum_nabla.biases.add_assign(&nabla.biases);
                    }
                }
                let scale_by = learning_rate / batch_size as f64;
                for (layer, sum_nabla) in self.layers.iter_mut().zip(sum_nablas) {
                    layer.weights.sub_assign(&sum_nabla.weights.scale(scale_by));
                    layer.biases.sub_assign(&sum_nabla.biases.scale(scale_by));
                }
            }
            if let Some(ref rep) = reporting {
                rep.report(Some(epoch), self.test(rep.data(), rep));
            }
            if let Some(path) = checkpoint_to {
                let nn_path = path.to_owned() + &format!["/{}.nn", epoch + 1];
                let mut file = File::create(nn_path)?;
                self.write(&mut file)?;
            }
        }
        Ok(())
    }

    fn backprop(&self, example: &Example) -> Vec<NablaLoss> {
        // backprop assumes that
        // 1. the loss function can be written as the avg of several training examples
        //    this is bc backprop allows us to compute delta(w,b) for a specific example only
        //    so we gotta have avg it out if we want to perform sgd
        // 2. the loss must be written as a fn of the neural networks outputs
        pub struct CalculatedLayer<'a> {
            weights: &'a Matrix,
            biases: &'a Matrix,
            weighted_input: Matrix,
            activation: Matrix,
            activation_fn: ActivationFn 
        }
        
        pub struct Error<'a> {
            weights: &'a Matrix,
            error: Matrix
        }

        let mut calculated_layers: Vec<CalculatedLayer> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let current = calculated_layers.last().map(|l| &l.activation).unwrap_or_else(|| &example.input);
            let weighted_input = (layer.weights.clone().dot(current)).add(&layer.biases);
            let activation = layer.activation_fn.activation(weighted_input.clone());
            calculated_layers.push(CalculatedLayer {
                weights: &layer.weights,
                biases: &layer.biases,
                weighted_input,
                activation,
                activation_fn: layer.activation_fn
            });
        }

        let last: CalculatedLayer = calculated_layers.pop().unwrap();
        let mut error = Error {
            weights: last.weights,
            error: L::delta_loss(&example.output, &last.activation).mul(&last.activation_fn.activation_prime(last.weighted_input))
        };

        let mut nablas = Vec::with_capacity(self.layers.len());
        let activations_in = calculated_layers.last()
            .map(|v| &v.activation)
            .unwrap_or_else(|| &example.input);
        nablas.push(L::nabla_loss(last.weights, last.biases, activations_in, &error.error));

        while let Some(calculated_layer) = calculated_layers.pop() {
            error = Error {
                weights: calculated_layer.weights,
                error: (error.weights.clone().transpose().dot(&error.error))
                    .mul(&calculated_layer.activation_fn.activation_prime(calculated_layer.weighted_input))
            };
            let activations_in = calculated_layers.last()
                .map(|v| &v.activation)
                .unwrap_or_else(|| &example.input);
            nablas.push(L::nabla_loss(calculated_layer.weights, calculated_layer.biases, activations_in, &error.error));
        }

        nablas.reverse();
        nablas
    }

    pub fn test(&self, test: &Test, success_criteria: &impl SuccessCriteria) -> TestResult {
        let mut sum = 0.0;
        let mut successes = 0;
        for example in &test.examples {
            let result = self.evaluate(&example.input);
            if success_criteria.is_success(example, &result) {
                successes += 1;
            }
            sum += L::loss(&example.output, &result);
        }
        return TestResult {
            avg_loss: sum / test.examples.len() as f64,
            successes
        }
    }

    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in &self.layers {
            let weighted = layer.weights.clone().dot(&current).add(&layer.biases);
            let activation = layer.activation_fn.activation(weighted);
            current = activation;
        }
        current
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(&self.layers.len().to_le_bytes())?;
        for layer in &self.layers {
            write.write_all(layer.activation_fn.id().as_bytes())?;
            layer.weights.write(write)?;
            layer.biases.write(write)?;
        }
        Ok(())
    }

    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut nb = [0u8; 8];
        read.read(&mut nb)?;
        let layer_count = usize::from_le_bytes(nb);
        let mut layers = Vec::with_capacity(layer_count);
        for _ in 0..layer_count {
            let mut fnb = [0u8; 8];
            read.read(&mut fnb)?;
            let activation_fn = ActivationFn::from_id(&String::from_utf8_lossy(&fnb).into_owned())
                .ok_or(io::Error::new(io::ErrorKind::Other, "invalid activation function"))?;
            let weights = Matrix::read(read)?;
            let biases  = Matrix::read(read)?;
            layers.push(HiddenLayer {
                weights,
                biases,
                activation_fn
            });
        }
        Ok(Self {
            layers,
            _loss: PhantomData
        })
    }
}

#[derive(Clone, Copy)]
pub enum ActivationFn {
    Sigmoid,
    ReLU
}

impl ActivationFn {
    fn activation(&self, input: Matrix) -> Matrix {
        match self {
            Self::Sigmoid => input.apply(Self::sigmoid),
            Self::ReLU => input.apply(Self::relu)
        }
    }

    fn activation_prime(&self, input: Matrix) -> Matrix {
        match self {
            Self::Sigmoid => input.apply(Self::sigmoid_prime),
            Self::ReLU => input.apply(Self::relu_prime),
        }
    }

    fn id(&self) -> &'static str {
        match self {
            Self::Sigmoid => "sigmoid.",
            Self::ReLU =>    "relu...."
        }
    }

    fn from_id(id: &str) -> Option<Self> {
        match id {
            "sigmoid." => Some(Self::Sigmoid),
            "relu...." => Some(Self::ReLU),
            _ => None
        }
    }

    fn sigmoid(n: f64) -> f64 {
        1f64 / (1f64 + f64::exp(-n))
    }
    
    fn sigmoid_prime(n: f64) -> f64 {
        Self::sigmoid(n) * (1.0 - Self::sigmoid(n))
    }

    fn relu(n: f64) -> f64 {
        if n > 0.0 { n } else { 0.0 }
    }

    fn relu_prime(n: f64) -> f64 {
        if n > 0.0 { 1.0 } else { 0.0 }
    }
}

pub trait Loss {
    fn loss(expected: &Matrix, got: &Matrix) -> f64;
    fn delta_loss(expected: &Matrix, got: &Matrix) -> Matrix;
    fn nabla_loss(weights: &Matrix, biases: &Matrix, activations_in: &Matrix, error: &Matrix) -> NablaLoss;
}

pub struct MSE;

impl Loss for MSE {
    fn loss(expected: &Matrix, got: &Matrix) -> f64 {
        return (got.clone().sub(expected)).length() / 2.0;
    }

    fn delta_loss(expected: &Matrix, got: &Matrix) -> Matrix {
        return got.clone().sub(expected);
    }
    
    fn nabla_loss(weights: &Matrix, _: &Matrix, activations_in: &Matrix, error: &Matrix) -> NablaLoss {
        NablaLoss {
            weights: weights.clone().apply_indexed(|i, j, _| {
                unsafe { activations_in.get_unchecked(j, 0) * error.get_unchecked(i, 0) }
            }),
            biases: error.clone()
        }
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

struct HiddenLayer {
    weights: Matrix,
    biases: Matrix,
    activation_fn: ActivationFn
}

pub struct NablaLoss {
    weights: Matrix,
    biases: Matrix
}

pub struct Layer {
    pub neurons: usize,
    pub activation_fn: ActivationFn
}