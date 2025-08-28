use std::{fs::File, io::{self, Read, Write}, marker::PhantomData};

use rand_distr::num_traits::sign;

use crate::{
    dataset::{Example, Test, Train},
    matrix::{Matrix, Shape},
};

pub struct NeuralNetwork {
    axons: Vec<Axon>
}

impl NeuralNetwork {
    pub fn new(layers: &[Layer]) -> Self {
        if layers.len() < 2 {
            panic!("not enough layers");
        }
        let mut axons = vec![];
        for i in 0..(layers.len() - 1) {
            let layer = &layers[i];
            let next_layer = &layers[i + 1];
            let in_size = layer.neurons;
            let out_size = next_layer.neurons;
            let axon= Axon {
                weights: Matrix::noisy(Shape { m: out_size, n: in_size,  }),
                biases: Matrix::noisy(Shape::vector(out_size)),
                activation_fn: layer.activation_fn
            };
            axons.push(axon);
        }

        Self {
            axons
        }
    }

    pub fn train(
        &mut self,
        train: &mut Train,
        epochs: u64,
        batch_size: usize,
        learning_rate: f64,
        loss: &impl Loss,
        regularization: &impl Regularization,
        reporting: Option<impl Reporting>,
        checkpoint_to: Option<&str>
    ) -> io::Result<()> {
        if let Some(ref rep) = reporting {
            rep.report(None, self.test(rep.data(), rep, loss, regularization));
        }
        if let Some(path) = checkpoint_to {
            let nn_path = path.to_owned() + "/init.nn";
            let mut file = File::create(nn_path)?;
            self.write(&mut file)?;
        }
        let train_size = train.examples.len();
        for epoch in 0..epochs {
            for batch in train.batch(batch_size) {
                // TODO: Switch to matrix slicing across a 3d matrix, requires tensors :(
                let mut sum_nablas: Vec<NablaLoss> = Vec::with_capacity(self.axons.len());
                for example in batch {
                    let mut nablas = self.backprop(example, loss);
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
                for (axon, sum_nabla) in self.axons.iter_mut().zip(sum_nablas) {
                    let regularization = regularization.nabla_regularization_w(train_size);
                    if regularization > 0.0 {
                        axon.weights.scale_assign(1.0 - (learning_rate * regularization));
                    }
                    axon.weights.sub_assign(&sum_nabla.weights.scale(scale_by));
                    axon.biases.sub_assign(&sum_nabla.biases.scale(scale_by));
                }
            }
            if let Some(ref rep) = reporting {
                rep.report(Some(epoch), self.test(rep.data(), rep, loss, regularization));
            }
            if let Some(path) = checkpoint_to {
                let nn_path = path.to_owned() + &format!["/{}.nn", epoch + 1];
                let mut file = File::create(nn_path)?;
                self.write(&mut file)?;
            }
        }
        Ok(())
    }

    fn backprop(&self, example: &Example, loss: &impl Loss) -> Vec<NablaLoss> {
        // backprop assumes that
        // 1. the loss function can be written as the avg of several training examples
        //    this is bc backprop allows us to compute delta(w,b) for a specific example only
        //    so we activationsta have avg it out if we want to perform sgd
        // 2. the loss must be written as a fn of the neural networks outputs
        
        pub struct ErrorAndWeights<'a> {
            weights: &'a Matrix,
            error: Matrix
        }

        let mut active_axons: Vec<ActiveAxon> = Vec::with_capacity(self.axons.len());
        for axon in &self.axons {
            let current = active_axons.last().map(|l| &l.activation).unwrap_or_else(|| &example.input);
            let weighted_input = (axon.weights.clone().dot(current)).add(&axon.biases);
            let activation = axon.activation_fn.activation(weighted_input.clone());
            active_axons.push(ActiveAxon {
                axon: &axon,
                weighted_input,
                activation,
            });
        }

        fn last_activations<'a>(example: &'a Example, active_axons: &'a Vec<ActiveAxon<'_>>) -> &'a Matrix {
            active_axons.last()
                .map(|v| &v.activation)
                .unwrap_or_else(|| &example.input)
        } 

        fn nabla_loss(active_axon: ActiveAxon<'_>, activations_in: &Matrix, error: &Matrix) -> NablaLoss {
            NablaLoss {
                weights: Matrix::new(active_axon.axon.weights.shape(), |i, j| {
                    unsafe { activations_in.get_unchecked(j, 0) * error.get_unchecked(i, 0) }
                }),
                biases: error.clone()
            }
        }


        let output_axon: ActiveAxon<'_> = active_axons.pop().unwrap();
        let activations_in = last_activations(example, &active_axons);
        let nabla_output = loss.for_train(&example.output, &output_axon, activations_in);
        let mut error = ErrorAndWeights {
            weights: &output_axon.axon.weights,
            error: nabla_output.error
        };

        let mut nablas = Vec::with_capacity(self.axons.len());
        nablas.push(nabla_output.nabla_loss.unwrap_or_else(|| nabla_loss(output_axon, activations_in, &error.error)));

        while let Some(active_axon) = active_axons.pop() {
            error = ErrorAndWeights {
                weights: &active_axon.axon.weights,
                error: (error.weights.clone().transpose().dot(&error.error))
                    .mul(&active_axon.axon.activation_fn.activation_prime(active_axon.weighted_input.clone()))
            };
            let activations_in = last_activations(example, &active_axons);
            nablas.push(nabla_loss(active_axon, &activations_in, &error.error));
        }

        nablas.reverse();
        nablas
    }

    pub fn test(&self, test: &Test, success_criteria: &impl SuccessCriteria, loss: &impl Loss, regularization: &impl Regularization) -> TestResult {
        let mut sum = 0.0;
        let mut successes = 0;
        for example in &test.examples {
            let result = self.evaluate(&example.input);
            if success_criteria.is_success(example, &result) {
                successes += 1;
            }
            sum += loss.for_test(&example.output, &result);
        }
        let loss = sum / test.examples.len() as f64;
        let regularization = regularization.regularization(self.axons.iter().map(|a| &a.weights), test.examples.len());
        return TestResult {
            avg_loss: loss + regularization,
            successes
        }
    }

    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for axon in &self.axons {
            let weighted = axon.weights.clone().dot(&current).add(&axon.biases);
            let activation = axon.activation_fn.activation(weighted);
            current = activation;
        }
        current
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write(b"NeuralNt")?;
        write.write_all(&self.axons.len().to_le_bytes())?;
        for axon in &self.axons {
            write.write_all(b"Axon\0\0\0\0")?;
            write.write_all(axon.activation_fn.id())?;
            axon.weights.write(write)?;
            axon.biases.write(write)?;
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
            let mut signature = [0u8; 8];
            read.read(&mut signature)?;
            if &signature != b"Axon\0\0\0\0" {
                return Err(io::Error::new(io::ErrorKind::Other, "invalid axon signature"));
            }
            let mut fnb = [0u8; 8];
            read.read(&mut fnb)?;
            let activation_fn = ActivationFn::from_id(&fnb)
                .ok_or(io::Error::new(io::ErrorKind::Other, "invalid activation function"))?;
            let weights = Matrix::read(read)?;
            let biases  = Matrix::read(read)?;
            axons.push(Axon {
                weights,
                biases,
                activation_fn
            });
        }
        Ok(Self { axons })
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

    fn id(&self) -> &'static [u8; 8] {
        match self {
            Self::Sigmoid => b"Sigmoid\0",
            Self::ReLU =>    b"ReLU\0\0\0\0"
        }
    }

    fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"Sigmoid\0" => Some(Self::Sigmoid),
            b"ReLU\0\0\0\0" => Some(Self::ReLU),
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
    fn for_test(&self, example_output: &Matrix, output_activations: &Matrix) -> f64;
    fn for_train(&self, example_output: &Matrix, output_axon: &ActiveAxon, activations_in: &Matrix) -> NablaOutput;
}

pub struct MSE;

impl Loss for MSE {
    fn for_test(&self, example_output: &Matrix, output_activations: &Matrix) -> f64 {
        (output_activations.clone().sub(example_output)).norm() / 2.0
    }

    fn for_train(&self, example_output: &Matrix, output_axon: &ActiveAxon, _: &Matrix) -> NablaOutput {
        let error = output_axon.activation.clone().sub(&example_output).mul(&output_axon.axon.activation_fn.activation(output_axon.weighted_input.clone()));
        NablaOutput { error, nabla_loss: None }
    }
}

pub struct CrossEntropy;

impl Loss for CrossEntropy {
    fn for_test(&self, example_output: &Matrix, output_activations: &Matrix) -> f64 {
        example_output.sum_with(|i, j, y| {
            let a = output_activations.get(i, j);
            -(y * a.ln()) + (1.0 - y) * (1.0 - a).ln()
        })
    }

    fn for_train(&self, example_output: &Matrix, output_axon: &ActiveAxon, activations_in: &Matrix) -> NablaOutput {
        let error = output_axon.activation.clone().sub(&example_output);
        let nabla_loss = NablaLoss {
            weights: output_axon.axon.weights.clone().apply_indexed(|i, j, _| {
                activations_in.get(j, 0) * 
                    ((output_axon.activation.get(i, 0) - 
                        example_output.get(i, 0)))
            }),
            biases: output_axon.activation.clone().sub(&example_output)
        };
        NablaOutput { error, nabla_loss: Some(nabla_loss) }
    }
}

pub trait Regularization {
    fn regularization<'a>(&self, weights: impl Iterator<Item=&'a Matrix>, train_size: usize) -> f64;
    fn nabla_regularization_w(&self, train_size: usize) -> f64;
}

pub struct NoRegularization;

impl Regularization for NoRegularization {
    fn regularization<'a>(&self, _: impl Iterator<Item=&'a Matrix>, _: usize) -> f64 {
        0.0
    }

    fn nabla_regularization_w(&self, _: usize) -> f64 {
        0.0
    }
}

pub struct L2 { pub lambda: f64 }

impl Regularization for L2 {
    fn regularization<'a>(&self, weights: impl Iterator<Item=&'a Matrix>, train_size: usize) -> f64 {
        (self.lambda / (2.0 * train_size as f64)) * weights.map(|w| w.norm()).sum::<f64>()
    }

    fn nabla_regularization_w(&self, train_size: usize) -> f64 {
        self.lambda / train_size as f64
    }
}

pub struct NablaOutput {
    error: Matrix,
    nabla_loss: Option<NablaLoss>
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

pub struct Axon {
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

pub struct ActiveAxon<'a> {
    axon: &'a Axon,
    weighted_input: Matrix,
    activation: Matrix,
}