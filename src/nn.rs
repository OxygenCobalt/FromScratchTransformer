use std::marker::PhantomData;

use crate::{
    dataset::{Example, Test, Train},
    matrix::{Matrix, Shape},
};

pub struct NeuralNetwork<L: Loss> {
    layers: Vec<HiddenLayer>,
    _cost: PhantomData<L>,
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
            _cost: PhantomData,
        }
    }

    pub fn train(&mut self, train: &mut Train, epochs: u64, batch_size: usize, learning_rate: f64, reporting: Option<impl Reporting>) {
            if let Some(ref rep) = reporting {
                rep.report(None, self.test(rep.data(), rep));
            }
        for epoch in 0..epochs {
            for batch in train.batch(batch_size) {
                // TODO: Switch to matrix slicing across a 3d matrix, requires tensors :(
                let mut sum_delta_weights: Vec<Matrix> = self.layers.iter().map(|l| Matrix::null(l.weights.shape())).collect();
                let mut sum_delta_biases: Vec<Matrix> = self.layers.iter().map(|l| Matrix::null(l.biases.shape())).collect();
                for example in batch {
                    let delta = self.backprop(example);
                    for (a, b) in sum_delta_weights.iter_mut().zip(delta.weights.into_iter()) {
                        *a += b;
                    }
                    for (a, b) in sum_delta_biases.iter_mut().zip(delta.biases.into_iter()) {
                        *a += b;
                    }
                }
                let scale_by = learning_rate / batch_size as f64;
                for ((layer, delta_weights), delta_biases) in self
                    .layers
                    .iter_mut()
                    .zip(sum_delta_weights.into_iter())
                    .zip(sum_delta_biases)
                {
                    layer.weights -= delta_weights.scale(scale_by);
                    layer.biases -= delta_biases.scale(scale_by);
                }
            }
            if let Some(ref rep) = reporting {
                rep.report(Some(epoch), self.test(rep.data(), rep));
            }
        }
    }

    fn backprop(&self, example: &Example) -> Delta {
        // backprop assumes that
        // 1. the cost function can be written as the avg of several training examples
        //    this is bc backprop allows us to compute delta(w,b) for a specific example only
        //    so we gotta have avg it out if we want to perform sgd
        // 2. the cost must be written as a fn of the neural networks outputs
        pub struct CalculatedLayer {
            weights: Matrix,
            weighted_input: Matrix,
            activation: Matrix,
            activation_fn: ActivationFn 
        }
        let mut calculated_layers: Vec<CalculatedLayer> = Vec::new();
        let mut current = example.input.clone();
        for layer in &self.layers {
            let weighted_input = (layer.weights.clone() * current) + layer.biases.clone();
            let activation = layer.activation_fn.activation(weighted_input.clone());
            current = activation.clone();
            calculated_layers.push(CalculatedLayer {
                weights: layer.weights.clone(),
                weighted_input,
                activation,
                activation_fn: layer.activation_fn
            });
        }
        let mut delta = Delta {
            weights: Vec::new(),
            biases: Vec::new()
        };

        pub struct Error {
            weights: Matrix,
            error: Matrix
        }

        let last: CalculatedLayer = calculated_layers.pop().unwrap();
        let mut error = Error {
            weights: last.weights.clone(),
            error: L::loss_prime(example.output.clone(), last.activation.clone()).hmul(last.activation_fn.activation_prime(last.weighted_input.clone()))
        };
        let activations_in = calculated_layers.last().map(|v| v.activation.clone()).unwrap_or(example.input.clone());
        delta.weights.push(error.weights.clone().apply_indexed(|i, j, _| {
            activations_in.get(j, 0) * error.error.get(i, 0)
        }));
        delta.biases.push(error.error.clone());

        while let Some(calculated_layer) = calculated_layers.pop() {
            let activations_in = calculated_layers.last().map(|v| v.activation.clone()).unwrap_or(example.input.clone());
            let this_error = (error.weights.transpose() * error.error).hmul(calculated_layer.activation_fn.activation_prime(calculated_layer.weighted_input));
            let weights = calculated_layer.weights.clone();
            delta.weights.push(weights.clone().apply_indexed(|i, j, _| {
                activations_in.get(j, 0) * this_error.get(i, 0)
            }));
            delta.biases.push(this_error.clone());
            error = Error {
                weights: calculated_layer.weights,
                error: this_error
            };
        }

        delta.weights.reverse();
        delta.biases.reverse();
        delta
    }

    pub fn test(&self, test: &Test, success_criteria: &impl SuccessCriteria) -> TestResult {
        let mut sum = 0.0;
        let mut successes = 0;
        for example in &test.examples {
            let result = self.evaluate(&example.input);
            if success_criteria.is_success(example, &result) {
                successes += 1;
            }
            sum += L::loss(example.output.clone(), result);
        }
        return TestResult {
            avg_cost: sum / test.examples.len() as f64,
            successes
        }
    }

    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        let mut current = input.clone();
        for layer in &self.layers {
            let weighted = layer.weights.clone() * current + layer.biases.clone();
            let activation = layer.activation_fn.activation(weighted);
            current = activation;
        }
        current
    }
}

#[derive(Clone, Copy)]
pub enum ActivationFn {
    Sigmoid,
}

impl ActivationFn {
    fn activation(&self, input: Matrix) -> Matrix {
        match self {
            Self::Sigmoid => input.apply(|n| 1f64 / (1f64 + f64::exp(-n)))
        }
    }

    fn activation_prime(&self, input: Matrix) -> Matrix {
        match self {
            Self::Sigmoid => input.apply(|n| (1f64 / (1f64 + f64::exp(-n))) * (1.0 - (1f64 / (1f64 + f64::exp(-n)))))
        }
    }
}

pub trait Loss {
    fn loss(expected: Matrix, got: Matrix) -> f64;
    fn loss_prime(expected: Matrix, got: Matrix) -> Matrix;
}

pub struct MSE;

impl Loss for MSE {
    fn loss(expected: Matrix, got: Matrix) -> f64 {
        return (got - expected).length() / 2.0;
    }

    fn loss_prime(expected: Matrix, got: Matrix) -> Matrix {
        return got - expected;
    }
}

pub struct TestResult {
    pub avg_cost: f64,
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

struct Delta {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>
}

pub struct Layer {
    pub neurons: usize,
    pub activation_fn: ActivationFn
}