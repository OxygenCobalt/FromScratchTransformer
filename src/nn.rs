use std::{io, marker::PhantomData, thread::current};

use crate::{
    dataset::{Example, Test, Train},
    matrix::{Matrix, Shape},
};

pub struct NeuralNetwork<F: ActivationFn, C: CostFn> {
    input_shape: Shape,
    layers: Vec<HiddenLayer>,
    _activation: PhantomData<F>,
    _cost: PhantomData<C>,
}

impl<F: ActivationFn, C: CostFn> NeuralNetwork<F, C> {
    pub fn new(io_shape: IOShape, hidden_layers: &[usize]) -> Self {
        let input_layer = HiddenLayer { 
            weights: Matrix::noisy(Shape { m: io_shape.input_size, n: hidden_layers[0] }, 0.0..1.0),
            biases: Matrix::noisy(Shape { m: 1, n: hidden_layers[0] }, 0.0..1.0)
        };
        let mut layers = vec![input_layer];
        for (i, _) in hidden_layers.iter().enumerate() {
            let in_shape = layers.last().unwrap().biases.shape();
            let out_size = hidden_layers.get(i + 1).cloned().unwrap_or_else(|| io_shape.output_size);
            let layer= HiddenLayer {
                weights: Matrix::noisy(Shape { m: in_shape.n, n: out_size }, 0.0..1.0),
                biases: Matrix::noisy(Shape { m: in_shape.m, n: out_size }, 0.0..1.0)
            };
            layers.push(layer);
        }

        Self {
            input_shape: Shape::vector(io_shape.input_size),
            layers: layers,
            _activation: PhantomData,
            _cost: PhantomData,
        }
    }

    pub fn train(&mut self, train: &mut Train, learning_rate: f64) {
        let batch_size = 10;
        for batch in train.batch(batch_size) {
            // TODO: Switch to matrix slicing across a 3d matrix, requires tensors :(
            let mut sum_delta_weights: Vec<Matrix> = self.layers.iter().map(|l| Matrix::null(l.weights.shape())).collect();
            let mut sum_delta_biases: Vec<Matrix> = self.layers.iter().map(|l| Matrix::null(l.biases.shape())).collect();
            for example in batch {
                let delta = self.backprop(example);
                // println!("{:?}", delta.biases.iter().map(|l| l.shape()).collect::<Vec<_>>());
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
                // println!("{:?}", layer.biases);
                layer.weights -= delta_weights.scale(scale_by);
                layer.biases -= delta_biases.scale(scale_by);
                // println!("{:?}", layer.biases);
            }
        }
    }

    fn backprop(&self, example: &Example) -> Delta {
        // backprop assumes that
        // 1. the cost function can be written as the avg of several training examples
        //    this is bc backprop allows us to compute delta(w,b) for a specific example only
        //    so we gotta have avg it out if we want to perform sgd
        // 2. the cost must be written as a fn of the neural networks outputs
        let mut weighted_inputs: Vec<Matrix> = Vec::new();
        let mut activations: Vec<Matrix> = Vec::new();
        let mut current = example.input.clone();
        for layer in &self.layers {
            let weighted = (current * layer.weights.clone()) + layer.biases.clone();
            weighted_inputs.push(weighted.clone());
            let activation = F::activation(weighted);
            activations.push(activation.clone());
            current = activation;
        }
        let mut delta = Delta {
            weights: Vec::new(),
            biases: Vec::new()
        };

        let mut errors = vec![Matrix::null(Shape::vector(1)); self.layers.len()];
        for i in (0..self.layers.len()).rev() {
            let weighted_input = weighted_inputs[i].clone();
            let activations_out = activations[i].clone();
            let next_idx = i + 1;
            let error = if next_idx < self.layers.len() {
                (self.layers[next_idx].weights.clone() * errors[next_idx].clone().transpose()).transpose().hmul(F::activation_prime(weighted_input))
            } else {
                C::cost_prime(example.output.clone(), activations_out).hmul(F::activation_prime(weighted_input))
            };
            let activations_in = if i > 0 {
                activations[i - 1].clone()
            } else {
                example.input.clone()
            };
            delta.weights.push(activations_in.transpose() * error.clone());
            delta.biases.push(error.clone());
            errors[i] = error;
        }

        delta.weights.reverse();
        delta.biases.reverse();
        delta
    }

    pub fn test(&self, test: &Test) -> f64 {
        let mut sum = 0.0;
        for (i, example) in test.examples.iter().enumerate() {
            sum += C::cost(example.output.clone(), self.evaluate(&example.input));
        }
        return sum / test.examples.len() as f64;
    }

    pub fn evaluate(&self, input: &Matrix) -> Matrix {
        if self.input_shape != input.shape() {
            panic!(
                "malformed input: need {:?}, got {:?}",
                self.input_shape,
                input.shape()
            )
        }
        let mut current = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let weighted = current * layer.weights.clone() + layer.biases.clone();
            let activation = F::activation(weighted);
            current = activation;
        }
        current
    }
}

pub struct IOShape {
    pub input_size: usize,
    pub output_size: usize,
}

pub trait ActivationFn {
    fn activation(input: Matrix) -> Matrix;
    fn activation_prime(input: Matrix) -> Matrix;
}

pub struct Sigmoid;

impl ActivationFn for Sigmoid {
    fn activation(input: Matrix) -> Matrix {
        input.apply(|n| 1f64 / (1f64 + f64::exp(-n)))
    }

    fn activation_prime(input: Matrix) -> Matrix {
        input.apply(|n| {
            let x = 1f64 + f64::exp(-n);
            let x_prime = -f64::exp(-n);
            -(1.0 / x*x) * x_prime
        })
    }
}

pub struct Result {
    input: Matrix,
    output: Matrix,
    expected_output: Matrix,
}

pub trait CostFn {
    fn cost(expected: Matrix, got: Matrix) -> f64;
    fn cost_prime(expected: Matrix, got: Matrix) -> Matrix;
}

pub struct MSE;

impl CostFn for MSE {
    fn cost(expected: Matrix, got: Matrix) -> f64 {
        return (expected - got).length() / 2.0;
    }

    fn cost_prime(expected: Matrix, got: Matrix) -> Matrix {
        return expected - got;
    }
}

struct HiddenLayer {
    weights: Matrix,
    biases: Matrix,
}

struct Delta {
    weights: Vec<Matrix>,
    biases: Vec<Matrix>
}