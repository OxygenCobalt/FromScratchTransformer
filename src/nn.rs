use std::{f32::consts::E, io, marker::PhantomData, thread::current};

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
            weights: Matrix::null(Shape { m: hidden_layers[0],  n: io_shape.input_size }),
            biases: Matrix::null(Shape::vector(hidden_layers[0]))
        };
        let mut layers = vec![input_layer];
        for (i, _) in hidden_layers.iter().enumerate() {
            let in_shape = layers.last().unwrap().biases.shape();
            let out_size = hidden_layers.get(i + 1).cloned().unwrap_or_else(|| io_shape.output_size);
            let layer= HiddenLayer {
                weights: Matrix::null(Shape { m: out_size, n: in_shape.m,  }),
                biases: Matrix::null(Shape::vector(out_size))
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
        for (i, batch) in train.batch(batch_size).enumerate() {
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
                // println!("{:?}", delta_weights.clone());

                layer.weights -= delta_weights.scale(scale_by);
                layer.biases -= delta_biases.scale(scale_by);

                // println!("{:?}", layer.biases);
            }
        }
        println!()
    }

    fn backprop(&self, example: &Example) -> Delta {
        // backprop assumes that
        // 1. the cost function can be written as the avg of several training examples
        //    this is bc backprop allows us to compute delta(w,b) for a specific example only
        //    so we gotta have avg it out if we want to perform sgd
        // 2. the cost must be written as a fn of the neural networks outputs
        pub struct CalculatedLayer {
            weights: Matrix,
            biases: Matrix,
            weighted_input: Matrix,
            activation: Matrix 
        }
        let mut calculated_layers: Vec<CalculatedLayer> = Vec::new();
        let mut current = example.input.clone();
        for layer in &self.layers {
            let weighted_input = (layer.weights.clone() * current) + layer.biases.clone();
            let activation = F::activation(weighted_input.clone());
            current = activation.clone();
            calculated_layers.push(CalculatedLayer {
                weights: layer.weights.clone(),
                biases: layer.biases.clone(),
                weighted_input,
                activation
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
            error: (last.activation.clone() - example.output.clone())
            .hmul(F::activation_prime(last.weighted_input.clone()))
        };
        let activations_in = calculated_layers.last().map(|v| v.activation.clone()).unwrap_or(example.input.clone());
        delta.weights.push(error.weights.clone().apply_indexed(|i, j, _| {
            activations_in.get(j, 0) * error.error.get(i, 0)
        }));
        delta.biases.push(error.error.clone());
        // println!("{:?}", error.error);

        while let Some(calculated_layer) = calculated_layers.pop() {
            let activations_in = calculated_layers.last().map(|v| v.activation.clone()).unwrap_or(example.input.clone());
            let this_error = (error.weights.transpose() * error.error).hmul(F::activation_prime(calculated_layer.weighted_input));
            // println!("{:?}", this_error);
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
            let current_shape =current.shape();
            // println!("layer {}: {:?} x {:?} + {:?}", i, layer.weights.shape(), current_shape, layer.biases.shape());
            let weighted = layer.weights.clone() * current + layer.biases.clone();
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
            -(x_prime / (x*x))
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