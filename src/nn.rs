use std::marker::PhantomData;

use crate::{
    dataset::{Example, Train},
    matrix::{Matrix, Shape},
};

pub struct NeuralNetwork<F: ActivationFn, C: CostFn> {
    input_shape: Shape,
    layers: Vec<HiddenLayer>,
    _activation: PhantomData<F>,
    _cost: PhantomData<C>,
}

impl<F: ActivationFn, C: CostFn> NeuralNetwork<F, C> {
    pub fn new(io_shape: IOShape, layers: &[usize]) -> Self {
        Self {
            input_shape: Shape {
                m: io_shape.input_size,
                n: 1,
            },
            layers: layers
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let weight_shape = Shape {
                        m: if i < layers.len() - 1 {
                            *s
                        } else {
                            io_shape.output_size
                        },
                        n: if i > 0 {
                            layers[i - 1]
                        } else {
                            io_shape.input_size
                        },
                    };

                    let bias_shape = Shape::vector(weight_shape.n);

                    HiddenLayer {
                        weights: Matrix::noisy(weight_shape, 0.0..1.0),
                        biases: Matrix::null(bias_shape),
                    }
                })
                .collect(),
            _activation: PhantomData,
            _cost: PhantomData,
        }
    }

    pub fn train(&mut self, train: &mut Train, learning_rate: f64) {
        let batch_size = 10;
        for batch in train.batch(batch_size) {
            // TODO: Switch to matrix slicing across a 3d matrix, requires tensors :(
            let mut sum_delta_weights: Vec<Matrix> = Vec::new();
            let mut sum_delta_biases: Vec<Matrix> = Vec::new();
            for example in batch {
                let (delta_weights, delta_biases) = self.backprop(example);
                for (a, b) in sum_delta_weights.iter_mut().zip(delta_weights.into_iter()) {
                    *a += b;
                }
                for (a, b) in sum_delta_biases.iter_mut().zip(delta_biases.into_iter()) {
                    *a += b;
                }
            }
            let scale_by = learning_rate / batch_size as f64;
            for ((layer, mut delta_weights), mut delta_biases) in self
                .layers
                .iter_mut()
                .zip(sum_delta_weights.into_iter())
                .zip(sum_delta_biases)
            {
                delta_weights.scale(scale_by);
                delta_biases.scale(scale_by);
                layer.weights -= delta_weights;
                layer.biases -= delta_biases;
            }
        }
    }

    fn backprop(&self, example: &Example) -> (Vec<Matrix>, Vec<Matrix>) {
        todo!();
    }

    pub fn evaluate(&self, input: Matrix) -> Matrix {
        if self.input_shape != input.shape() {
            panic!(
                "malformed input: need {:?}, got {:?}",
                self.input_shape,
                input.shape()
            )
        }
        let mut current = input;
        for layer in &self.layers {
            unsafe {
                current.mul_assign_unchecked(&layer.weights);
                current.add_assign_unchecked(&layer.biases);
                F::invoke(&mut current);
            }
        }
        current
    }
}

pub struct IOShape {
    pub input_size: usize,
    pub output_size: usize,
}

pub trait ActivationFn {
    fn invoke(input: &mut Matrix);
}

pub struct Sigmoid;

impl ActivationFn for Sigmoid {
    fn invoke(outputs: &mut Matrix) {
        outputs.apply(|n| 1f64 / (1f64 + f64::exp(-n)));
    }
}

pub struct Result {
    input: Matrix,
    output: Matrix,
    expected_output: Matrix,
}

pub trait CostFn {
    fn invoke(results: Vec<Result>) -> f64;
}

pub struct MSE;

impl CostFn for MSE {
    fn invoke(results: Vec<Result>) -> f64 {
        let mut sum = 0.0;
        let n = results.len() as f64;
        for result in results {
            sum += (result.expected_output - result.output).length();
        }
        sum / (2.0 * n as f64)
    }
}

struct HiddenLayer {
    weights: Matrix,
    biases: Matrix,
}
