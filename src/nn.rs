use std::marker::PhantomData;

use crate::matrix::{Shape, Matrix};

pub struct NeuralNetwork<F: ActivationFn, C: CostFn> {
    input_shape: Shape,
    layers: Vec<HiddenLayer>,
    _activation: PhantomData<F>,
    _cost: PhantomData<C>
}

impl <F: ActivationFn, C: CostFn> NeuralNetwork<F, C> {
    pub fn new(io_shape: IOShape, layers: &[usize]) -> Self {
        Self { 
            input_shape: Shape { m: 1, n: io_shape.input_size },
            layers: layers.iter().enumerate().map(|(i, s)| {
                let weight_shape = Shape { 
                    m: if i > 0 { layers[i - 1] } else { io_shape.input_size },
                    n: if i < layers.len() - 1 { *s } else { io_shape.output_size }
                };

                let bias_shape = Shape {
                    m: 1,
                    n: weight_shape.n  
                };
                
                HiddenLayer { weights: Matrix::noisy(weight_shape, 0.0..1.0), biases: Matrix::null(bias_shape) }
            }).collect(),
            _activation: PhantomData,
            _cost: PhantomData
        }
    }

    pub fn evaluate(&self, input: Matrix) -> Matrix {
        if self.input_shape != input.shape() {
            panic!("malformed input: need {:?}, got {:?}", self.input_shape, input.shape())
        }
        let mut current = input;
        for layer in &self.layers {
            unsafe {
                current.mul_assign_unchecked(&layer.weights);
                current.add_assign_unchecked(&layer.biases);
                current.apply(|n| {
                    1f64 / (1f64 + f64::exp(-n))
                });
                F::invoke(&mut current);
            }
        }
        current
    }
}

pub struct IOShape {
    pub input_size: usize,
    pub output_size: usize
}

pub trait ActivationFn {
    fn invoke(input: &mut Matrix);
}

pub struct Identity;

impl ActivationFn for Identity {
    fn invoke(_: &mut Matrix) {}
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
            sum += (result.expected_output - result.output).sum();
        }
        sum / (2.0 * n as f64)
    }
}

struct HiddenLayer {
    weights: Matrix,
    biases: Matrix
}