use std::marker::PhantomData;

use crate::matrix::{Shape, Matrix};

pub struct NeuralNetwork<F: ActivationFunction> {
    input_shape: Shape,
    layers: Vec<HiddenLayer>,
    _activation: PhantomData<F>,
}

impl <F: ActivationFunction> NeuralNetwork<F> {
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
                
                HiddenLayer { weights: Matrix::noisy(weight_shape), biases: Matrix::noisy(bias_shape) }
            }).collect(),
            _activation: PhantomData,
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

pub trait ActivationFunction {
    fn invoke(input: &mut Matrix);
}

pub struct Identity;

impl ActivationFunction for Identity {
    fn invoke(_: &mut Matrix) {}
}

struct HiddenLayer {
    weights: Matrix,
    biases: Matrix
}