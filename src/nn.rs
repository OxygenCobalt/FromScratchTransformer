use crate::matrix::{Dimensions, Matrix};

pub struct NeuralNetwork {
    layers: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(input: usize, layers: &[usize], output: usize) -> Self {
        Self { 
            layers: layers.iter().enumerate().map(|(i, s)| {
                Matrix::noisy(Dimensions { m: if i > 0 { layers[i - 1] } else { input }, n: if i < layers.len() - 1 { *s } else { output } })
            }).collect(),
        }
    }

    pub fn evaluate(&self, input: Matrix) -> Matrix {
        let mut current = input;
        for layer in &self.layers {
            current = current * layer.clone();
        }
        current
    }
}
