use crate::nn::{MSE, NeuralNetwork, Sigmoid};

mod dataset;
mod matrix;
mod nn;

fn main() {
    let mut mnist = dataset::mnist().unwrap();
    let mut nn = NeuralNetwork::<Sigmoid, MSE>::new(mnist.io_shape, &[15]);
    nn.train(&mut mnist.train, 0.01);
}
