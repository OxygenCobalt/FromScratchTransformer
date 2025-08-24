use crate::{matrix::{Dimensions, Matrix}, nn::NeuralNetwork};

mod matrix;
mod nn;

fn main() {
    let inputs = Matrix::noisy(Dimensions { m: 1, n: 4 });
    let nn = NeuralNetwork::new(4, &[10, 100, 1000, 10000, 100, 10], 3);
    println!("{:?}", nn.evaluate(inputs));
}
