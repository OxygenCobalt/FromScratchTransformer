use crate::{matrix::{Matrix, Shape}, nn::{IOShape, Identity, NeuralNetwork, MSE}};

mod matrix;
mod nn;

fn main() {
    let shape = IOShape {
        input_size: 4,
        output_size: 3
    };
    let nn = NeuralNetwork::<Identity, MSE>::new(shape, &[3, 100, 3]);

    let inputs = Matrix::noisy(Shape { m: 1, n: 4 }, 0.0..1.0);
    println!("{:?}", nn.evaluate(inputs));
}
