use crate::{matrix::{Shape, Matrix}, nn::{IOShape, Identity, NeuralNetwork}};

mod matrix;
mod nn;

fn main() {
    let shape = IOShape {
        input_size: 4,
        output_size: 3
    };
    let nn = NeuralNetwork::<Identity>::new(shape, &[10, 100, 1000, 10000, 100, 10]);

    let inputs = Matrix::noisy(Shape { m: 1, n: 4 });
    println!("{:?}", nn.evaluate(inputs));
}
