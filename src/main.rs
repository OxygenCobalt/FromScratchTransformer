use std::fs::File;

use parquet::arrow::arrow_reader::ArrowReaderBuilder;

use crate::{matrix::{Matrix, Shape}, nn::{IOShape, Identity, NeuralNetwork, MSE}};

mod matrix;
mod nn;

fn main() {
    let train = File::open("./mnist/mnist/train-00000-of-00001.parquet").unwrap();
    let parquet = ArrowReaderBuilder::try_new(train).unwrap().build();
    let shape = IOShape {
        input_size: 784,
        output_size: 10
    };
    let nn = NeuralNetwork::<Identity, MSE>::new(shape, &[15]);
}
