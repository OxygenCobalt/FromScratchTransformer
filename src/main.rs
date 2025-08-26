use std::{thread, time::Duration};

use crate::nn::{MSE, NeuralNetwork, Sigmoid};

mod dataset;
mod matrix;
mod nn;

fn main() {
    let mut mnist = dataset::mnist().unwrap();
    let mut nn = NeuralNetwork::<Sigmoid, MSE>::new(mnist.io_shape, &[15]);
    println!("init mse:    {:?}", nn.test(&mnist.test));
    for i in 0..16 {
        nn.train(&mut mnist.train, 0.1);
        println!("trained mse: {:?}", nn.test(&mnist.test));
    }
    // thread::sleep(Duration::from_millis(1000));
}
