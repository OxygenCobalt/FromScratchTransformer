use std::{thread, time::Duration};

use crate::nn::{MSE, NeuralNetwork, Sigmoid};

mod dataset;
mod matrix;
mod nn;

fn main() {
    let mut mnist = dataset::mnist().unwrap();
    let mut nn = NeuralNetwork::<Sigmoid, MSE>::new(mnist.io_shape, &[100]);
    let mut successes = 0;
    for example in &mnist.test.examples {
        let result = nn.evaluate(&example.input);
        if example.output.top() == result.top() {
            successes += 1;
        }
    };
    println!("init, {}% success", (successes as f64 / mnist.test.examples.len() as f64) * 100.0);
    for i in 0..30 {
        nn.train(&mut mnist.train, 3.0);
        successes = 0;
        for example in &mnist.test.examples {
            let result = nn.evaluate(&example.input);
            if example.output.top() == result.top() {
                successes += 1;
            }
        };
        println!("epoch {} done, {}% success", i, (successes as f64 / mnist.test.examples.len() as f64) * 100.0)
    }
    // thread::sleep(Duration::from_millis(1000));
}
