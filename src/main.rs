use std::fs::File;

use crate::{dataset::{Example, Test}, matrix::Matrix, nn::{ActivationFn, Layer, LogLikelihood, NeuralNetwork, Reporting, SuccessCriteria, L2}};

mod dataset;
mod matrix;
mod nn;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

struct MnistReporting<'a> {
    test: &'a Test
}

impl <'a> Reporting for MnistReporting<'a> {
    fn data(&self) -> &Test {
        &self.test
    }


    fn report(&self, epoch: Option<u64>, result: nn::TestResult) {
        let percent_successful = (result.successes as f64 / self.test.examples.len() as f64) * 100.0;
        let phase = epoch.map(|e| format!["epoch {}", e + 1]).unwrap_or("init".to_owned()) ;
        println!("{}: loss = {}, {} / {} ({}%) successful", phase, result.avg_loss, result.successes, self.test.examples.len(), percent_successful)
    }
}

impl <'a> SuccessCriteria for MnistReporting<'a> {
    fn is_success(&self, example: &Example, output: &Matrix) -> bool {
        example.output.argmax() == output.argmax()
    }
}

fn main() {
    println!("loading data...");
    let mut mnist = dataset::mnist().unwrap();
    println!("loading done: {} train examples, {} test examples.", mnist.train.examples.len(), mnist.test.examples.len());
    println!("begin training...");
    let mut nn = NeuralNetwork::new(&[
        Layer { neurons: mnist.io_shape.in_size, activation_fn: ActivationFn::Sigmoid },
        Layer { neurons: 100, activation_fn: ActivationFn::Sigmoid },
        Layer { neurons: mnist.io_shape.out_size, activation_fn: ActivationFn::Softmax },
    ]);
    nn.train(
        &mut mnist.train,  
        60, 10, 0.1, 
        &LogLikelihood, 
        &L2 { lambda: 5.0 },
        Some(MnistReporting { test: &mnist.test }), 
        None
    ).unwrap();
    println!("saving result to file...");
    let mut output = File::create("mnist.nn").unwrap();
    nn.write(&mut output).unwrap();
}
