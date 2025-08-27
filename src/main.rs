use crate::{dataset::{Example, Test}, matrix::Matrix, nn::{ActivationFn, Layer, NeuralNetwork, Reporting, SuccessCriteria, MSE}};

mod dataset;
mod matrix;
mod nn;

struct MnistReporting {
    test: Test
}

impl Reporting for MnistReporting {
    fn data(&self) -> &Test {
        &self.test
    }


    fn report(&self, epoch: Option<u64>, result: nn::TestResult) {
        let percent_successful = (result.successes as f64 / self.test.examples.len() as f64) * 100.0;
        let phase = epoch.map(|e| format!["epoch {}", e + 1]).unwrap_or("init".to_owned()) ;
        println!("{}: loss = {}, {} / {} ({}%) successful", phase, result.avg_cost, result.successes, self.test.examples.len(), percent_successful)
    }
}

impl SuccessCriteria for MnistReporting {
    fn is_success(&self, example: &Example, output: &Matrix) -> bool {
        example.output.argmax() == output.argmax()
    }
}

fn main() {
    println!("loading data...");
    let mut mnist = dataset::mnist().unwrap();
    println!("loading done: {} train examples, {} test examples.", mnist.train.examples.len(), mnist.test.examples.len());
    println!("begin training...");
    let mut nn = NeuralNetwork::<MSE>::new(&[
        Layer { neurons: mnist.io_shape.in_size, activation_fn: ActivationFn::Sigmoid },
        Layer { neurons: 30, activation_fn: ActivationFn::Sigmoid },
        Layer { neurons: mnist.io_shape.out_size, activation_fn: ActivationFn::Sigmoid },
    ]);
    nn.train(&mut mnist.train,  30, 10, 3.0, Some(MnistReporting { test: mnist.test }));
}
