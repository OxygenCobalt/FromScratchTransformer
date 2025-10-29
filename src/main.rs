use std::path::Path;

use colored::Colorize;
use rayon::ThreadPoolBuilder;

use crate::{
    activation::Activation,
    loss::{LogLikelihood, MSE},
    mnist::MNIST,
    nn::{Checkpoint, Hyperparams, Layer, Layers, NeuralNetwork},
    tensor::{CPUTensor, Field},
};

mod activation;
mod vocab;
mod loss;
mod mnist;
mod nn;
mod tensor;
mod wikitext;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let experiment = match args.get(1) {
        Some(arg) => arg.clone(),
        None => {
            println!(
                "{}: please specify an experiment: shallow_mnist, dropout_mnist, conv_mnist",
                "error".red()
            );
            return;
        }
    };

    let threads = std::thread::available_parallelism().unwrap().get();
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    println!("{}: using {} threads", "init".white(), threads);

    match experiment {
        ref exp if exp == "shallow_mnist" => {
            shallow_mnist();
        }
        ref exp if exp == "dropout_mnist" => {
            dropout_mnist();
        }
        ref exp if exp == "conv_mnist" => {
            conv_mnist();
        }
        _ => {
            println!("{}: unknown experiment '{}'", "error".red(), experiment);
        }
    }
}

fn shallow_mnist() {
    let mnist = MNIST::<CPUTensor>::load(Path::new("data/mnist")).unwrap();
    let layers = Layers::new(vec![
        Layer::Dense {
            input_shape: Some(vec![28, 28]),
            neurons: 100,
            activation: Activation::Sigmoid,
        },
        Layer::Dense {
            input_shape: None,
            neurons: 10,
            activation: Activation::Sigmoid,
        },
    ])
    .unwrap();
    let checkpointing =
        Checkpoint::new(&layers, &mnist, Path::new("data/checkpoints/mnist/shallow"));
    let hyperparams = Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0,
    };
    NeuralNetwork::train(&checkpointing, &checkpointing, &mnist, &hyperparams, &MSE).unwrap();
}

fn dropout_mnist() {
    let mnist = MNIST::<CPUTensor>::load(Path::new("data/mnist")).unwrap();
    let layers = Layers::new(vec![
        Layer::Dropout {
            input_shape: Some(vec![28, 28]),
            neurons: 100,
            rate: 0.01,
            activation: Activation::Sigmoid,
        },
        Layer::Dense {
            input_shape: None,
            neurons: 10,
            activation: Activation::Sigmoid,
        },
    ])
    .unwrap();
    let checkpointing =
        Checkpoint::new(&layers, &mnist, Path::new("data/checkpoints/mnist/dropout"));
    let hyperparams = Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0,
    };
    NeuralNetwork::train(&checkpointing, &checkpointing, &mnist, &hyperparams, &MSE).unwrap();
}

fn conv_mnist() {
    let mnist = MNIST::<CPUTensor>::load(Path::new("data/mnist")).unwrap();
    let layers = Layers::new(vec![
        Layer::Conv2D {
            input_size: 28,
            field: Field {
                size: 5,
                stride: 1,
                padding: 0,
            },
            filters: 20,
            activation: Activation::ReLU,
        },
        Layer::Pool2D {
            input_size: 24,
            field: Field {
                size: 2,
                stride: 2,
                padding: 0,
            },
            filters: 20,
        },
        Layer::Dense {
            input_shape: None,
            neurons: 10,
            activation: Activation::Softmax,
        },
    ])
    .unwrap();
    let checkpointing = Checkpoint::new(&layers, &mnist, Path::new("data/checkpoints/mnist/conv"));
    let hyperparams = Hyperparams {
        epochs: 60,
        batch_size: 10,
        learning_rate: 0.1,
    };
    NeuralNetwork::par_train(
        &checkpointing,
        &checkpointing,
        &mnist,
        &hyperparams,
        &LogLikelihood,
    )
    .unwrap();
}
