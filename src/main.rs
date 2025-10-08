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
mod autograd;
mod embeddings;
mod loss;
mod mnist;
mod nn;
mod tensor;
mod wikitext;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    let threads = std::thread::available_parallelism().unwrap().get() / 4;
    println!("{}: using {} threads", "init".white(), threads);
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    let mnist = MNIST::<CPUTensor>::load(Path::new("data/mnist")).unwrap();
    let layers = Layers::new(vec![
        Layer::Conv2D {
            input_size: 28,
            field: Field { size: 5, stride: 1, padding: 0 },
            filters: 20,
            activation: Activation::ReLU,
        },
        Layer::Pool2D {
            input_size: 24,
            field: Field { size: 2, stride: 2, padding: 0 },
            filters: 20,
        },
        Layer::Dense {
            neurons: 10,
            activation: Activation::Softmax,
        },
    ])
    .unwrap();
    let checkpointing = Checkpoint::new(&layers, &mnist, Path::new("data/checkpoints/mnist"));
    let hyperparams = Hyperparams {
        epochs: 1,
        batch_size: 10,
        learning_rate: 0.1,
    };
    NeuralNetwork::train(&layers, &mnist, &mnist, &hyperparams, &LogLikelihood).unwrap();
}
