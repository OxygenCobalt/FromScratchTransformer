use std::path::Path;

use colored::Colorize;
use rayon::ThreadPoolBuilder;

use crate::{
    activation::Activation, loss::MSE, mnist::MNIST, nn::{Checkpoint, Hyperparams, Layer, Layers, NeuralNetwork}, tensor::CPUTensor
};

mod activation;
mod autograd;
mod loss;
mod nn;
mod tensor;
mod mnist;
mod embeddings;
mod wikitext;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;


fn main() {
    let threads = std::thread::available_parallelism().unwrap().get() / 4;
    println!("{}: using {} threads", "init".white(), threads);
    ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();

    let mnist = MNIST::<CPUTensor>::load(Path::new("data/mnist")).unwrap();
    let layers = Layers::new(
vec![
            Layer::Dense {
                neurons: 784,
                activation: Activation::Sigmoid,
            },
            Layer::Dropout {
                neurons: 100,
                activation: Activation::Sigmoid,
                rate: 0.01
            },
            Layer::Dense {
                neurons: 10,
                activation: Activation::Sigmoid,
            },
        ]
    ).unwrap();
    let checkpointing = Checkpoint::new(&layers, &mnist, Path::new("data/checkpoints/mnist"));
    let hyperparams = Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0
    };
    NeuralNetwork::train(&checkpointing, &checkpointing, &mnist, &hyperparams, &MSE).unwrap();
}
