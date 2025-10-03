use std::path::Path;

use colored::Colorize;
use rayon::ThreadPoolBuilder;

use crate::{
    activation::Activation, loss::MSE, mnist::MNIST, nn::{Checkpoint, Hyperparams, Layer, Layers, NeuralNetwork}, tensor::{CPUTensor, Field}
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
            Layer::Conv2D {
                input_size: 28,
                field: Field { size: 4, stride: 1},
                filters: 20,
                activation: Activation::ReLU,
            },
            Layer::Pool2D {
                input_size: 20,
                field: Field { size: 2, stride: 1 }
            },
            Layer::Dense {
                neurons: 100,
                // TODO: softmax
                activation: Activation::Sigmoid,
            },
        ]
    ).unwrap();
    // let checkpointing = Checkpoint::new(&layers, &mnist, Path::new("data/checkpoints/mnist"));
    let hyperparams = Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0
    };
    NeuralNetwork::train(&layers, &mnist, &mnist, &hyperparams, &MSE).unwrap();
}
