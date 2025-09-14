use std::{marker::PhantomData, path::{Path, PathBuf}};

use crate::{
    activation::Activation, loss::MSE, nn::{Checkpoint, Hyperparams, Layer, NeuralNetwork}, obw::one_billion_words, tensor::CPUTensor
};

mod activation;
mod autograd;
mod loss;
mod nn;
mod tensor;
mod mnist;
mod obw;
mod embeddings;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    let obw = one_billion_words::<CPUTensor>(Path::new("data/1-billion-word-language-modeling-benchmark-r13output")).unwrap();
    println!("{}", obw.train.len());
    // let mnist = mnist::mnist::<CPUTensor>(Path::new("data/mnist")).unwrap();
    // let mut nn = NeuralNetwork::new(&[
    //     Layer::Dense {
    //         neurons: 784,
    //         activation: Activation::Sigmoid,
    //     },
    //     Layer::Dropout {
    //         neurons: 100,
    //         activation: Activation::Sigmoid,
    //         rate: 0.01
    //     },
    //     Layer::Dense {
    //         neurons: 10,
    //         activation: Activation::Sigmoid,
    //     },
    // ]);
    // let hyperparams = Hyperparams {
    //     epochs: 30,
    //     batch_size: 10,
    //     learning_rate: 3.0
    // };
    // nn.train(
    //     &Checkpoint::<_, _>(&mnist, PathBuf::from("data/mnist_checkpts"), PhantomData),
    //     hyperparams,
    //     &MSE,
    // )
    // .unwrap();
}
