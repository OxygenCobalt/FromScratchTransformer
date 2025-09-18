use std::{marker::PhantomData, num::NonZeroUsize, path::{Path, PathBuf}};

use crate::{
    activation::Activation, embeddings::{HashedEmbeddings, TrainEmbeddings}, loss::MSE, nn::{Checkpoint, Hyperparams, Layer, NeuralNetwork}, tensor::CPUTensor, wikitext::WikiText103
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
    // let wt = WikiText103::load(&Path::new("./data/wikitext")).unwrap();
    // let mut train_embeddings = HashedEmbeddings::new(NonZeroUsize::new(50).unwrap());
    // for word in &wt.train {
    //     train_embeddings.add(word.clone());
    // }
    // for word in &wt.test {
    //     train_embeddings.add(word.clone());
    // }
    // let embeddings = train_embeddings.build();

    let mnist = mnist::mnist::<CPUTensor>(Path::new("data/mnist")).unwrap();
    let mut nn = NeuralNetwork::new(&[
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
    ]);
    let hyperparams = Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0
    };
    nn.train(
        &Checkpoint::<_, _>(&mnist, PathBuf::from("data/mnist_checkpts"), PhantomData),
        hyperparams,
        &MSE,
    )
    .unwrap();
}
