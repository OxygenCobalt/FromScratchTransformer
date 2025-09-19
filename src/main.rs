use std::{marker::PhantomData, num::NonZeroUsize, path::{Path, PathBuf}, sync::LazyLock};

use rand_distr::{Distribution, Normal};

use crate::{
    activation::Activation, embeddings::{HashedEmbeddings, TrainEmbeddings}, loss::MSE, nn::{Checkpoint, Hyperparams, Layer, NeuralNetwork}, tensor::{CPUTensor, Fill, Generate, Tensor}, wikitext::WikiText103
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

static NORMAL2: LazyLock<Normal<f64>> =
    std::sync::LazyLock::new(|| Normal::new(0.0, 1.0).unwrap());


fn main() {
    // for i in 0..1000 {
    //     let a = CPUTensor::tensor(Generate { shape: vec![1023, 381], with: || NORMAL2.sample(&mut rand::rng()) }).unwrap();
    //     let b = CPUTensor::tensor(Generate { shape: vec![381, 12], with: || NORMAL2.sample(&mut rand::rng()) }).unwrap();
    //     a.dot(&b, 1).unwrap();
    // }
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
        epochs: 1,
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
