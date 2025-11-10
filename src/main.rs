use std::path::{Path, PathBuf};

use colored::Colorize;
use rayon::ThreadPoolBuilder;

use crate::{
    dataset::{TestSet, TrainSet, mnist::Mnist, wikitext::{WikiText2, WikiText103}}, ml::{activation::Activation, language::{FixedSequencer, TokenizedExample, Tokenizer, WordTokenizer}, loss::{AccuracyOf, Loss, LossesOn}, nn::{Checkpoint, Hyperparams, Layer, Layers, NeuralNetwork}}, tensor::{Field, cpu::CPUTensor}
};

mod tensor;
mod dataset;
mod ml;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let experiment = match args.get(1) {
        Some(arg) => arg.clone(),
        None => {
            println!(
                "{}: please specify an experiment: shallow_mnist, dropout_mnist, conv_mnist, shallow_wikitext",
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
        ref exp if exp == "shallow_wikitext" => {
            shallow_wikitext();
        }
        _ => {
            println!("{}: unknown experiment '{}'", "error".red(), experiment);
        }
    }
}

fn shallow_mnist() {
    let mnist = Mnist(PathBuf::from("data/mnist"));
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
    let test = mnist.test().unwrap();
    let reporting = LossesOn::new(&test, &[Loss::MSE, Loss::Accuracy(AccuracyOf::Argmax)]);
    let checkpointing =
        Checkpoint::new(&layers, &reporting, Path::new("data/checkpoints/mnist/shallow"));
    let hyperparams = Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0,
    };
    NeuralNetwork::<CPUTensor>::train(&checkpointing, &checkpointing, &mnist.train().unwrap(), &hyperparams, Loss::MSE).unwrap();
}

fn dropout_mnist() {
    let mnist = Mnist(PathBuf::from("data/mnist"));
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
    let train = mnist.train().unwrap();
    let test = mnist.test().unwrap();
    let reporting = LossesOn::new(&test, &[Loss::MSE, Loss::Accuracy(AccuracyOf::Argmax)]);
    let checkpointing =
        Checkpoint::new(&layers, &reporting, Path::new("data/checkpoints/mnist/dropout"));
    let hyperparams = Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0,
    };
    NeuralNetwork::<CPUTensor>::train(&checkpointing, &checkpointing, &train, &hyperparams, Loss::MSE).unwrap();
}

fn conv_mnist() {
    let mnist = Mnist(PathBuf::from("data/mnist"));
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
    let train = mnist.train().unwrap();
    let test = mnist.test().unwrap();
    let reporting = LossesOn::new(&test, &[Loss::LogLikelihood, Loss::Accuracy(AccuracyOf::Argmax)]);
    let checkpointing = Checkpoint::new(&layers, &reporting, Path::new("data/checkpoints/mnist/conv"));
    let hyperparams = Hyperparams {
        epochs: 60,
        batch_size: 10,
        learning_rate: 0.1,
    };
    NeuralNetwork::<CPUTensor>::par_train(
        &checkpointing,
        &checkpointing,
        &train,
        &hyperparams,
        Loss::LogLikelihood,
    )
    .unwrap();
}

fn shallow_wikitext() {
    let wikitext = WikiText2(PathBuf::from("data/wikitext"));
    let train = wikitext.train().unwrap();
    let test = wikitext.test().unwrap();
    let tokenizer = WordTokenizer::train(&train, &test, None);
    let context = FixedSequencer::new(5);
    let train = train.map(|s| {
        context.split(&tokenizer.forward(&s).unwrap()).iter()
            .map(|s| TokenizedExample::from(s, &tokenizer))
            .collect::<Vec<TokenizedExample>>().into_iter()
    });
    let test = test.map(|s| {
        context.split(&tokenizer.forward(&s).unwrap()).iter()
            .map(|s| TokenizedExample::from(s, &tokenizer))
            .collect::<Vec<TokenizedExample>>().into_iter()
    });
    let reporting = LossesOn::new(&test, &[Loss::LogLikelihood]);
    let layers = Layers::new(vec![
        Layer::Embeddings { size: 60, vocab: tokenizer.vocab(), context: 5 },
        Layer::Dense { input_shape: None, neurons: 128, activation: Activation::Tanh },
        Layer::Dense { input_shape: None, neurons: tokenizer.vocab(), activation: Activation::Softmax },
    ]).unwrap();
    // let checkpointing = Checkpoint::new(&layers, &reporting, Path::new("data/wikitext"));
    let hyperparams = Hyperparams {
        epochs: 1,
        batch_size: 10,
        learning_rate: 0.1,
    };
    NeuralNetwork::<CPUTensor>::par_train(
        &layers,
        &reporting,
        &train,
        &hyperparams,
        Loss::LogLikelihood,
    )
    .unwrap();
    
}