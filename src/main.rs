#![feature(portable_simd)]
use std::path::{Path, PathBuf};

use colored::Colorize;
use rayon::ThreadPoolBuilder;

use crate::{
    dataset::{
        Example, TestSet, TrainSet, distill::Distill, mnist::Mnist, shakespeare::Shakespeare,
    },
    ml::{
        language::{FixedSequencer, TokenizedExample, Tokenizer, WordTokenizer},
        loss2,
        nn2::{self, ActivationFn},
    },
};

mod dataset;
mod ml;
mod tensor;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let experiment = match args.get(1) {
        Some(arg) => arg.clone(),
        None => {
            println!(
                "{}: please specify an experiment: shallow_mnist, dropout_mnist, conv_mnist, shallow_shakespeare",
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
        ref exp if exp == "shallow_shakespeare" => {
            shallow_shakespeare();
        }
        _ => {
            println!("{}: unknown experiment '{}'", "error".red(), experiment);
        }
    }
}

fn shallow_mnist() {
    let mnist = Mnist(PathBuf::from("data/mnist"));
    let layers = nn2::Layers::new(vec![
        nn2::Layer::Dense {
            input_shape: Some(vec![28, 28]),
            neurons: 100,
        },
        nn2::Layer::Activation {
            function: nn2::ActivationFn::Sigmoid,
            dropout: 0.01,
        },
        nn2::Layer::Dense {
            input_shape: None,
            neurons: 10,
        },
        nn2::Layer::Activation {
            function: nn2::ActivationFn::Sigmoid,
            dropout: 0.0,
        },
    ])
    .unwrap();
    let test = mnist.test().unwrap();
    let reporting = loss2::LossesOn::new(
        &test,
        &[
            loss2::Loss::MSE,
            loss2::Loss::Accuracy(loss2::AccuracyOf::Argmax),
        ],
    );
    let checkpointing = nn2::Checkpoint::new(
        &layers,
        &reporting,
        Path::new("data/checkpoints/mnist/shallow"),
    );
    let hyperparams = nn2::Hyperparams {
        epochs: 30,
        batch_size: 10,
        learning_rate: 3.0,
    };
    nn2::NeuralNetwork::train(
        &checkpointing,
        &checkpointing,
        &mnist.train().unwrap(),
        &hyperparams,
        loss2::Loss::MSE,
    )
    .unwrap();
}

fn dropout_mnist() {
    // let mnist = Mnist(PathBuf::from("data/mnist"));
    // let layers = Layers::new(vec![
    //     Layer::Dropout {
    //         input_shape: Some(vec![28, 28]),
    //         neurons: 100,
    //         rate: 0.01,
    //         activation: Activation::Sigmoid,
    //     },
    //     Layer::Dense {
    //         input_shape: None,
    //         neurons: 10,
    //         activation: Activation::Sigmoid,
    //     },
    // ])
    // .unwrap();
    // let train = mnist.train().unwrap();
    // let test = mnist.test().unwrap();
    // let reporting = LossesOn::new(&test, &[Loss::MSE, Loss::Accuracy(AccuracyOf::Argmax)]);
    // let checkpointing = Checkpoint::new(
    //     &layers,
    //     &reporting,
    //     Path::new("data/checkpoints/mnist/dropout"),
    // );
    // let hyperparams = Hyperparams {
    //     epochs: 30,
    //     batch_size: 10,
    //     learning_rate: 3.0,
    // };
    // NeuralNetwork::<CPUTensor>::train(
    //     &checkpointing,
    //     &checkpointing,
    //     &train,
    //     &hyperparams,
    //     Loss::MSE,
    // )
    // .unwrap();
}

fn conv_mnist() {
    // let mnist = Mnist(PathBuf::from("data/mnist"));
    // let layers = Layers::new(vec![
    //     Layer::Conv2D {
    //         input_size: 28,
    //         field: Field {
    //             size: 5,
    //             stride: 1,
    //             padding: 0,
    //         },
    //         filters: 20,
    //         activation: Activation::ReLU,
    //     },
    //     Layer::Pool2D {
    //         input_size: 24,
    //         field: Field {
    //             size: 2,
    //             stride: 2,
    //             padding: 0,
    //         },
    //         filters: 20,
    //     },
    //     Layer::Dense {
    //         input_shape: None,
    //         neurons: 10,
    //         activation: Activation::Softmax,
    //     },
    // ])
    // .unwrap();
    // let train = mnist.train().unwrap();
    // let test = mnist.test().unwrap();
    // let reporting = LossesOn::new(
    //     &test,
    //     &[Loss::LogLikelihood, Loss::Accuracy(AccuracyOf::Argmax)],
    // );
    // let checkpointing = Checkpoint::new(
    //     &layers,
    //     &reporting,
    //     Path::new("data/checkpoints/mnist/conv"),
    // );
    // let hyperparams = Hyperparams {
    //     epochs: 60,
    //     batch_size: 10,
    //     learning_rate: 0.1,
    // };
    // NeuralNetwork::<CPUTensor>::par_train(
    //     &checkpointing,
    //     &checkpointing,
    //     &train,
    //     &hyperparams,
    //     Loss::LogLikelihood,
    // )
    // .unwrap();
}

fn shallow_shakespeare() {
    let wikitext = Shakespeare(PathBuf::from("data/tiny_shakespeare"));
    let distill = Distill(1.0);
    let train = distill.train(wikitext.train().unwrap());
    let test = distill.test(wikitext.test().unwrap());
    println!("{}", train.iter().next().unwrap());
    let tokenizer = WordTokenizer::train(&train, &test, None);
    let context = FixedSequencer::new(5);
    let train = train.map(|s| {
        let stripped = s.replace("\n", " ");
        context
            .split(&tokenizer.forward(&stripped).unwrap())
            .iter()
            .map(|s| TokenizedExample::from(s, &tokenizer))
            .collect::<Vec<TokenizedExample>>()
            .into_iter()
    });
    let test = test.map(|s| {
        let stripped = s.replace("\n", " ");
        context
            .split(&tokenizer.forward(&stripped).unwrap())
            .iter()
            .map(|s| TokenizedExample::from(s, &tokenizer))
            .collect::<Vec<TokenizedExample>>()
            .into_iter()
    });
    let reporting = loss2::LossesOn::new(
        &test,
        &[
            loss2::Loss::LogLikelihood,
            loss2::Loss::Accuracy(loss2::AccuracyOf::Argmax),
        ],
    );
    let layers = nn2::Layers::new(vec![
        nn2::Layer::Embeddings {
            size: 60,
            vocab: tokenizer.vocab(),
            context: 5,
        },
        nn2::Layer::Dense {
            input_shape: None,
            neurons: 128,
        },
        nn2::Layer::Activation {
            function: ActivationFn::Tanh,
            dropout: 0.00,
        },
        nn2::Layer::Dense {
            input_shape: None,
            neurons: tokenizer.vocab(),
        },
        nn2::Layer::Activation {
            function: ActivationFn::Softmax,
            dropout: 0.00,
        },
    ])
    .unwrap();
    // let checkpointing = Checkpoint::new(&layers, &reporting, Path::new("data/wikitext"));
    let hyperparams = nn2::Hyperparams {
        epochs: 1,
        batch_size: 10,
        learning_rate: 0.01,
    };
    nn2::NeuralNetwork::train(
        &layers,
        &reporting,
        &train,
        &hyperparams,
        loss2::Loss::LogLikelihood,
    )
    .unwrap();
}
