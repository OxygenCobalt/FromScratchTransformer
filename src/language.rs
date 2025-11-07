use std::collections::HashMap;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{dataset::{Example, Test, Train, Validation, ValidationSet}, tensor::Tensor};

pub trait Tokenizer {
    fn train(train: &Train<String>, test: &Test<String>, validation: Option<&Validation<String>>) -> Self;
    fn forward(&self, string: &str) -> Option<Vec<usize>>;
    fn backward(&self, tokens: &[usize]) -> Option<String>;
    fn vocab(&self) -> usize;
}

pub struct WordTokenizer {
    forward: HashMap<String, usize>,
    backward: Vec<String>
}

impl Tokenizer for WordTokenizer {
    fn train(train: &Train<String>, test: &Test<String>, validation: Option<&Validation<String>>) -> Self {
        let train_bar = ProgressBar::new((train.len() + test.len() + validation.map(|v| v.len()).unwrap_or(0)) as u64)
            .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] / vocab size = {msg}")
                            .unwrap()
                            .progress_chars("=> "))
            .with_prefix("wordtok".red().to_string());
        let mut forward = HashMap::new();
        let mut backward = Vec::new();
        for sentence in train.iter().chain(test.iter()).chain(validation.into_iter().flat_map(|v| v.iter())) {
            for word in sentence.split_whitespace() {
                if !forward.contains_key(word) {
                    let index = backward.len();
                    forward.insert(word.to_string(), index);
                    backward.push(word.to_string());
                    train_bar.set_message(backward.len().to_string());
                }
            }
            train_bar.inc(1);
        }
        train_bar.finish();
        Self { forward, backward }
    }

    fn forward(&self, string: &str) -> Option<Vec<usize>> {
        let mut tokens = vec![];
        for word in string.split_whitespace() {
            match self.forward.get(word) {
                Some(&index) => tokens.push(index),
                None => return None
            }
        }
        Some(tokens)
    }

    fn backward(&self, tokens: &[usize]) -> Option<String> {
        let mut words = vec![];
        for &token in tokens {
            match self.backward.get(token) {
                Some(word) => words.push(word.clone()),
                None => return None
            }
        }
        Some(words.join(" "))
    }

    fn vocab(&self) -> usize {
        self.backward.len()
    }
}

pub struct TokenizedExample {
    input: Vec<usize>,
    output: usize,
    // slightly bad duplication but better than carrying around vocab refs everywhere
    vocab: usize,
}

impl TokenizedExample {
    pub fn from(tokens: &[usize], tokenizer: &impl Tokenizer) -> Self {
        Self {
            input: tokens[..tokens.len() - 1].to_vec(),
            output: tokens[tokens.len() - 1],
            vocab: tokenizer.vocab(),
        }
    }
}

impl <T: Tensor> Example<T> for TokenizedExample {
    fn input(&self) -> T {
        T::vector(self.input.iter().map(|i| *i as f64).collect::<Vec<f64>>()).unwrap()
    }

    fn output(&self) -> T {
        let mut output = vec![0.0; self.vocab];
        output[self.output] = 1.0;
        T::vector(output).unwrap()
    }
}

pub struct FixedSequencer {
    context: usize
}

impl FixedSequencer {
    pub fn new(context: usize) -> Self {
        Self { context }
    }

    pub fn split(&self, tokens: &[usize]) -> Vec<Vec<usize>> {
        let mut sequences = vec![];
        if tokens.len() < self.context + 1 {
            return sequences;
        }
        for i in 0..=(tokens.len() - (self.context + 1)) {
            sequences.push(tokens[i..(i + self.context + 1)].to_vec());
        }
        sequences
    }
}
