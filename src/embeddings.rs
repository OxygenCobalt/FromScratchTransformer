use std::{
    collections::{HashMap, HashSet},
    io::{self, Write},
    num::NonZeroUsize,
};

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use ordered_float::OrderedFloat;
use rand::{Rng, SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};

use crate::tensor::{CPUTensor, Tensor, TensorIO};

pub trait TrainEmbeddings {
    fn add(&mut self, word: String);
    fn build(self) -> Embeddings;
}

pub struct HashedEmbeddings {
    words: HashSet<String>,
    size: NonZeroUsize,
}

impl HashedEmbeddings {
    pub fn new(size: NonZeroUsize) -> Self {
        Self {
            words: HashSet::new(),
            size,
        }
    }
}

impl TrainEmbeddings for HashedEmbeddings {
    fn add(&mut self, word: String) {
        self.words.insert(word);
    }

    fn build(self) -> Embeddings {
        let train_progress = ProgressBar::new(self.words.len() as u64)
            .with_style(
                ProgressStyle::with_template(
                    "{prefix}: training {bar:40} {pos:>4}/{len:4} [{eta_precise}]",
                )
                .unwrap(),
            )
            .with_prefix("hashed embeddings".red().to_string());
        let word_amount = self.words.len();
        let hashes: HashMap<Vec<OrderedFloat<f64>>, String> = self
            .words
            .into_iter()
            .map(|w| {
                let mut seed = Sha256::new();
                seed.write(w.as_bytes()).unwrap();
                let mut rng = StdRng::from_seed(seed.finalize().into());
                let mut vector = Vec::new();
                for _ in 0..self.size.get() {
                    vector.push(OrderedFloat(rng.random_range(0.0..1.0)));
                }
                train_progress.inc(1);
                (vector, w)
            })
            .collect();
        train_progress.finish();
        if hashes.len() != word_amount {
            panic!("embedding collission! you should increase the embedding size.")
        }
        let to_vecs = hashes.iter().map(|(k, v)| (v.clone(), k.clone())).collect();
        Embeddings {
            to_vecs,
            to_words: hashes,
        }
    }
}

pub struct Embeddings {
    to_vecs: HashMap<String, Vec<OrderedFloat<f64>>>,
    to_words: HashMap<Vec<OrderedFloat<f64>>, String>,
}

impl Embeddings {
    pub fn to_vec<T: Tensor>(&self, word: impl AsRef<str>) -> Option<T> {
        self.to_vecs
            .get(word.as_ref())
            .and_then(|v| T::vector(v.iter().cloned().map(|x| x.0).collect::<Vec<f64>>()))
    }

    pub fn to_words<T: Tensor>(&self, vector: &T) -> Option<&str> {
        let key: Vec<OrderedFloat<f64>> = vector.iter().cloned().map(|x| OrderedFloat(x)).collect();
        self.to_words.get(&key).map(|s| s.as_str())
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"Embedngs")?;
        write.write_all(&self.to_vecs.len().to_le_bytes())?;
        let write_progress = ProgressBar::new(self.to_vecs.len() as u64)
            .with_style(
                ProgressStyle::with_template(
                    "{prefix}: writing {bar:40} {pos:>4}/{len:4} [{eta_precise}]",
                )
                .unwrap(),
            )
            .with_prefix("embeddings".red().to_string());
        for (word, vec) in &self.to_vecs {
            write.write_all(&word.len().to_le_bytes())?;
            write.write_all(word.as_bytes())?;
            CPUTensor::vector(vec.iter().cloned().map(|x| x.0).collect::<Vec<f64>>())
                .unwrap()
                .write(write)?;
            write_progress.inc(1);
        }
        write_progress.finish();
        Ok(())
    }
}
