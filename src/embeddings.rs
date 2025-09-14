use std::collections::HashMap;

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::tensor::Tensor;

pub struct LookupEmbeddings {
    mapping: HashMap<String, usize>,
    shape: usize
}

impl LookupEmbeddings {
    pub fn train<'a>(words: impl Iterator<Item=&'a str>, len: usize) -> Self {
        let mut mapping: HashMap<String, usize> = HashMap::new();
        let i = 0;
        let sgd_bar = ProgressBar::new(len as u64)
            .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] {msg}")
                            .unwrap()
                            .progress_chars("=> "))
            .with_prefix("training embeddings".purple().to_string());
        for word in words {
            if mapping.contains_key(word) {
                continue;
            }
            sgd_bar.set_message(format!["/ shape: {}", i + 1]);
            mapping.insert(word.to_string(), i);
        }
        Self {
            mapping,
            shape: i + 1
        }
    }

    pub fn test<T: Tensor>(&self, word: String) -> Option<T> {
        let idx = *self.mapping.get(&word)?;
        let mut embedding = vec![0.0; self.shape];
        embedding[idx] = 1.0;
        T::vector(embedding)
    }
}