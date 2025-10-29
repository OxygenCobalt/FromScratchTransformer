use std::collections::HashMap;

use crate::tensor::Tensor;

pub struct Vocabulary {
    mapping: HashMap<String, usize>
}

impl Vocabulary {
    pub fn train(corpus: Vec<String>) -> Self {
        let mut mapping = HashMap::new();
        for sentence in corpus {
            for word in sentence.split_whitespace() {
                mapping.insert(word.to_string(), mapping.len());
            }
        }
        Vocabulary { mapping }
    }

    pub fn test<T: Tensor>(&self, sentence: String) -> Option<T> {
        T::vector(sentence.split_whitespace().map(|word| self.mapping[word] as f64).collect::<Vec<f64>>())
    }
}

fn partials(corpus: Vec<String>) -> Vec<String> {
    corpus.iter().flat_map(|sentence| {
        // first split by whitespace
        let mut words = vec![];
        let mut current = String::new();
        for c in sentence.chars() {
            if c.is_whitespace() {
                words.push((current.clone(), Some(c)));
                current.clear();
                continue;
            }
            current.push(c);
        }
        words.push((current, None));
        // then make partials by accumulating words
        // i.e from "The cat sat." we get "The", "The cat", "The cat sat."
        // we could use split_whitespace but that destroys the whitespace context
        // which may be important for language modeling
        let mut partials = Vec::new();
        let mut current = String::new();
        for (word, sep) in words {
            current.push_str(&word);
            if let Some(c) = sep {
                current.push(c);
            }
            partials.push(current.clone());
        }
        partials
    }).collect()
}