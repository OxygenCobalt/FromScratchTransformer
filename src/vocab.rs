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