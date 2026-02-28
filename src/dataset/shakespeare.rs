use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
};

use crate::dataset::{Test, TestSet, Train, TrainSet, Validation, ValidationSet};

pub struct Shakespeare(pub PathBuf);

fn load_shakespeare(path: &Path) -> io::Result<Vec<String>> {
    let file = File::open(path)?;
    let buf_reader = BufReader::new(file);
    let mut sequences = Vec::new();
    let mut current_sequence = String::new();
    for line in buf_reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            if !current_sequence.is_empty() {
                sequences.push(current_sequence.clone());
                current_sequence.clear();
            }
        } else {
            current_sequence.push_str(&line);
            current_sequence.push('\n');
            continue;
        }
    }
    if !current_sequence.is_empty() {
        sequences.push(current_sequence);
    }
    Ok(sequences)
}

impl TrainSet for Shakespeare {
    type Example = String;
    fn train(&self) -> io::Result<Train<Self::Example>> {
        let train = load_shakespeare(&self.0.join("train.txt"))?;
        Ok(Train::new(train))
    }
}

impl TestSet for Shakespeare {
    type Example = String;
    fn test(&self) -> io::Result<Test<Self::Example>> {
        let test = load_shakespeare(&self.0.join("test.txt"))?;
        Ok(Test::new(test))
    }
}

impl ValidationSet for Shakespeare {
    type Example = String;
    fn validation(&self) -> io::Result<Validation<Self::Example>> {
        let validation = load_shakespeare(&self.0.join("validation.txt"))?;
        Ok(Validation::new(validation))
    }
}
