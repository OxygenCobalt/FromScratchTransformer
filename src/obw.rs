use std::{fs::File, io::{self, BufRead, BufReader}, iter, path::Path};

use colored::Colorize;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};

use crate::{embeddings::EmbeddingsBuilder, tensor::Tensor};

pub struct OneBillionWords<T: Tensor> {
    pub train: Vec<T>,
    pub test: Vec<T>
}

pub fn one_billion_words<T: Tensor>(path: &Path, mut embeddings_builder: impl EmbeddingsBuilder) -> io::Result<OneBillionWords<T>> {
    println!("{}: building embeddings (this will take a bit)", "obw".green());
    let train_path = path.join("training-monolingual.tokenized.shuffled");
    let test_path = path.join("training-monolingual.tokenized.shuffled");
    let all_sentences = stream_words(&train_path, "training embeddings from train")?
        .chain(stream_words(&test_path, "training embeddings from test")?)
        .flatten()
        .flatten();
    for sentence in all_sentences {
        for word in sentence? {
            embeddings_builder.add(word);
        }
    }

    let embeddings = embeddings_builder.build();
    println!("{}: saving embeddings for later use", "obw".green());
    embeddings.write(&mut File::create("obw.emb").unwrap()).unwrap();

    let mut train = Vec::new();
    for sentence in stream_words(&train_path, "mapping train to embeddings")?.flatten().flatten() {
        for word in sentence? {
            train.push(embeddings.to_vec(word).unwrap())
        }
    }
    let mut test = Vec::new();
    for sentence in stream_words(&test_path, "mapping test to embeddings")?.flatten().flatten() {
        for word in sentence? {
            test.push(embeddings.to_vec(word).unwrap())
        }
    }

    Ok(OneBillionWords { train, test })
}

fn stream_words(path: &Path, msg: &str) -> io::Result<impl Iterator<Item=io::Result<impl Iterator<Item=io::Result<Vec<String>>>>>> {
    let count = path.read_dir()?.count();
    Ok(path.read_dir()?.map(|i| i.and_then(|f| {
        let mut file = BufReader::new(File::open(f.path())?);
        Ok(iter::from_fn(move || {
            let mut line = String::new();
            match file.read_line(&mut line) {
                Ok(i) if i > 0 => Some(Ok(line.split(' ').map(|s| s.to_string()).collect())),
                Ok(_) => None,
                Err(e) => Some(Err(e))
            }
        }))
    })).progress_with(
        ProgressBar::new(count as u64)
        .with_style(ProgressStyle::with_template("{prefix}: {msg}... {bar:40} {pos:>4}/{len:4} [{eta_precise}]").unwrap().progress_chars("=> "))
        .with_message(msg.to_string())
        .with_prefix("obw".green().to_string())
    ))
}
