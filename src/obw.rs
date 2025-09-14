use std::{fs::File, io::{self, BufRead, BufReader}, path::Path};

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{embeddings::LookupEmbeddings, tensor::Tensor};

pub struct OneBillionWords<T: Tensor> {
    pub train: Vec<Vec<T>>,
    pub test: Vec<Vec<T>>
}

pub fn one_billion_words<T: Tensor>(path: &Path) -> io::Result<OneBillionWords<T>> {
    let train = read_worddir(&path.join("training-monolingual.tokenized.shuffled"), "obw train".green().to_string())?;
    let test = read_worddir(&path.join("heldout-monolingual.tokenized.shuffled"), "obw test".green().to_string())?;
    let all = train.iter().flatten().map(|s| s.as_str()).chain(test.iter().flatten().map(|s| s.as_str()));
    let embeddings = LookupEmbeddings::train(all, train.len() + test.len());
    let train_embeddings = train.into_iter().map(|t| t.into_iter().map(|w| embeddings.test(w).unwrap()).collect()).collect();
    let test_embeddings = test.into_iter().map(|t| t.into_iter().map(|w| embeddings.test(w).unwrap()).collect()).collect();
    Ok(OneBillionWords {
        train: train_embeddings,
        test: test_embeddings
    })
}

fn read_worddir(path: &Path, prefix: String) -> io::Result<Vec<Vec<String>>> {
    let mut sentences = Vec::new();
    let count = path.read_dir().unwrap().count();
    let sgd_bar = ProgressBar::new(count as u64)
        .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] {msg}")
                        .unwrap()
                        .progress_chars("=> "))
        .with_prefix(prefix);
    for child in path.read_dir()? {
        let mut file = BufReader::new(File::open(child?.path())?);
        let mut line = String::new();
        while file.read_line(&mut line)? > 0 {
            sentences.push(line.split(' ').map(String::from).collect());
            line.clear();
            sgd_bar.set_message(format!["/ {} sentences", sentences.len()]);
        }
        sgd_bar.inc(1);
    }
    Ok(sentences)
}