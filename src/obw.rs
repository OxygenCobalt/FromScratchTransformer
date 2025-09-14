use std::{fs::File, io::{self, BufRead, BufReader}, path::Path};

use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};

use crate::{nn::Example, tensor::Tensor};

pub struct OneBillionWords {
    pub train: Vec<Vec<String>>,
    pub test: Vec<Vec<String>>
}

pub fn one_billion_words(path: &Path) -> io::Result<OneBillionWords> {
    let train = read_worddir(&path.join("training-monolingual.tokenized.shuffled"), "obw train".green().to_string())?;
    let test = read_worddir(&path.join("heldout-monolingual.tokenized.shuffled"), "obw test".green().to_string())?;
    Ok(OneBillionWords {
        train,
        test
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