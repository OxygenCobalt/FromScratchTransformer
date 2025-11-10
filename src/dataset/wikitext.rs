use std::{fs::File, path::{Path, PathBuf}};

use arrow::array::{Array, StringArray};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ArrowReaderBuilder;

use crate::dataset::{TestSet, Train, TrainSet, ValidationSet};

pub struct WikiText103(pub PathBuf);

impl TrainSet for WikiText103 {
    type Example = String;

    fn train(&self) -> std::io::Result<crate::dataset::Train<Self::Example>> {
        let train = load_wikitext(&self.0.join("wikitext-103-raw-v1/train-00000-of-00002.parquet"))?
            .into_iter()
            .chain(load_wikitext(&self.0.join("wikitext-103-raw-v1/train-00001-of-00002.parquet"))?)
            .collect();
        Ok(Train::new(train))
    }
}

impl TestSet for WikiText103 {
    type Example = String;

    fn test(&self) -> std::io::Result<crate::dataset::Test<Self::Example>> {
        let test = load_wikitext(&self.0.join("wikitext-103-raw-v1/test-00000-of-00001.parquet"))?;
        Ok(crate::dataset::Test::new(test))
    }
}

impl ValidationSet for WikiText103 {
    type Example = String;

    fn validation(&self) -> std::io::Result<crate::dataset::Validation<Self::Example>> {
        let validation = load_wikitext(&self.0.join("wikitext-103-raw-v1/validation-00000-of-00001.parquet"))?;
        Ok(crate::dataset::Validation::new(validation))
    }
}

pub struct WikiText2(pub PathBuf);

impl TrainSet for WikiText2 {
    type Example = String;

    fn train(&self) -> std::io::Result<crate::dataset::Train<Self::Example>> {
        let train = load_wikitext(&self.0.join("wikitext-2-raw-v1/train-00000-of-00001.parquet"))?;
        Ok(Train::new(train))
    }
}

impl TestSet for WikiText2 {
    type Example = String;

    fn test(&self) -> std::io::Result<crate::dataset::Test<Self::Example>> {
        let test = load_wikitext(&self.0.join("wikitext-2-raw-v1/test-00000-of-00001.parquet"))?;
        Ok(crate::dataset::Test::new(test))
    }
}

impl ValidationSet for WikiText2 {
    type Example = String;

    fn validation(&self) -> std::io::Result<crate::dataset::Validation<Self::Example>> {
        let validation = load_wikitext(&self.0.join("wikitext-2-raw-v1/validation-00000-of-00001.parquet"))?;
        Ok(crate::dataset::Validation::new(validation))
    }
}

fn load_wikitext(path: &Path) -> Result<Vec<String>, parquet::errors::ParquetError> {
    let load_progress = ProgressBar::new_spinner()
        .with_style(ProgressStyle::with_template("{prefix}: loading {msg} {pos:>4}").unwrap())
        .with_prefix("wikitext".green().to_string())
        .with_message(path.file_name().unwrap().to_string_lossy().to_string());
    let train = File::open(path)?;
    let parquet = ArrowReaderBuilder::try_new(train)?.build()?;
    let mut texts = Vec::new();
    for item in parquet {
        let record_batch = item.unwrap();
        let text_column = record_batch
            .column_by_name("text")
            .expect("Column 'text' not found")
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..text_column.len() {
            let text = text_column.value(i);
            texts.push(text.to_string());
            load_progress.inc(1);
        }
    }
    load_progress.finish();
    Ok(texts)
}
