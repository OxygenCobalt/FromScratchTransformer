use std::{fs::File, path::Path};

use arrow::array::{Array, StringArray};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ArrowReaderBuilder;

pub struct WikiText103 {
    pub train: Vec<String>,
    pub test: Vec<String>,
    pub validation: Vec<String>,
}

impl WikiText103 {
    pub fn load(path: &Path) -> Result<Self, parquet::errors::ParquetError> {
        let mut train =
            load_wikitext(&path.join("wikitext-103-raw-v1/train-00000-of-00002.parquet"))?;
        train.extend(load_wikitext(
            &path.join("wikitext-103-raw-v1/train-00001-of-00002.parquet"),
        )?);
        let test = load_wikitext(&path.join("wikitext-103-raw-v1/test-00000-of-00001.parquet"))?;
        let validation =
            load_wikitext(&path.join("wikitext-103-raw-v1/validation-00000-of-00001.parquet"))?;
        println!(
            "{}: {} / {}/ {}",
            "wikitext".green(),
            format!["{} train examples", train.len()],
            format!["{} test examples", test.len()],
            format!["{} validation examples", test.len()]
        );
        Ok(Self {
            train,
            test,
            validation,
        })
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
        let text = text_column.value(0);
        texts.push(text.to_string());
        load_progress.inc(1);
    }
    load_progress.finish();
    Ok(texts)
}
