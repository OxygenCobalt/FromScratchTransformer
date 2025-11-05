use std::{fs::File, io, path::{Path, PathBuf}};

use arrow::array::{Array, BinaryArray, Int64Array, StructArray};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ArrowReaderBuilder;

use crate::{
    dataset::{Example, Test, TestSet, Train, TrainSet}, tensor::Tensor
};

pub struct Mnist(pub PathBuf);

fn load_mnist(path: &Path) -> Result<Vec<MnistDigit>, parquet::errors::ParquetError> {
    let load_progress = ProgressBar::new_spinner()
        .with_style(ProgressStyle::with_template("{prefix}: loading {msg} {pos:>4}").unwrap())
        .with_prefix("mnist".green().to_string())
        .with_message(path.file_name().unwrap().to_string_lossy().to_string());
    let train = File::open(path)?;
    let parquet = ArrowReaderBuilder::try_new(train)?.build()?;
    let mut examples: Vec<MnistDigit> = Vec::new();
    for item in parquet {
        let record_batch = item.unwrap();
        let image_column = record_batch
            .column_by_name("image")
            .expect("Column 'image' not found");
        let image_struct = image_column
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("Failed to cast image column to StructArray");
        let bytes_column = image_struct
            .column_by_name("bytes")
            .expect("No 'bytes' field in image struct");
        let image_bytes_array = bytes_column
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("Failed to cast bytes to BinaryArray");
        let label_column = record_batch
            .column_by_name("label")
            .expect("Column 'label' not found");
        let label_array = label_column
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Failed to cast label column to Int64Array");
        for i in 0..image_struct.len() {
            let image_bytes = image_bytes_array.value(i);
            let decoder = png::Decoder::new(image_bytes);
            let mut reader = decoder.read_info().unwrap();
            let (w, h) = reader.info().size();
            let mut buf: Vec<u8> = vec![0; reader.output_buffer_size()];
            reader.next_frame(&mut buf).unwrap();
            let label = label_array.value(i);
            examples.push(MnistDigit {
                pixels: buf,
                width: w,
                height: h,
                label: label,
            });
            load_progress.inc(1);
        }
    }
    load_progress.finish();
    Ok(examples)
}

impl TrainSet for Mnist {
    type Example = MnistDigit;
    fn train(&self) -> io::Result<Train<Self::Example>> {
        let train = load_mnist(&self.0.join("mnist/train-00000-of-00001.parquet"))?;
        Ok(Train::new(train))
    }
}

impl TestSet for Mnist {
    type Example = MnistDigit;
    fn test(&self) -> io::Result<Test<Self::Example>> {
        let test = load_mnist(&self.0.join("./mnist/test-00000-of-00001.parquet"))?;
        Ok(Test::new(test))
    }
}

#[derive(Clone)]
pub struct MnistDigit {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    label: i64,
}

impl <T: Tensor> Example<T> for MnistDigit {
    fn input(&self) -> T {
        T::vector(self.pixels.iter().map(|&b| b as f64 / 255.0).collect::<Vec<f64>>())
            .unwrap()
            .reshape(&[self.width as usize, self.height as usize])
            .unwrap()
    }

    fn output(&self) -> T {
        let mut output = vec![0.0; 10];
        output[self.label as usize] = 1.0;
        T::vector(output).unwrap()
    }
}