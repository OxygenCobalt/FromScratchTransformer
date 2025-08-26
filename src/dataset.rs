use std::slice::Chunks;

use rand::seq::SliceRandom;

use crate::{matrix::Matrix, nn::IOShape};
use std::fs::File;

use arrow::array::{Array, BinaryArray, Int64Array, StructArray};
use parquet::arrow::arrow_reader::ArrowReaderBuilder;

#[derive(Clone)]
pub struct Train {
    examples: Vec<Example>,
}

impl Train {
    pub fn batch(&mut self, size: usize) -> Chunks<Example> {
        self.examples.shuffle(&mut rand::rng());
        return self.examples.chunks(size);
    }
}

#[derive(Clone)]
pub struct Example {
    pub input: Matrix,
    pub output: Matrix,
}

pub struct Test {
    pub examples: Vec<Example>,
}

pub struct Dataset {
    pub train: Train,
    pub test: Test,
    pub io_shape: IOShape,
}

pub fn mnist() -> Result<Dataset, parquet::errors::ParquetError> {
    let train = load_mnist("./mnist/mnist/train-00000-of-00001.parquet")?;
    let test = load_mnist("./mnist/mnist/test-00000-of-00001.parquet")?;
    let dataset = Dataset {
        train: Train { examples: train },
        test: Test { examples: test },
        io_shape: IOShape {
            input_size: 784,
            output_size: 10,
        },
    };
    Ok(dataset)
}

fn load_mnist(path: &str) -> Result<Vec<Example>, parquet::errors::ParquetError> {
    let train = File::open(path)?;
    let parquet = ArrowReaderBuilder::try_new(train)?.build()?;
    let mut images: Vec<Matrix> = Vec::new();
    let mut labels: Vec<Matrix> = Vec::new();
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
            let mut buf = vec![0; reader.output_buffer_size()];
            reader.next_frame(&mut buf).unwrap();
            let pixels: Vec<f64> = buf.iter().map(|&b| b as f64 / 255.0).collect();

            let image_matrix = Matrix::vector(pixels);
            images.push(image_matrix);

            let label = label_array.value(i);
            let mut label_vec = vec![0.0; 10];
            label_vec[label as usize] = 1.0;
            let label_matrix = Matrix::vector(label_vec);
            labels.push(label_matrix);
        }
    }
    let examples = images
        .into_iter()
        .zip(labels.into_iter())
        .map(|(input, output)| Example { input, output });
    Ok(examples.collect())
}
