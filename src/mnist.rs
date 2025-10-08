use std::{fs::File, path::Path};

use arrow::array::{Array, BinaryArray, Int64Array, StructArray};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ArrowReaderBuilder;

use crate::{
    loss::Loss,
    nn::{Example, NeuralNetwork, Reporting, Test, Train, TrainingSet},
    tensor::Tensor,
};

pub struct MNIST<T: Tensor> {
    train: Train<T>,
    test: Test<T>,
}

impl<T: Tensor> MNIST<T> {
    pub fn load(at: &Path) -> Result<Self, parquet::errors::ParquetError> {
        let train = load_mnist(&at.join("mnist/train-00000-of-00001.parquet"))?;
        let test = load_mnist(&at.join("./mnist/test-00000-of-00001.parquet"))?;
        println!(
            "{}: {} / {}",
            "mnist".green(),
            format!["{} train examples", train.len()],
            format!["{} test examples", test.len()]
        );
        let dataset = MNIST {
            train: Train { examples: train },
            test: Test { examples: test },
        };
        Ok(dataset)
    }
}

impl<T: Tensor> TrainingSet<T> for MNIST<T> {
    fn get(&self) -> std::io::Result<Train<T>> {
        Ok(self.train.clone())
    }
}

impl<'a, T: Tensor> Reporting<T> for MNIST<T> {
    fn report(
        &self,
        nn: &NeuralNetwork<T>,
        loss: &impl Loss,
        _: Option<u64>,
    ) -> std::io::Result<()> {
        fn argmax<T: Tensor>(tensor: &T) -> usize {
            tensor
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0
        }
        let results = nn.test(&self.test, loss);
        let successes = results
            .results
            .iter()
            .filter(|result| argmax(&result.example.output) == argmax(&result.activations))
            .count();
        let percent = (successes as f64 / results.results.len() as f64) * 100.0;
        println!(
            "{}: avg loss = {:.3}, success = {} / {} ({:.2}%)",
            "mnist".green(),
            results.avg_loss,
            successes,
            results.results.len(),
            percent
        );
        Ok(())
    }
}

fn load_mnist<T: Tensor>(path: &Path) -> Result<Vec<Example<T>>, parquet::errors::ParquetError> {
    let load_progress = ProgressBar::new_spinner()
        .with_style(ProgressStyle::with_template("{prefix}: loading {msg} {pos:>4}").unwrap())
        .with_prefix("mnist".green().to_string())
        .with_message(path.file_name().unwrap().to_string_lossy().to_string());
    let train = File::open(path)?;
    let parquet = ArrowReaderBuilder::try_new(train)?.build()?;
    let mut images: Vec<T> = Vec::new();
    let mut labels: Vec<T> = Vec::new();
    for item in parquet.take(1) {
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
            let mut buf = vec![0; reader.output_buffer_size()];
            reader.next_frame(&mut buf).unwrap();
            let pixels: Vec<f64> = buf.iter().map(|&b| b as f64 / 255.0).collect();

            let image_matrix = T::vector(pixels)
                .unwrap()
                .reshape(&[w as usize, h as usize])
                .unwrap();
            images.push(image_matrix);

            let label = label_array.value(i);
            let mut label_vec = vec![0.0; 10];
            label_vec[label as usize] = 1.0;
            let label_matrix = T::vector(label_vec).unwrap();
            labels.push(label_matrix);
            load_progress.inc(1);
        }
    }
    load_progress.finish();
    let examples = images
        .into_iter()
        .zip(labels.into_iter())
        .map(|(input, output)| Example { input, output });
    Ok(examples.collect())
}
