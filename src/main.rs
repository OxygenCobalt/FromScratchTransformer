use std::fs::File;

use arrow::array::{Array, BinaryArray, Int64Array, StructArray};
use parquet::arrow::arrow_reader::ArrowReaderBuilder;

use crate::{activation::Activation, dataset::{Dataset, Example, IOShape, Test, Train}, loss::MSE, matrix::Matrix, nn::{Layer, NeuralNetwork, Reporting, SuccessCriteria, TestResult}};

mod matrix;
mod autograd;
mod nn;
mod dataset;
mod loss;
mod activation;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() {
    println!("loading data...");
    let mut mnist = mnist().unwrap();
    println!("loading done: {} train examples, {} test examples.", mnist.train.examples.len(), mnist.test.examples.len());
    println!("begin training...");
    let mut nn = NeuralNetwork::new(&[
        Layer { neurons: mnist.io_shape.in_size, activation_fn: Activation::Sigmoid },
        Layer { neurons: 30, activation_fn: Activation::Sigmoid },
        Layer { neurons: mnist.io_shape.out_size, activation_fn: Activation::Sigmoid },
    ]);
    nn.train(
        &mut mnist.train,  
        30, 10, 3.0, 
        &MSE, 
        Some(MnistReporting { test: &mnist.test }), 
        None
    ).unwrap();
    println!("saving result to file...");
    let mut output = File::create("mnist.nn").unwrap();
    nn.write(&mut output).unwrap();
}

struct MnistReporting<'a> {
    test: &'a Test
}

impl <'a> Reporting for MnistReporting<'a> {
    fn data(&self) -> &Test {
        &self.test
    }

    fn report(&self, epoch: Option<u64>, result: TestResult) {
        let percent_successful = (result.successes as f64 / self.test.examples.len() as f64) * 100.0;
        let phase = epoch.map(|e| format!["epoch {}", e + 1]).unwrap_or("init".to_owned()) ;
        println!("{}: loss = {}, {} / {} ({}%) successful", phase, result.avg_loss, result.successes, self.test.examples.len(), percent_successful)
    }
}

impl <'a> SuccessCriteria for MnistReporting<'a> {
    fn is_success(&self, example: &Example, output: &Matrix) -> bool {
        example.output.argmax() == output.argmax()
    }
}

pub fn mnist() -> Result<Dataset, parquet::errors::ParquetError> {
    let train = load_mnist("./mnist/mnist/train-00000-of-00001.parquet")?;
    let test = load_mnist("./mnist/mnist/test-00000-of-00001.parquet")?;
    let io_shape = IOShape {
            in_size: 784,
            out_size: 10,
        };
    let dataset = Dataset {
        train: Train { examples: train, io_shape },
        test: Test { examples: test, io_shape },
        io_shape: IOShape {
            in_size: 784,
            out_size: 10,
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
