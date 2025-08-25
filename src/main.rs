use std::fs::File;

use parquet::arrow::arrow_reader::ArrowReaderBuilder;
use arrow::array::{Array, BinaryArray, Int64Array, StructArray};

use crate::{matrix::{Matrix, Shape}, nn::{IOShape, Identity, NeuralNetwork, MSE}};

mod matrix;
mod nn;

fn main() {
    let train = File::open("./mnist/mnist/train-00000-of-00001.parquet").unwrap();
    let parquet = ArrowReaderBuilder::try_new(train).unwrap().build().unwrap();
    let mut images: Vec<Matrix> = Vec::new();
    let mut labels: Vec<Matrix> = Vec::new();
    for item in parquet {
        let record_batch = item.unwrap();
        
        // Get the "image" column - it's a struct with "bytes" and "path" fields
        let image_column = record_batch.column_by_name("image")
            .expect("Column 'image' not found");
        
        // Cast to StructArray
        let image_struct = image_column.as_any()
            .downcast_ref::<StructArray>()
            .expect("Failed to cast image column to StructArray");
        
        // Access the "bytes" field from the struct
        let bytes_column = image_struct.column_by_name("bytes")
            .expect("No 'bytes' field in image struct");
        
        // Cast the bytes field to BinaryArray
        let image_bytes_array = bytes_column.as_any()
            .downcast_ref::<BinaryArray>()
            .expect("Failed to cast bytes to BinaryArray");
        
        // Get the labels
        let label_column = record_batch.column_by_name("label")
            .expect("Column 'label' not found");
        
        let label_array = label_column.as_any()
            .downcast_ref::<Int64Array>()
            .expect("Failed to cast label column to Int64Array");
        
        // Iterate through the batch
        for i in 0..image_struct.len() {
            // Get image bytes
            if !image_bytes_array.is_null(i) {
                let image_bytes = image_bytes_array.value(i);
                
                // Decode PNG image
                let decoder = png::Decoder::new(image_bytes);
                if let Ok(mut reader) = decoder.read_info() {
                    let mut buf = vec![0; reader.output_buffer_size()];
                    
                    // Read the entire image
                    if let Ok(_) = reader.next_frame(&mut buf) {
                        // MNIST images are 28x28 grayscale
                        if buf.len() == 784 {
                            // Convert to f64 and normalize to 0-1
                            let pixels: Vec<f64> = buf.iter()
                                .map(|&b| b as f64 / 255.0)
                                .collect();
                            
                            let image_matrix = Matrix::from_vec(784, 1, pixels);
                            images.push(image_matrix);
                        }
                    }
                }
            }
            
            // Get label
            if !label_array.is_null(i) {
                let label = label_array.value(i);
                
                // Create one-hot encoded label vector
                let mut label_vec = vec![0.0; 10];
                label_vec[label as usize] = 1.0;
                let label_matrix = Matrix::from_vec(10, 1, label_vec);
                labels.push(label_matrix);
            }
        }
    }
    println!("\nLoaded {} images and {} labels", images.len(), labels.len());
    
    let shape = IOShape {
        input_size: 784,
        output_size: 10
    };
    let mut nn = NeuralNetwork::<Identity, MSE>::new(shape, &[15]);
    nn.train(images.into_iter().zip(labels.into_iter()).collect(), 0.01);
}
