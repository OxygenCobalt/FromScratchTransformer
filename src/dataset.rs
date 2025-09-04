use rand::seq::SliceRandom;

use crate::matrix::Shape;

use super::matrix::Matrix;


#[derive(Clone)]
pub struct Train {
    pub io_shape: IOShape,
    pub examples: Vec<Example>,
}

impl Train {
    pub fn batch(&self, size: usize) -> Vec<Example> {
        let mut examples = self.examples.clone();
        examples.shuffle(&mut rand::rng());
        self.examples.chunks(size).map(|examples| {                    
            let input = Matrix::columns(
                examples.iter().map(|e| &e.input).collect::<Vec<_>>().as_slice()
            );          
            let output = Matrix::columns(
                examples.iter().map(|e| &e.output).collect::<Vec<_>>().as_slice()
            );
            Example { input, output }
        }).collect()
    }
}

#[derive(Clone)]
pub struct Example {
    pub input: Matrix,
    pub output: Matrix,
}

pub struct Test {
    pub io_shape: IOShape,
    pub examples: Vec<Example>,
}

#[derive(Clone, Copy)]
pub struct IOShape {
    pub in_size: usize,
    pub out_size: usize
}

pub struct Dataset {
    pub train: Train,
    pub test: Test,
    pub io_shape: IOShape,
}
