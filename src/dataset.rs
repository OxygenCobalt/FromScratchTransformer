use std::slice::Chunks;

use rand::seq::SliceRandom;

use super::matrix::Matrix;


#[derive(Clone)]
pub struct Train {
    pub examples: Vec<Example>,
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

pub struct IOShape {
    pub in_size: usize,
    pub out_size: usize
}

pub struct Dataset {
    pub train: Train,
    pub test: Test,
    pub io_shape: IOShape,
}
