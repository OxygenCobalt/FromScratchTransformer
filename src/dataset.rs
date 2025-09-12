use rand::seq::SliceRandom;

use crate::tensor::{CPUTensor, Th};

use super::tensor::Tensor;

#[derive(Clone)]
pub struct Train {
    pub io_shape: IOShape,
    pub examples: Vec<Example>,
}

impl Train {
    pub fn batch(&self, size: usize) -> Vec<Example> {
        let mut examples = self.examples.clone();
        examples.shuffle(&mut rand::rng());
        self.examples
            .chunks(size)
            .map(|examples| {
                let input = CPUTensor::tensor(Th::C(
                    examples
                        .iter()
                        .map(|e| Th::R(e.input.iter().cloned().collect()))
                        .collect::<Vec<Th>>(),
                ))
                .unwrap();
                let output = CPUTensor::tensor(Th::C(
                    examples
                        .iter()
                        .map(|e| Th::R(e.output.iter().cloned().collect()))
                        .collect::<Vec<Th>>(),
                ))
                .unwrap();
                Example { input, output }
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct Example {
    pub input: CPUTensor,
    pub output: CPUTensor,
}

pub struct Test {
    pub io_shape: IOShape,
    pub examples: Vec<Example>,
}

#[derive(Clone, Copy)]
pub struct IOShape {
    pub in_size: usize,
    pub out_size: usize,
}

pub struct Dataset {
    pub train: Train,
    pub test: Test,
    pub io_shape: IOShape,
}
