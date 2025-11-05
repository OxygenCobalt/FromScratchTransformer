use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::tensor::{Tensor, Tt};

pub trait TrainSet {
    type Example: LazyExample;
    fn train(&self) -> Train<Self::Example>;
}

pub struct Train<'a, E: LazyExample>(&'a [E]);

impl <'a, E: LazyExample> Train<'a, E> {
    pub fn new(data: &'a [E]) -> Self {
        Self(data)
    }

    pub fn batch<T: Tensor>(&self, size: usize) -> impl ExactSizeIterator<Item=Example<T>> {
        let mut examples = self.0.to_vec();
        examples.shuffle(&mut rand::rng());
        let batches: Vec<Vec<E>> = examples.chunks(size).map(|chunks| chunks.to_vec()).collect();
        batches.into_iter().map(|chunk| {
            let inputs = T::tensor(Tt(chunk.iter().map(|e| e.input::<T>()).collect())).unwrap();
            let outputs = T::tensor(Tt(chunk.iter().map(|e| e.output::<T>()).collect())).unwrap();
            Example { input: inputs, output: outputs }
        })
    }

    pub fn par_batch<T: Tensor + Send + Sync>(&self, size: usize) -> impl ExactSizeIterator<Item=impl ParallelIterator<Item=Example<T>>> {
        let mut examples = self.0.to_vec();
        examples.shuffle(&mut rand::rng());
        let batches: Vec<Vec<E>> = examples.chunks(size).map(|chunks| chunks.to_vec()).collect();
        batches.into_iter().map(|chunk| chunk.into_par_iter().map(|e| Example { input: e.input(), output: e.output() }))
    }
}

pub trait TestSet {
    type Example: LazyExample;
    fn test(&self) -> Test<Self::Example>;
}

pub struct Test<'a, E: LazyExample>(&'a [E]);

impl <'a, E: LazyExample> Test<'a, E> {
    pub fn new(data: &'a [E]) -> Self {
        Self(data)
    }

    pub fn iter<T: Tensor>(&self) -> impl ExactSizeIterator<Item=Example<T>> {
        self.0.iter().map(|e| Example { input: e.input(), output: e.output() })
    }

    pub fn par_iter<T: Tensor + Send + Sync>(&self) -> impl ParallelIterator<Item=Example<T>> {
        self.0.par_iter().map(|e| Example { input: e.input(), output: e.output() })
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub trait ValidationSet {
    type Example: LazyExample;
    fn validation(&self) -> Test<Self::Example>;
}

pub struct Validation<'a, E: LazyExample>(&'a [E]);

impl <'a, E: LazyExample> Validation<'a, E> {
    pub fn new(data: &'a [E]) -> Self {
        Self(data)
    }

    pub fn iter<T: Tensor>(&self) -> impl ExactSizeIterator<Item=Example<T>> {
        self.0.iter().map(|e| Example { input: e.input(), output: e.output() })
    }

    pub fn par_iter<T: Tensor + Send + Sync>(&self) -> impl ParallelIterator<Item=Example<T>> {
        self.0.par_iter().map(|e| Example { input: e.input(), output: e.output() })
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

pub trait LazyExample: Clone + Send + Sync {
    fn input<T: Tensor>(&self) -> T;
    fn output<T: Tensor>(&self) -> T;
}

pub struct Example<T: Tensor> {
    pub input: T,
    pub output: T
}
