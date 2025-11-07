use colored::Colorize;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::io;

pub trait TrainSet {
    type Example;
    fn train(&self) -> io::Result<Train<Self::Example>>;
}

pub struct Train<E>(Vec<E>);

impl <E> Train<E> {
    pub fn new(data: Vec<E>) -> Self {
        Self(data)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item=&E> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn map<F: Iterator<Item=G>, G>(self, remapper: impl Fn(E) -> F) -> Train<G> {
        let map_bar = ProgressBar::new_spinner()
            .with_style(ProgressStyle::with_template("{prefix}: {pos:>4}")
                            .unwrap()
                            .progress_chars("=> "))
            .with_prefix("map@train".yellow().to_string());
        Train(self.0.into_iter().flat_map(remapper).progress_with(map_bar).collect())
    }
}

impl <'a, E> Train<E> {
    pub fn batch(&self, size: usize) -> impl ExactSizeIterator<Item=impl Iterator<Item=&E>> {
        let mut examples: Vec<&E> = self.0.iter().collect();
        examples.shuffle(&mut rand::rng());
        let batches: Vec<Vec<&E>> = examples.chunks(size).map(|chunks| chunks.to_vec()).collect();
        batches.into_iter().map(|v| v.into_iter())
    }
}

impl <'a, E: Send + Sync> Train<E> {
    pub fn par_batch(&self, size: usize) -> impl ExactSizeIterator<Item=impl ParallelIterator<Item=&E>> {
        let mut examples: Vec<&E> = self.0.iter().collect();
        examples.shuffle(&mut rand::rng());
        let batches: Vec<Vec<&E>> = examples.chunks(size).map(|chunks| chunks.to_vec()).collect();
        batches.into_iter().map(|chunk| chunk.into_par_iter())
    }
}

pub trait TestSet {
    type Example;
    fn test(&self) -> io::Result<Test<Self::Example>>;
}

pub struct Test<E>(Vec<E>);

impl <E> Test<E> {
    pub fn new(data: Vec<E>) -> Self {
        Self(data)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item=&E> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn map<F: Iterator<Item=G>, G>(self, remapper: impl Fn(E) -> F) -> Test<G> {
        let map_bar = ProgressBar::new_spinner()
            .with_style(ProgressStyle::with_template("{prefix}: {pos:>4}")
                            .unwrap()
                            .progress_chars("=> "))
            .with_prefix("map@test".yellow().to_string());
        Test(self.0.into_iter().flat_map(remapper).progress_with(map_bar).collect())
    }
}

impl <E: Send + Sync> Test<E> {
    pub fn par_iter(&self) -> impl ParallelIterator<Item=&E> {
        self.0.par_iter()
    }
}
pub trait ValidationSet {
    type Example;
    fn validation(&self) -> io::Result<Validation<Self::Example>>;
}

pub struct Validation<E>(Vec<E>);

impl <E> Validation<E> {
    pub fn new(data: Vec<E>) -> Self {
        Self(data)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item=&E> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn map<F: Iterator<Item=G>, G>(self, remapper: impl Fn(E) -> F) -> Validation<G> {
        let map_bar = ProgressBar::new_spinner()
            .with_style(ProgressStyle::with_template("{prefix}: {pos:>4}")
                            .unwrap()
                            .progress_chars("=> "))
            .with_prefix("map@validation".yellow().to_string());
        Validation(self.0.into_iter().flat_map(remapper).progress_with(map_bar).collect())
    }
}

impl <E: Send + Sync> Validation<E> {
    pub fn par_iter(&self) -> impl ParallelIterator<Item=&E> {
        self.0.par_iter()
    }
}

pub trait Example<T> {
    fn input(&self) -> T;
    fn output(&self) -> T;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerExample<T : Clone> {
    pub input: T,
    pub output: T
}

impl <T : Clone> Example<T> for EagerExample<T> {
    fn input(&self) -> T {
        self.input.clone()
    }

    fn output(&self) -> T {
        self.output.clone()
    }
}
