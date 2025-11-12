use rand::seq::{IndexedRandom, IteratorRandom};

use crate::dataset::{Test, Train, Validation};

pub struct Distill(pub f64);

impl Distill {
    pub fn train<T>(&self, train: Train<T>) -> Train<T> {
        let len = (train.0.len() as f64 * self.0) as usize;
        Train::new(train.0.into_iter().choose_multiple(&mut rand::rng(), len))
    }

    pub fn test<T>(&self, test: Test<T>) -> Test<T> {
        let len = (test.0.len() as f64 * self.0) as usize;
        Test::new(test.0.into_iter().choose_multiple(&mut rand::rng(), len))
    }
    
    pub fn validation<T>(&self, validation: Validation<T>) -> Validation<T> {
        let len = (validation.0.len() as f64 * self.0) as usize;
        Validation::new(validation.0.into_iter().choose_multiple(&mut rand::rng(), len))
    }
}