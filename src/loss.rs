use crate::tensor::Tensor;

pub trait Loss {
    fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T;
}

pub struct MSE;

impl Loss for MSE {
    fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
        batch_activations.sub(output).unwrap().pow(2).sum()
    }
}

pub struct LogLikelihood;

impl Loss for LogLikelihood {
    fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
        let max = batch_activations.colmax().unwrap();
        let mut shape = vec![1];
        shape.extend_from_slice(batch_activations.shape()[1..].into());
        let max_broadcast = max.clone().reshape(&shape).unwrap();
        let shifted = batch_activations.clone().sub(&max_broadcast).unwrap();
        let log_sum_exp = shifted.clone().exp().sum().ln().add(&max).unwrap();
        log_sum_exp.sub(&batch_activations.at_argmax(&output).unwrap()).unwrap()
    }
}
