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
        // dbg!(batch_activations.iter().cloned().collect::<Vec<f64>>());
        // dbg!(output.iter().cloned().collect::<Vec<f64>>());
        batch_activations.at_argmax(&output).unwrap().ln().neg()
    }
}
