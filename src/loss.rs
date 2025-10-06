use crate::tensor::Tensor;

pub trait Loss {
    fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T;
}

pub struct MSE;

impl Loss for MSE {
    fn loss<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
        // println!("akti {:?} {:?}", batch_activations.shape(), output.shape());
        batch_activations.sub(output).unwrap().pow(2).sum()
    }
}
