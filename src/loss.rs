use crate::{autograd::{AutogradNode, Operation, OperationFactory}, matrix::Matrix, tensor::Tensor};

pub trait Loss {
    fn loss(&self, batch_activations: &Matrix, output: &Matrix) -> f64;
    fn loss2<T: Tensor>(&self, batch_activations: &T, output: &T) -> T;
    fn op(&self, output: Matrix) -> impl OperationFactory;
}

pub struct MSE;

impl Loss for MSE {
    fn loss(&self, batch_activations: &Matrix, output: &Matrix) -> f64 {
        batch_activations.flatten().iter().zip(output.flatten().iter()).map(|(a, o)| (a - o).powi(2)).sum()
    }
    
    fn loss2<T: Tensor>(&self, batch_activations: &T, output: &T) -> T {
        batch_activations.clone().sub(output).unwrap().norm(2)
    }

    fn op(&self, output: Matrix) -> impl OperationFactory {
        MSEFactory { output }
    }
}

struct MSEFactory { output: Matrix }

impl OperationFactory for MSEFactory {
    fn new(self, this: AutogradNode) -> Box<dyn Operation> {
        Box::new(MSEOperation { batch_activations: this, output: self.output })
    }
}

struct MSEOperation {
    batch_activations: AutogradNode,
    output: Matrix
}

impl Operation for MSEOperation {
    fn f(&self) -> Matrix {
        Matrix::scalar(MSE.loss(&self.batch_activations.borrow().matrix, &self.output))
    }

    fn df(&mut self, _: &Matrix) {
        let result = self.batch_activations.borrow().matrix.clone().sub(&self.output);
        self.batch_activations.borrow_mut().backward(result);
    }
}