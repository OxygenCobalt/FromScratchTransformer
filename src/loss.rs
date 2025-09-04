use crate::{autograd::{AutogradNode, Operation, OperationFactory}, matrix::Matrix};

pub trait Loss {
    fn loss(&self, batch_activations: &Matrix, output: &Matrix) -> f64;
    fn op(&self, output: Matrix) -> impl OperationFactory;
}

pub struct MSE;

impl Loss for MSE {
    fn loss(&self, batch_activations: &Matrix, output: &Matrix) -> f64 {
        batch_activations.flatten().iter().zip(output.flatten().iter()).map(|(a, o)| (a - o).powi(2)).sum()
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