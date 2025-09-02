use crate::{autograd::{AutogradNode, Operation, OperationFactory}, matrix::Matrix};


#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
}

impl OperationFactory for Activation {
    fn new(self, this: AutogradNode) -> Box<dyn Operation> {
        Box::new(SigmoidActivation(this))
    }
}

impl Activation {
    pub fn id(&self) -> &'static [u8; 8] {
        match self {
            Self::Sigmoid => b"Sigmoid\0",
        }
    }

    pub fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"Sigmoid\0" => Some(Self::Sigmoid),
            _ => None
        }
    }

    pub fn f(&self, matrix: Matrix) -> Matrix {
        matrix.apply(SigmoidActivation::sigmoid)
    }
}

pub struct SigmoidActivation(AutogradNode);

impl SigmoidActivation {
    fn sigmoid(n: f64) -> f64 {
        1f64 / (1f64 + (-n).exp())
    }

    fn sigmoid_prime(n: f64) -> f64 {
        Self::sigmoid(n) * (1.0 - Self::sigmoid(n))
    }
}

impl Operation for SigmoidActivation {
    fn f(&self) -> Matrix {
        self.0.borrow().matrix.clone().apply(Self::sigmoid)
    }

    fn df(&mut self, grad: &Matrix) {
        let result = self.0.borrow().matrix.clone().apply(Self::sigmoid_prime).mul(grad);
        self.0.borrow_mut().backward(result);
    }
}