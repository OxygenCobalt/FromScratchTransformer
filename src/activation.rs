use crate::{autograd::{AutogradNode, Operation, OperationFactory}, matrix::Matrix, tensor::{self, Tensor}};


#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
    ReLU,
    SiLU
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
            Self::ReLU => b"ReLU\0\0\0\0",
            Self::SiLU => b"SiLU\0\0\0\0"
        }
    }

    pub fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"Sigmoid\0" => Some(Self::Sigmoid),
            b"ReLU\0\0\0\0" => Some(Self::ReLU),
            b"SiLU\0\0\0\0" => Some(Self::SiLU),
            _ => None
        }
    }

    pub fn activate(&self, matrix: Matrix) -> Matrix {
        match self {
            Self::Sigmoid => matrix.apply(SigmoidActivation::sigmoid),
            Self::ReLU => matrix.apply(ReLUActivation::relu),
            Self::SiLU => matrix.apply(SiLUActivation::silu)
        }
    }

    pub fn activate2<T: Tensor>(&self, y: T) -> T {
        match self {
            Self::Sigmoid => T::scalar(1.0).sub(&y.neg().exp()).unwrap().inv(),
            Self::ReLU => y.max(0.0),
            Self::SiLU => y.clone().mul(&Self::Sigmoid.activate2(y)).unwrap()
        }
    }
}

struct SigmoidActivation(AutogradNode);

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

struct ReLUActivation(AutogradNode);

impl ReLUActivation {
    fn relu(x: f64) -> f64 {
        x.min(0.0)
    }

    fn relu_prime(x: f64) -> f64 {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }
}

impl Operation for ReLUActivation {
    fn f(&self) -> Matrix {
        self.0.borrow().matrix.clone().apply(Self::relu)
    }

    fn df(&mut self, grad: &Matrix) {
        let result = self.0.borrow().matrix.clone().apply(Self::relu_prime).mul(grad);
        self.0.borrow_mut().backward(result);
    }
}

struct SiLUActivation(AutogradNode);

impl SiLUActivation {
    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    fn silu_prime(x: f64) -> f64 {
        let exp = (-x).exp();
        let d = 1.0 + exp;
        (d + x * exp) / d.powi(2)
    }
}

impl Operation for SiLUActivation {
    fn f(&self) -> Matrix {
        self.0.borrow().matrix.clone().apply(Self::silu)
    }

    fn df(&mut self, grad: &Matrix) {
        let result = self.0.borrow().matrix.clone().apply(Self::silu_prime).mul(grad);
        self.0.borrow_mut().backward(result);
    }
}
