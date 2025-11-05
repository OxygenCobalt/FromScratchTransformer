use crate::tensor::Tensor;

#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
    ReLU,
    SiLU,
    Softmax,
    Tanh
}

impl Activation {
    pub fn id(&self) -> &'static [u8; 8] {
        match self {
            Self::Sigmoid => b"ActvSigm",
            Self::ReLU => b"ActvReLU",
            Self::SiLU => b"ActvSiLU",
            Self::Softmax => b"ActvSfmx",
            Self::Tanh => b"ActvTanh",
        }
    }

    pub fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"ActvSigm" => Some(Self::Sigmoid),
            b"ActvReLU" => Some(Self::ReLU),
            b"ActvSiLU" => Some(Self::SiLU),
            b"ActvSfmx" => Some(Self::Softmax),
            b"ActvTanh" => Some(Self::Tanh),
            _ => None,
        }
    }

    pub fn activate<T: Tensor>(&self, y: T) -> T {
        match self {
            Self::Sigmoid => T::scalar(1.0).add(&y.neg().exp()).unwrap().pow(-1),
            Self::ReLU => y.max(0.0),
            Self::SiLU => y.clone().mul(&Self::Sigmoid.activate(y)).unwrap(),
            Self::Softmax => {
                let max = {
                    let mut shape = vec![1];
                    shape.extend_from_slice(y.shape()[1..].into());
                    y.colmax().unwrap().reshape(&shape).unwrap()
                };
                let shifted = y.clone().sub(&max).unwrap();

                let exp = shifted.clone().exp();
                let norm = {
                    let mut shape = vec![1];
                    shape.extend_from_slice(exp.shape()[1..].into());
                    exp.clone().sum().pow(-1).reshape(&shape).unwrap()
                };
                exp.mul(&norm).unwrap()
            },
            Self::Tanh => y.tanh()
        }
    }
}
