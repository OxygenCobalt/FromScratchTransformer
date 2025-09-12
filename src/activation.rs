use crate::tensor::Tensor;

#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
    ReLU,
    SiLU
}

impl Activation {
    pub fn id(&self) -> &'static [u8; 8] {
        match self {
            Self::Sigmoid => b"ActvSigm",
            Self::ReLU => b"ActvReLU",
            Self::SiLU => b"ActvSiLU"
        }
    }

    pub fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"ActvSigm" => Some(Self::Sigmoid),
            b"ActvReLU" => Some(Self::ReLU),
            b"ActvSiLU" => Some(Self::SiLU),
            _ => None
        }
    }

    pub fn activate<T: Tensor>(&self, y: T) -> T {
        match self {
            Self::Sigmoid => T::scalar(1.0).add(&y.neg().exp()).unwrap().pow(-1),
            Self::ReLU => y.max(0.0),
            Self::SiLU => y.clone().mul(&Self::Sigmoid.activate(y)).unwrap()
        }
    }
}
