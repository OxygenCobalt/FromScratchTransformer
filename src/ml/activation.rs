use crate::tensor::{Tensor, cpu2};

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
            Self::Softmax => y.softmax().unwrap(),
            Self::Tanh => y.tanh()
        }
    }

    pub fn forward(&self, x: f64) -> f64 {
        match self {
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::ReLU => if x > 0.0 { x } else { 0.0 },
            Self::SiLU => x / (1.0 + (-x).exp()),
            Self::Softmax => x, // Softmax is typically applied over a vector, not a single value
            Self::Tanh => x.tanh(),
        }
    }

    pub fn forward_all<'i>(&self, mut x: cpu2::CPUTensor<'i>) -> cpu2::CPUTensor<'i> {
        match self {
            Self::Sigmoid => x.data.to_mut().iter_mut().for_each(|v| *v = 1.0 / (1.0 + (-*v).exp())),
            Self::ReLU => x.data.to_mut().iter_mut().for_each(|v| *v = if *v > 0.0 { *v } else { 0.0 }),
            Self::SiLU => x.data.to_mut().iter_mut().for_each(|v| *v = *v / (1.0 + (-*v).exp())),
            Self::Softmax => {
                // what we want to do is be able to take the softmax
                // while preserving many-dimensional structures, so
                // we flatten down to [classes, cols] where cols is
                // just the flattened remaining dimensions that we
                // all factor into the norm.
                let classes = x.shape[0];
                let cols = x.data.len() / classes;
                let mut flat_x = x.reshape(&[classes, cols]).unwrap();
                let mut idx = 0;
                let flat_x_data = flat_x.data.to_mut().as_mut_slice();
                for _ in 0..cols { 
                    let mut max = f64::NEG_INFINITY;
                    let mut max_idx = idx;
                    for j in 0..classes {
                        let x = flat_x_data[max_idx];
                        if x > max {
                            max = x;
                        }
                        max_idx += flat_x.stride[0];
                    }
                    let mut norm = 0.0;
                    let mut pass1_idx = idx;
                    for _ in 0..classes {
                        flat_x_data[pass1_idx] = (flat_x_data[pass1_idx] - max).exp();
                        norm += flat_x_data[pass1_idx];
                        pass1_idx += flat_x.stride[0];
                    }
                    let mut pass2_idx = idx;
                    for _ in 0..classes {
                        flat_x_data[pass2_idx] /= norm;
                        pass2_idx += flat_x.stride[0];
                    }
                    idx += flat_x.stride[1];
                }
            }
            Self::Tanh => x.data.to_mut().iter_mut().for_each(|v| *v = v.tanh()),
        }
        x
    }

    pub fn backward_all<'i>(&self, x: &cpu2::CPUTensor<'i>, mut grad: cpu2::CPUTensor<'i>) -> cpu2::CPUTensor<'i> {
        match self {
            Self::Sigmoid => x.data.iter().zip(grad.data.to_mut().iter_mut()).for_each(|(v, g)| {
                let sig = 1.0 / (1.0 + (-*v).exp());
                *g = sig * (1.0 - sig) * (*g);
            }),
            Self::ReLU => x.data.iter().zip(grad.data.to_mut().iter_mut()).for_each(|(v, g)| {
                *g = if *v > 0.0 { *g } else { 0.0 };
            }),
            Self::SiLU => x.data.iter().zip(grad.data.to_mut().iter_mut()).for_each(|(v, g)| {
                let sig = 1.0 / (1.0 + (-*v).exp());
                *g = sig + *v * sig * (1.0 - sig);
            }),
            Self::Softmax => {
                todo!("Softmax backward pass is not implemented yet");
            }
            Self::Tanh => x.data.iter().zip(grad.data.to_mut().iter_mut()).for_each(|(v, g)| {
                let t = v.tanh();
                *g = (1.0 - t * t) * (*g);
            }),
        }
        grad
    }
}
