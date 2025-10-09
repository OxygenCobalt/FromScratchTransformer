use crate::tensor::Tensor;

#[derive(Clone, Copy)]
pub enum Activation {
    Sigmoid,
    ReLU,
    SiLU,
    Softmax,
}

impl Activation {
    pub fn id(&self) -> &'static [u8; 8] {
        match self {
            Self::Sigmoid => b"ActvSigm",
            Self::ReLU => b"ActvReLU",
            Self::SiLU => b"ActvSiLU",
            Self::Softmax => b"ActvSfmx",
        }
    }

    pub fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"ActvSigm" => Some(Self::Sigmoid),
            b"ActvReLU" => Some(Self::ReLU),
            b"ActvSiLU" => Some(Self::SiLU),
            b"ActvSfmx" => Some(Self::Softmax),
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
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{activation::Activation, tensor::{CPUTensor, Tensor, Th}};

    #[test]
    fn whywhyhwy() {
        let softmax = Activation::Softmax.activate(CPUTensor::tensor(Th::C(vec![Th::R(vec![1.0, 2.0]), Th::R(vec![1.0, 2.0])])).unwrap());
        assert_eq!(1.0, *softmax.get(&[0, 0]).unwrap() + *softmax.get(&[1, 0]).unwrap())
    }

    #[test]
    fn softmax_columns_sum_to_one() {
        use crate::tensor::Tt;

        let v1 = CPUTensor::vector(vec![1.0, 10.0, 100.0]).unwrap();
        let v2 = CPUTensor::vector(vec![2.0, 20.0, 200.0]).unwrap();
        let logits = CPUTensor::tensor(Tt(vec![v1, v2])).unwrap();
        assert_eq!(logits.shape(), &[3, 2]);

        let probs = Activation::Softmax.activate(logits);
        let col0_sum = probs.get(&[0, 0]).unwrap()
            + probs.get(&[1, 0]).unwrap()
            + probs.get(&[2, 0]).unwrap();
        assert!((col0_sum - 1.0).abs() < 1e-6);

        let col1_sum = probs.get(&[0, 1]).unwrap()
            + probs.get(&[1, 1]).unwrap()
            + probs.get(&[2, 1]).unwrap();
        assert!((col1_sum - 1.0).abs() < 1e-6);
    }
}
