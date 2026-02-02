use crate::tensor::cpu2::Tensor;

pub mod ff;

pub enum Axon<'a> {
    Dense(ff::FeedForward<'a>),
}

impl <'a> Axon<'a> {
    pub fn forward<'i, 'o>(&self, a_in: Tensor<'i>) -> Tensor<'o> {
        match self {
            Self::Dense(ff) => ff.forward(a_in)
        }
    }

    pub fn backward<'i, 'o>(
        &mut self,
        c: f64,
        a_in: &Tensor<'i>,
        a_out: &Tensor<'i>,
        grad: Tensor<'i>,
    ) -> Tensor<'o> {
        match self {
            Self::Dense(ff) => ff.backward(c, a_in, a_out, grad)
        }
    }
}