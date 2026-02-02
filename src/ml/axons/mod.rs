use crate::{ml::axons::ff::FeedForward, tensor::cpu2::Tensor};
use std::io::{self, Read, Write};

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

impl <'a> Axon<'a> {
    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut id = [0u8; 8];
        read.read_exact(&mut id)?;
        match &id {
            b"AxonDnse" => Ok(Self::Dense(FeedForward::read(read)?)),
            _ => Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid axon signature",
            )),
        }
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        match self {
            Self::Dense(ff) => {
                write.write_all(b"AxonDnse")?;
                ff.write(write)
            }
        }
    }
}