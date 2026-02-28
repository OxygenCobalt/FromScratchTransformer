use crate::{ml::axons::ff::FeedForward, tensor::cpu2::Tensor};
use std::io::{self, Read, Write};

pub mod act;
pub mod emb;
pub mod ff;

pub enum Axon {
    Dense(ff::FeedForward),
    Activation(act::Activation),
    Embeddings(emb::Embeddings),
}

impl Axon {
    pub fn prepare(&mut self, train: bool, batch: usize) {
        match self {
            Self::Dense(ff) => {}
            Self::Activation(act) => act.prepare(train, batch),
            Self::Embeddings(emb) => {}
        }
    }

    pub fn forward(&self, a_in: Tensor) -> Tensor {
        match self {
            Self::Dense(ff) => ff.forward(a_in),
            Self::Activation(act) => act.forward(a_in),
            Self::Embeddings(emb) => emb.forward(a_in),
        }
    }

    pub fn backward<'i>(&mut self, c: f64, a_in: Tensor, grad: Tensor) -> Tensor {
        match self {
            Self::Dense(ff) => ff.backward(c, a_in, grad),
            Self::Activation(act) => act.backward(c, a_in, grad),
            Self::Embeddings(emb) => emb.forward(a_in),
        }
    }
}

impl Axon {
    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut id = [0u8; 8];
        read.read_exact(&mut id)?;
        match &id {
            b"AxonDnse" => Ok(Self::Dense(FeedForward::read(read)?)),
            b"AxonActv" => Ok(Self::Activation(act::Activation::read(read)?)),
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
            Self::Activation(act) => {
                write.write_all(b"AxonActv")?;
                act.write(write)
            }
            Self::Embeddings(emb) => Ok(()),
        }
    }
}
