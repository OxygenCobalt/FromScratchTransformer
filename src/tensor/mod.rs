pub mod cpu;

use core::f64;
use std::{
    io::{self, Read, Write}
};

use crate::tensor::cpu::CPUTensor;

pub trait Tensor
where
    Self: Sized + Clone,
{
    fn scalar(c: impl Into<f64>) -> Self;
    fn vector(v: impl Into<Vec<f64>>) -> Option<Self>;
    fn tensor(i: impl TensorInit) -> Option<Self>;
    fn ndim(&self) -> usize;
    fn shape(&self) -> &[usize];
    fn get(&self, point: &[usize]) -> Option<&f64>;
    fn iter(&self) -> impl Iterator<Item = &f64>;
    fn add(&self, other: &Self) -> Option<Self>;
    fn sub(&self, other: &Self) -> Option<Self>;
    fn mul(&self, other: &Self) -> Option<Self>;
    fn dot(&self, other: &Self, axes: usize) -> Option<Self>;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn tanh(self) -> Self;
    fn pow(self, i: i32) -> Self;
    fn sum(&self) -> Self;
    fn neg(self) -> Self;
    fn max(self, y: f64) -> Self;
    fn cols_at(&self, indices: &Self) -> Option<Self>;
    fn colify(&self, field: Field) -> Option<Self>;
    fn colmax(&self) -> Option<Self>;
    fn reshape(self, shape: &[usize]) -> Option<Self>;
    fn transpose(self, axes: &[usize]) -> Option<Self>;
    fn at_argmax(&self, of: &Self) -> Option<Self>;
    fn softmax(self) -> Option<Self>;
}

pub trait DifferentiableTensor: Tensor {
    type Autograd: Autograd<Self>;
    fn autograd(self) -> Self::Autograd;
}

pub trait Autograd<Parent>: Tensor {
    fn backward(self);
    fn into_grad(self) -> Option<Parent>;
}

pub trait TensorMut: Tensor {
    fn get_mut(&mut self, point: &[usize]) -> Option<&mut f64>;
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64>;
}

pub trait TensorIO: Tensor {
    fn read(read: &mut impl Read) -> io::Result<Self>;
    fn write(&self, write: &mut impl Write) -> io::Result<()>;
}

#[derive(Clone, Copy)]
pub struct Field {
    pub size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Field {
    pub fn locations_on(&self, size: usize) -> Option<usize> {
        let numer = size - self.size + self.padding + self.padding;
        let denom = self.stride;
        if numer % denom != 0 {
            return None;
        }
        Some((numer / denom) + 1)
    }

    pub fn read(read: &mut impl Read) -> io::Result<Field> {
        let mut buf = [0u8; 8];
        read.read_exact(&mut buf)?;
        if &buf != b"CnvField" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid field signature",
            ));
        }
        read.read_exact(&mut buf)?;
        let size = usize::from_le_bytes(buf);
        read.read_exact(&mut buf)?;
        let stride = usize::from_le_bytes(buf);
        read.read_exact(&mut buf)?;
        let padding = usize::from_le_bytes(buf);
        Ok(Field {
            size,
            stride,
            padding,
        })
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"CnvField")?;
        write.write_all(&self.size.to_le_bytes())?;
        write.write_all(&self.stride.to_le_bytes())?;
        write.write_all(&self.padding.to_le_bytes())?;
        Ok(())
    }
}

pub trait TensorInit {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)>;
}

pub struct Tt<T: Tensor>(pub Vec<T>);

impl<T: Tensor> TensorInit for Tt<T> {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        if self.0.is_empty() {
            return None;
        }
        let mut shape = self.0.first().unwrap().shape().to_vec();
        if self.0.iter().any(|t| t.shape() != shape) {
            return None;
        }
        shape.push(self.0.len());
        let data = self.0.iter().map(|t| t.iter()).flatten().cloned().collect();
        Some((shape, data))
    }
}

#[derive(Clone)]
#[cfg(test)]
pub enum Th {
    R(Vec<f64>),
    C(Vec<Self>),
}

#[cfg(test)]
impl TensorInit for Th {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        let mut shape: Vec<usize> = Vec::new();
        let mut stack = Vec::new();
        stack.push((self, 0));
        let mut data = Vec::new();
        while let Some((_, depth)) = stack.get(0) {
            let depth = *depth;
            let len = match stack.remove(0) {
                (Th::R(mut row), _) => {
                    let len = row.len();
                    data.append(&mut row);
                    len
                }
                (Th::C(col), depth) => {
                    let len = col.len();
                    stack.extend(col.into_iter().map(|tv| (tv, depth + 1)));
                    len
                }
            };
            match shape.get(depth) {
                Some(existing) => {
                    if *existing != len {
                        return None;
                    }
                }
                None => {
                    shape.resize(depth + 1, 0);
                    shape[depth] = len;
                }
            }
        }
        shape.reverse();
        Some((shape, data))
    }
}

pub struct Fill {
    pub shape: Vec<usize>,
    pub with: f64,
}

impl Fill {
    fn null(shape: Vec<usize>) -> Self {
        Self { shape, with: 0.0 }
    }
}

impl TensorInit for Fill {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        let data = vec![self.with; CPUTensor::len(&self.shape)];
        return Some((self.shape, data));
    }
}

pub struct Generate<F: FnMut() -> f64> {
    pub shape: Vec<usize>,
    pub with: F,
}

impl<F: FnMut() -> f64> TensorInit for Generate<F> {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        let mut data = Vec::new();
        data.resize_with(CPUTensor::len(self.shape.as_slice()), self.with);
        return Some((self.shape, data));
    }
}

pub struct FillUninit {
    pub shape: Vec<usize>,
}

impl FillUninit {
    pub unsafe fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl TensorInit for FillUninit {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        let len = CPUTensor::len(&self.shape);
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }
        Some((self.shape, data))
    }
}