use std::{fmt::Debug, io::{self, Read, Write}};

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
    fn pow(self, i: i32) -> Self;
    fn sum(&self) -> Self;
    fn neg(self) -> Self;
    fn max(self, y: f64) -> Self;
}

pub trait TensorInit {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)>;
}

/// a "sharp tensor" is just an extension of non-differentiable
/// tensor methods. well, in practice things like transpose
/// *are* differentiable, but nothing im working on needs the
/// gradient of a transpose yet so it's also here.
pub trait SharpTensor: Tensor {
    fn tranpose(&self, axes: &[usize]) -> Option<Self>;
    fn get_mut(&mut self, point: &[usize]) -> Option<&mut f64>;
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64>;
}

pub trait TensorIO: Tensor {
    fn read(read: &mut impl Read) -> io::Result<Self>;
    fn write(&self, write: &mut impl Write) -> io::Result<()>;
}

#[derive(Clone)]
pub enum Th {
    R(Vec<f64>),
    C(Vec<Self>),
}

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

#[derive(Clone, PartialEq)]
pub struct CPUTensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl CPUTensor {
    fn len(shape: &[usize]) -> usize {
        if shape.is_empty() {
            return 1;
        }
        shape.iter().cloned().reduce(|a, v| a * v).unwrap_or(0)
    }

    fn point_index(&self, of: &[usize]) -> Option<usize> {
        if of.len() != self.shape.len() {
            return None;
        }
        let mut idx = 0;
        let mut mult = 1;
        for (i, p) in of.iter().enumerate() {
            idx += mult * p;
            mult *= self.shape[i];
        }
        Some(idx)
    }

    fn arithmetic(&self, other: &Self, op: impl Fn(&f64, &f64) -> f64) -> Option<Self> {
        let mut new_shape = Vec::new();
        for i in 0..self.shape.len().max(other.shape.len()) {
            let lhs = self.shape.get(i).cloned().unwrap_or(1);
            let rhs = other.shape.get(i).cloned().unwrap_or(1);
            if lhs == rhs {
                new_shape.push(lhs);
            } else if lhs == 1 {
                new_shape.push(rhs);
            } else if rhs == 1 {
                new_shape.push(lhs);
            } else {
                return None;
            }
        }

        let mut new = Self {
            data: vec![0.0; Self::len(new_shape.as_slice())],
            shape: new_shape,
        };
        let mut new_point = vec![0; new.shape.len()];
        'iterate: loop {
            let lhs_point: Vec<usize> = new_point
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < self.shape.len())
                .map(|(i, x)| x % self.shape[i])
                .collect();
            let rhs_point: Vec<usize> = new_point
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < other.shape.len())
                .map(|(i, x)| x % other.shape[i])
                .collect();
            *new.get_mut(&new_point).unwrap() = op(
                self.get(&lhs_point).unwrap(),
                other.get(&rhs_point).unwrap(),
            );
            for (p, s) in new_point.iter_mut().zip(new.shape.iter()) {
                let o = *p;
                *p = (*p + 1) % s;
                if *p > o {
                    continue 'iterate;
                }
            }
            break;
        }
        Some(new)
    }


    fn debug_shape(&self, f: &mut std::fmt::Formatter<'_>, point: Vec<usize>) -> std::fmt::Result {
        if point.len() == self.ndim() {
            write![f, "{} ", self.get(point.as_slice()).unwrap()]?;
        }
        write![f, "["]?;
        for i in 0..self.shape[point.len()] {
            let mut new_point = point.clone();
            new_point.push(i);
            self.debug_shape(f, new_point)?;
        }
        write![f, "]\n"]?;
        Ok(())
    }
}

impl Tensor for CPUTensor {
    fn scalar(c: impl Into<f64>) -> Self {
        Self {
            shape: Default::default(),
            data: vec![c.into()],
        }
    }

    fn vector(v: impl Into<Vec<f64>>) -> Option<Self> {
        let data = v.into();
        if data.is_empty() {
            return None;
        }
        if data.len() == 1 {
            return Some(Self::scalar(data[0]));
        }
        Some(Self {
            shape: vec![data.len()],
            data,
        })
    }

    fn tensor(i: impl TensorInit) -> Option<Self> {
        let (shape, data) = i.make()?;
        if shape.iter().any(|i| *i == 0) {
            return None;
        }
        if shape.len() < 1 {
            return Self::vector(data);
        }
        Some(Self { data, shape })
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn get(&self, point: &[usize]) -> Option<&f64> {
        self.point_index(point).and_then(|i| self.data.get(i))
    }

    fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter()
    }

    fn add(&self, other: &Self) -> Option<Self> {
        self.arithmetic(other, |rhs, lhs| *rhs + *lhs)
    }

    fn sub(&self, other: &Self) -> Option<Self> {
        self.arithmetic(other, |rhs, lhs| *rhs - *lhs)
    }

    fn mul(&self, other: &Self) -> Option<Self> {
        self.arithmetic(other, |rhs, lhs| *rhs * *lhs)
    }

    fn dot(&self, other: &Self, depth: usize) -> Option<Self> {
        if depth > self.shape.len() || depth > other.shape.len() {
            return None;
        }
        let lhs_contraction = &self.shape[self.shape.len() - depth..];
        let rhs_contraction = &other.shape[..depth];
        if lhs_contraction != rhs_contraction {
            return None;
        }
        let contraction_shape = lhs_contraction.to_vec();
        let lhs_survivors = &self.shape[..self.shape.len() - depth];
        let rhs_survivors = &other.shape[depth..];
        let mut new_shape: Vec<usize> = lhs_survivors.to_vec();
        new_shape.extend_from_slice(rhs_survivors);

        let mut new = Self {
            data: vec![0.0; Self::len(new_shape.as_slice())],
            shape: new_shape,
        };

        let mut new_point = vec![0; new.shape.len()];
        let mut lhs_point = vec![0; self.ndim()];
        let mut rhs_point = vec![0; other.ndim()];
        'iterate: loop {
            lhs_point
                .iter_mut()
                .rev()
                .take(lhs_contraction.len())
                .for_each(|x| *x = 0);
            lhs_point
                .iter_mut()
                .take(lhs_survivors.len())
                .zip(&new_point)
                .for_each(|(x, y)| *x = *y);
            rhs_point
                .iter_mut()
                .take(lhs_contraction.len())
                .for_each(|x| *x = 0);
            rhs_point
                .iter_mut()
                .rev()
                .take(new_point.len() - lhs_survivors.len())
                .zip(new_point.iter().rev())
                .for_each(|(x, y)| *x = *y);
            let mut sum = 0.0;
            'summate: loop {
                sum += *self.get(&lhs_point).unwrap() * *other.get(&rhs_point).unwrap();

                for ((p_rhs, p_lhs), s) in lhs_point
                    .iter_mut()
                    .rev()
                    .take(lhs_contraction.len())
                    .zip(rhs_point.iter_mut().take(lhs_contraction.len()))
                    .zip(contraction_shape.iter())
                {
                    let o_rhs = *p_rhs;
                    *p_rhs = (*p_rhs + 1) % s;
                    *p_lhs = (*p_lhs + 1) % s;
                    if *p_rhs > o_rhs {
                        // both points should be synchronized so we can just check for the change in one
                        continue 'summate;
                    }
                }

                break;
            }

            *new.get_mut(&new_point).unwrap() = sum;

            for (p, s) in new_point.iter_mut().zip(new.shape.iter()) {
                let o = *p;
                *p = (*p + 1) % s;
                if *p > o {
                    continue 'iterate;
                }
            }

            break;
        }

        Some(new)
    }

    fn sum(&self) -> Self {
        Self::scalar(self.data.iter().sum::<f64>())
    }

    fn ln(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.ln());
        self
    }

    fn exp(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.exp());
        self
    }

    fn pow(mut self, i: i32) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.powi(i));
        self
    }

    fn neg(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = -*x);
        self
    }

    fn max(mut self, y: f64) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.max(y));
        self
    }
}

impl SharpTensor for CPUTensor {
    fn tranpose(&self, axes: &[usize]) -> Option<Self> {
        if self.shape.len() != axes.len() || axes.iter().any(|i| *i > self.shape.len()) {
            return None;
        }
        let new_shape: Vec<usize> = axes.iter().map(|j| self.shape[*j]).collect();
        let mut new = Self {
            data: vec![0.0; Self::len(new_shape.as_slice())],
            shape: new_shape,
        };

        let mut point = vec![0; self.shape.len()];
        'iterate: loop {
            let new_point: Vec<usize> = axes.iter().map(|i| point[*i]).collect();
            *new.get_mut(&new_point).unwrap() = *self.get(&point.as_slice()).unwrap();

            for (p, s) in point.iter_mut().zip(self.shape.iter()) {
                let o = *p;
                *p = (*p + 1) % s;
                if *p > o {
                    continue 'iterate;
                }
            }

            break;
        }

        Some(new)
    }

    fn get_mut(&mut self, point: &[usize]) -> Option<&mut f64> {
        self.point_index(point).and_then(|i| self.data.get_mut(i))
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.data.iter_mut()
    }
}

impl TensorIO for CPUTensor {
    fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read(&mut signature)?;
        if &signature != b"CPUTensr" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid tesnor signature",
            ));
        }
        let mut ndimb = [0u8; 8];
        read.read(&mut ndimb)?;
        let ndim = usize::from_le_bytes(ndimb);
        let mut shape = vec![0; ndim];
        for s in &mut shape {
            let mut dim = [0u8; 8];
            read.read(&mut dim)?;
            *s = usize::from_le_bytes(ndimb);
        }
        let size = Self::len(shape.as_slice()) * 8;
        let mut data = vec![0.0; size];
        for d in &mut data {
            let mut x = [0u8; 8];
            read.read(&mut x)?;
            *d = f64::from_le_bytes(x);
        }
        Ok(Self { shape, data })
    }

    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"CPUTensr")?;
        write.write_all(&self.shape.len().to_le_bytes())?;
        for s in &self.shape {
            write.write_all(&s.to_be_bytes())?;
        }
        for x in &self.data {
            write.write_all(&x.to_be_bytes())?;
        }
        Ok(())
    }
}

impl Debug for CPUTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.debug_shape(f, Vec::with_capacity(self.shape.len()))
    }
}