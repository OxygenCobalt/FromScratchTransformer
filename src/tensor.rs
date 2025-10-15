use core::f64;
use std::{
    fmt::Debug,
    io::{self, Read, Write},
};

use rayon::prelude::*;

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
    fn colify(&self, field: Field) -> Option<Self>;
    fn colmax(&self) -> Option<Self>;
    fn reshape(self, shape: &[usize]) -> Option<Self>;
    fn transpose(&self, axes: &[usize]) -> Option<Self>;
    fn at_argmax(&self, of: &Self) -> Option<Self>;
}

#[derive(Clone, Copy)]
pub struct Field {
    pub size: usize,
    pub stride: usize,
    pub padding: usize
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

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"CnvField")?;
        write.write_all(&self.size.to_le_bytes())?;
        write.write_all(&self.stride.to_le_bytes())?;
        Ok(())
    }
}

pub trait TensorInit {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)>;
}

pub trait TensorMut: Tensor {
    fn get_mut(&mut self, point: &[usize]) -> Option<&mut f64>;
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64>;
}

pub trait TensorIO: Tensor {
    fn read(read: &mut impl Read) -> io::Result<Self>;
    fn write(&self, write: &mut impl Write) -> io::Result<()>;
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

#[derive(Clone, PartialEq)]
pub struct CPUTensor {
    shape: Vec<usize>,
    stride: Vec<usize>,
    data: Vec<f64>,
}

impl CPUTensor {
    pub fn len(shape: &[usize]) -> usize {
        if shape.is_empty() {
            return 1;
        }
        shape.iter().cloned().reduce(|a, v| a * v).unwrap_or(0)
    }

    fn point_index(&self, of: &[usize]) -> Option<usize> {
        if of.len() != self.shape.len() {
            return None;
        }
        let idx = match of.len() {
            // some fast cases for common tensor dimensions
            1 => self.stride[0] * of[0],
            2 => self.stride[0] * of[0] + self.stride[1] * of[1],
            3 => self.stride[0] * of[0] + self.stride[1] * of[1] + self.stride[2] * of[2],
            4 => self.stride[0] * of[0] + self.stride[1] * of[1] + self.stride[2] * of[2] + self.stride[3] * of[3],
            5 => self.stride[0] * of[0] + self.stride[1] * of[1] + self.stride[2] * of[2] + self.stride[3] * of[3] + self.stride[4] * of[4],
            6 => self.stride[0] * of[0] + self.stride[1] * of[1] + self.stride[2] * of[2] + self.stride[3] * of[3] + self.stride[4] * of[4] + self.stride[5] * of[5],
            _ => {
                let mut idx = 0;
                for i in 0..of.len() {
                    idx += self.stride[i] * of[i];
                }
                idx
            }
        };
        Some(idx)
    }

    fn arithmetic(&self, other: &Self, op: impl Fn(&f64, &f64) -> f64) -> Option<Self> {
        let mut new_shape = Vec::new();
        let mut lhs_strides = Vec::new();
        let mut rhs_strides = Vec::new();
        for i in 0..self.shape.len().max(other.shape.len()) {
            let lhs = self.shape.get(i).cloned().unwrap_or(1);
            let rhs = other.shape.get(i).cloned().unwrap_or(1);
            if lhs == rhs {
                new_shape.push(lhs);
                lhs_strides.push(self.stride[i]);
                rhs_strides.push(other.stride[i]);
            } else if lhs == 1 {
                new_shape.push(rhs);
                lhs_strides.push(0);
                rhs_strides.push(other.stride[i]);
            } else if rhs == 1 {
                new_shape.push(lhs);
                lhs_strides.push(self.stride[i]);
                rhs_strides.push(0);
            } else {
                return None;
            }
        }

        let mut new = Self::tensor(Fill::null(new_shape)).unwrap();
        let mut new_point = vec![0; new.shape.len()];

        let mut new_idx = 0;
        let mut lhs_idx = 0;
        let mut rhs_idx = 0;

        'iterate: loop {
            new.data[new_idx] = op(
                &self.data[lhs_idx],
                &other.data[rhs_idx]
            );
            for i in 0..new_point.len() {
                if new_point[i] == new.shape[i] - 1 {
                    new_point[i] = 0;
                    new_idx -= new.stride[i] * (new.shape[i] - 1);
                    lhs_idx -= lhs_strides[i] * (new.shape[i] - 1);
                    rhs_idx -= rhs_strides[i] * (new.shape[i] - 1);
                } else {
                    new_point[i] += 1;
                    new_idx += new.stride[i];
                    lhs_idx += lhs_strides[i];
                    rhs_idx += rhs_strides[i];
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
            return Ok(())
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
            stride: Default::default(),
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
            stride: vec![1],
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
        let mut stride = Vec::new();
        let mut i = 0;
        let mut mult = 1;
        stride.resize_with(shape.len(), || {
            let x = mult;
            mult *= shape[i];
            i += 1;
            x
        });
        
        Some(Self { data, shape, stride })
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
        let contraction_shape = lhs_contraction;

        let lhs_survivors = &self.shape[..self.shape.len() - depth];
        let rhs_survivors = &other.shape[depth..];
        let lhs_survivor_len = lhs_survivors.len();
        let rhs_survivor_len = rhs_survivors.len();
        let mut new_shape: Vec<usize> = lhs_survivors.to_vec();
        new_shape.extend_from_slice(rhs_survivors);
        let mut new = Self::tensor(Fill::null(new_shape)).unwrap();
        let mut new_point = vec![0; new.shape.len()];
        let mut new_idx = 0;
        let mut lhs_idx = 0;
        let mut rhs_idx = 0;
        let mut contraction_point = vec![0; contraction_shape.len()];

        let k: usize = contraction_shape.iter().product();

        let lhs_contraction_stride = &self.stride[self.ndim() - depth..];
        let mut lhs_contract_offsets: Vec<usize> = vec![];
        let rhs_contraction_stride = &other.stride[..depth];
        let mut rhs_contract_offsets: Vec<usize> = vec![];

        let mut lhs_contract_idx = 0;
        let mut rhs_contract_idx = 0;
        'precompute: loop {
            lhs_contract_offsets.push(lhs_contract_idx);
            rhs_contract_offsets.push(rhs_contract_idx);
            for i in 0..contraction_point.len() {
                if contraction_point[i] == contraction_shape[i] - 1 {
                    contraction_point[i] = 0;
                    lhs_contract_idx -= lhs_contraction_stride[i] * (contraction_shape[i] - 1);
                    rhs_contract_idx -= rhs_contraction_stride[i] * (contraction_shape[i] - 1);
                } else {
                    contraction_point[i] += 1;
                    lhs_contract_idx += lhs_contraction_stride[i];
                    rhs_contract_idx += rhs_contraction_stride[i];
                    continue 'precompute;
                }
            }
            break;
        }
        'iterate: loop {
            let mut sum = 0.0;
            for i in 0..k {
                sum += self.data[lhs_idx + lhs_contract_offsets[i]] * other.data[rhs_idx + rhs_contract_offsets[i]];
            }
            new.data[new_idx] = sum;
            
            for (i, (p, s)) in new_point.iter_mut().zip(new.shape.iter()).enumerate() {
                if *p == *s - 1 {
                    *p = 0;
                    new_idx -= new.stride[i] * (new.shape[i] - 1);
                    if i < lhs_survivor_len {
                        lhs_idx -= self.stride[i] * (self.shape[i] - 1);
                    }
                    if i >= lhs_survivor_len && rhs_survivor_len > 0 {
                        let rhs_axis = i - lhs_survivor_len;
                        let rhs_c = depth + rhs_axis;
                        rhs_idx -= other.stride[rhs_c] * (other.shape[rhs_c] - 1);
                    }
                } else {
                    *p += 1;
                    new_idx += new.stride[i];
                    if i < lhs_survivor_len {
                        lhs_idx += self.stride[i];
                    }
                    if i >= lhs_survivor_len {
                        let rhs_axis = i - lhs_survivor_len;
                        let rhs_c = depth + rhs_axis;
                        rhs_idx += other.stride[rhs_c];
                    }
                    continue 'iterate;
                }
            }

            break;
        }

        Some(new)
    }

    fn sum(&self) -> Self {
        if self.ndim() == 0 {
            return self.clone();
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(0);
        let mut new = Self::tensor(Fill::null(new_shape)).unwrap();
        let mut new_point = vec![0; new.shape.len()];
        let mut column_point = vec![0; self.shape.len()];
        'iterate: loop {
            let mut sum = 0.0;
            column_point[1..].copy_from_slice(&new_point);
            for i in 0..self.shape[0] {
                column_point[0] = i;
                sum += *self.get(&column_point).unwrap();
            }
            *new.get_mut(&new_point).unwrap() = sum;
            for (p, s) in new_point.iter_mut().zip(new.shape.iter()) {
                if *p == *s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        new
    }

    fn ln(mut self) -> Self {
        self.data.par_iter_mut().for_each(|x| *x = x.ln());
        self
    }

    fn exp(mut self) -> Self {
        self.data.par_iter_mut().for_each(|x| *x = x.exp());
        self
    }

    fn pow(mut self, i: i32) -> Self {
        self.data.par_iter_mut().for_each(|x| *x = x.powi(i));
        self
    }

    fn neg(mut self) -> Self {
        self.data.par_iter_mut().for_each(|x| *x = -*x);
        self
    }

    fn max(mut self, y: f64) -> Self {
        self.data.par_iter_mut().for_each(|x| *x = x.max(y));
        self
    }

    fn colify(&self, field: Field) -> Option<Self> {
        if self.ndim() < 2 {
            return None;
        }
        // TODO: remove invariant for square imgs
        if self.shape[0] != self.shape[1] {
            return None;
        }
        let locations = field.locations_on(*self.shape.first().unwrap())?;
        let new_shape: Vec<usize> = [field.size * field.size, locations * locations]
            .into_iter()
            .chain(self.shape[2..].iter().cloned())
            .collect();
        let mut new = Self::tensor(Fill::null(new_shape)).unwrap();
        let mut new_point = vec![0; new.shape.len()];
        let mut old_point = vec![0; self.ndim()];
        'iterate: loop {
            let field_idx = new_point[0];
            let location_idx = new_point[1];
            let x = -(field.padding as i64) + (((location_idx % locations) * field.stride) + (field_idx % field.size)) as i64;
            let y = -(field.padding as i64) + (((location_idx / locations) * field.stride) + (field_idx / field.size)) as i64;
            *new.get_mut(&new_point).unwrap() = if x >= 0 && y >= 0 && x < self.shape[0] as i64 && y < self.shape[1] as i64 {
                old_point[0] = x as usize;
                old_point[1] = y as usize;
                old_point[2..].copy_from_slice(&new_point[2..]);
                *self.get(&old_point).unwrap()
            } else {
                0.0
            };

            for (p, s) in new_point.iter_mut().zip(new.shape.iter()) {
                if *p == *s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        Some(new)
    }

    fn colmax(&self) -> Option<Self> {
        if self.ndim() == 0 {
            return None;
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(0);
        let mut new = Self::tensor(Fill {
            shape: new_shape.clone(),
            with: f64::NEG_INFINITY
        }).unwrap();
        let mut new_point = vec![0; new_shape.len()];
        let mut column_point = vec![0; self.shape.len()];
        'iterate: loop {
            column_point[1..].copy_from_slice(&new_point);
            for i in 0..self.shape[0] {
                column_point[0] = i;
                if *self.get(&column_point).unwrap() > *new.get(&new_point).unwrap() {
                    *new.get_mut(&new_point).unwrap() = *self.get(&column_point).unwrap();
                }
            }
            for (p, s) in new_point.iter_mut().zip(new.shape.iter()) {
                if *p == *s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        Some(new)
    }

    fn reshape(mut self, shape: &[usize]) -> Option<Self> {
        if Self::len(shape) != Self::len(&self.shape) {
            return None;
        }
        self.shape = shape.to_vec();
        self.stride.clear();

        let mut i = 0;
        let mut mult = 1;
        self.stride.resize_with(shape.len(), || {
            let x = mult;
            mult *= shape[i];
            i += 1;
            x
        });
        Some(self)
    }

    fn transpose(&self, axes: &[usize]) -> Option<Self> {
        if self.shape.len() != axes.len() || axes.iter().any(|i| *i > self.shape.len()) {
            return None;
        }
        let new_shape: Vec<usize> = axes.iter().map(|j| self.shape[*j]).collect();
        let mut new = Self::tensor(Fill::null(new_shape)).unwrap();

        let mut point = vec![0; self.shape.len()];
        let mut new_point = vec![0; new.shape.len()];
        'iterate: loop {
            new_point
                .iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = point[axes[i]]);
            *new.get_mut(&new_point).unwrap() = *self.get(&point.as_slice()).unwrap();

            for i in 0..self.ndim() {
                let (p, s) = (&mut point[i], self.shape[i]);
                if *p == s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }

            break;
        }

        Some(new)
    }

    fn at_argmax(&self, of: &Self) -> Option<Self> {
        if self.shape() != of.shape() {
            return None;
        }
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape.remove(0);
        let mut new = Self::tensor(Fill::null(new_shape)).unwrap();
        let mut new_point = vec![0; new.shape.len()];
        let mut column_point = vec![0; self.shape.len()];
        'iterate: loop {
            let mut argmax = 0;
            let mut max = f64::NEG_INFINITY;
            column_point[1..].copy_from_slice(&new_point);
            for i in 0..self.shape[0] {
                column_point[0] = i;
                let x = *of.get(&column_point).unwrap();
                if x > max {
                    argmax = i;
                    max = x;
                }
            }
            column_point[0] = argmax;
            *new.get_mut(&new_point).unwrap() = *self.get(&column_point).unwrap();
            for (p, s) in new_point.iter_mut().zip(new.shape.iter()) {
                if *p == *s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        Some(new)
    }
}

impl TensorMut for CPUTensor {
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
                "invalid tensor signature",
            ));
        }
        let mut ndimb = [0u8; 8];
        read.read(&mut ndimb)?;
        let ndim = usize::from_le_bytes(ndimb);
        let mut shape = vec![0; ndim];
        for s in &mut shape {
            let mut dim = [0u8; 8];
            read.read(&mut dim)?;
            *s = usize::from_le_bytes(dim);
        }
        let mut stride = vec![0; ndim];
        for s in &mut stride {
            let mut dim = [0u8; 8];
            read.read(&mut dim)?;
            *s = usize::from_le_bytes(dim);
        }
        let size = Self::len(shape.as_slice());
        let mut data = vec![0.0; size];
        for d in &mut data {
            let mut x = [0u8; 8];
            read.read(&mut x)?;
            *d = f64::from_le_bytes(x);
        }
        Ok(Self { shape, stride, data })
    }

    fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"CPUTensr")?;
        write.write_all(&self.shape.len().to_le_bytes())?;
        for s in &self.shape {
            write.write_all(&s.to_le_bytes())?;
        }
        for s in &self.stride {
            write.write_all(&s.to_le_bytes())?;
        }
        for x in &self.data {
            write.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn field_locations_on_counts_valid_stride_windows() {
//         let field = Field {
//             size: 2,
//             stride: 1,
//             padding: 0,
//         };

//         assert_eq!(field.locations_on(4).unwrap(), 3);
//     }

//     #[test]
//     fn colify_collects_all_stride_one_patches_without_padding() {
//         let input = CPUTensor {
//             shape: vec![4, 4],
//             data: (0..16).map(|v| v as f64).collect(),
//         };
//         let field = Field {
//             size: 2,
//             stride: 1,
//             padding: 0,
//         };

//         let actual = input.colify(field).unwrap();

//         assert_eq!(actual.shape(), &[4, 9]);

//         let expected_columns = vec![
//             vec![0.0, 1.0, 4.0, 5.0],
//             vec![1.0, 2.0, 5.0, 6.0],
//             vec![2.0, 3.0, 6.0, 7.0],
//             vec![4.0, 5.0, 8.0, 9.0],
//             vec![5.0, 6.0, 9.0, 10.0],
//             vec![6.0, 7.0, 10.0, 11.0],
//             vec![8.0, 9.0, 12.0, 13.0],
//             vec![9.0, 10.0, 13.0, 14.0],
//             vec![10.0, 11.0, 14.0, 15.0],
//         ];

//         for (location_idx, column) in expected_columns.iter().enumerate() {
//             for (field_idx, expected_value) in column.iter().enumerate() {
//                 println!("{} {} {:?} {:?}", location_idx, field_idx, actual.get(&[field_idx, location_idx]).copied(), Some(*expected_value));
//                 assert_eq!(
//                     actual.get(&[field_idx, location_idx]).copied(),
//                     Some(*expected_value)
//                 );
//             }
//         }
//     }

//     #[test]
//     fn colify_obeys_strides_greater_than_one() {
//         let input = CPUTensor {
//             shape: vec![4, 4],
//             data: (0..16).map(|v| v as f64).collect(),
//         };
//         let field = Field {
//             size: 2,
//             stride: 2,
//             padding: 0,
//         };

//         let actual = input.colify(field).unwrap();

//         assert_eq!(actual.shape(), &[4, 4]);

//         let expected_columns = vec![
//             vec![0.0, 1.0, 4.0, 5.0],
//             vec![2.0, 3.0, 6.0, 7.0],
//             vec![8.0, 9.0, 12.0, 13.0],
//             vec![10.0, 11.0, 14.0, 15.0],
//         ];

//         for (location_idx, column) in expected_columns.iter().enumerate() {
//                 // println!("{} {} {:?} {:?}", location_idx, field_idx, actual.get(&[field_idx, location_idx]).copied(), Some(*expected_value));
//             for (field_idx, expected_value) in column.iter().enumerate() {
//                 assert_eq!(
//                     actual.get(&[field_idx, location_idx]).copied(),
//                     Some(*expected_value)
//                 );
//             }
//         }
//     }

//     #[test]
//     fn colify_injects_zero_padding_into_patches() {
//         let input = CPUTensor {
//             shape: vec![2, 2],
//             data: vec![1.0, 2.0, 3.0, 4.0],
//         };
//         let field = Field {
//             size: 2,
//             stride: 1,
//             padding: 1,
//         };

//         let actual = input.colify(field).unwrap();

//         assert_eq!(actual.shape(), &[4, 9]);

//         // [0 0 0 0]
//         // [0 1 2 0]
//         // [0 3 4 0]
//         // [0 0 0 0]

//         let expected_columns = vec![
//             vec![0.0, 0.0, 0.0, 1.0],
//             vec![0.0, 0.0, 1.0, 2.0],
//             vec![0.0, 0.0, 2.0, 0.0],
//             vec![0.0, 1.0, 0.0, 3.0],
//             vec![1.0, 2.0, 3.0, 4.0],
//             vec![2.0, 0.0, 4.0, 0.0],
//             vec![0.0, 3.0, 0.0, 0.0],
//             vec![3.0, 4.0, 0.0, 0.0],
//             vec![4.0, 0.0, 0.0, 0.0],
//         ];

//         for (location_idx, column) in expected_columns.iter().enumerate() {
//             for (field_idx, expected_value) in column.iter().enumerate() {
//                 println!("{} {} {:?} {:?}", location_idx, field_idx, actual.get(&[field_idx, location_idx]).copied(), Some(*expected_value));
//                 assert_eq!(
//                     actual.get(&[field_idx, location_idx]).copied(),
//                     Some(*expected_value)
//                 );
//             }
//         }
//     }
// }

// impl Debug for CPUTensor {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         self.debug_shape(f, Vec::with_capacity(self.shape.len()))
//     }
// }
