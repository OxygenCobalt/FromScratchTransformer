use std::io::{self, Read, Write};
use std::{borrow::Cow, collections::HashSet};

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub data: Vec<f64>,
}

pub enum Cast {
    ReshapeThenTranspose {
        reshape: Vec<usize>,
        transpose: Vec<usize>,
    },
    Reshape {
        reshape: Vec<usize>,
    },
    Transpose {
        transpose: Vec<usize>,
    },
}

impl Tensor {
    pub fn init(init: impl TensorInit) -> Option<Self> {
        let (shape, data) = init.make()?;
        let stride = stride_of(&shape);
        Some(Self {
            shape,
            stride,
            data,
        })
    }

    pub fn t(&self, axes: &[usize]) -> Result<TensorView, InvalidShape> {
        if self.shape.len() != axes.len() || axes.iter().any(|i| *i >= self.shape.len()) {
            return Err(InvalidShape);
        }
        let axis_set = axes.iter().copied().collect::<HashSet<usize>>();
        if axis_set.len() != self.shape.len() || axis_set != (0..self.shape.len()).collect() {
            return Err(InvalidShape);
        }
        let old_shape = self.shape.to_vec();
        let old_stride = self.stride.to_vec();
        let dim = self.shape.len();
        let mut new_shape = Vec::with_capacity(dim);
        unsafe {
            new_shape.set_len(dim);
        }
        let mut new_stride = Vec::with_capacity(dim);
        unsafe {
            new_stride.set_len(dim);
        }
        for (i, j) in axes.iter().enumerate() {
            new_shape[i] = old_shape[*j];
            new_stride[i] = old_stride[*j];
        }
        Ok(TensorView {
            tensor: self,
            shape: new_shape,
            stride: new_stride,
        })
    }

    pub fn r(&self, shape: &[usize]) -> Result<TensorView, ReshapeError> {
        if length_of(&self.shape) != length_of(&shape) {
            return Err(ReshapeError::InvalidShape);
        }
        if self.shape.is_empty() {
            // short circuit case for scalar so i dont have to deal
            // with it in general reshaping code
            // we can just return the same scalar since shape must be []
            // and stride is already []
            return Ok(TensorView {
                tensor: self,
                shape: self.shape.clone(),
                stride: self.stride.clone(),
            });
        }

        if is_row_major_contiguous(&self.shape, &self.stride) {
            return Ok(TensorView {
                tensor: self,
                shape: shape.to_vec(),
                stride: stride_of(shape),
            });
        }
        Err(ReshapeError::NotContiguous)
    }

    pub fn r_mut(&mut self, shape: &[usize]) -> Result<TensorViewMut, ReshapeError> {
        if length_of(&self.shape) != length_of(&shape) {
            return Err(ReshapeError::InvalidShape);
        }
        if self.shape.is_empty() {
            // short circuit case for scalar so i dont have to deal
            // with it in general reshaping code
            // we can just return the same scalar since shape must be []
            // and stride is already []
            return Ok(TensorViewMut {
                shape: self.shape.clone(),
                stride: self.stride.clone(),
                tensor: self,
            });
        }

        if is_row_major_contiguous(&self.shape, &self.stride) {
            return Ok(TensorViewMut {
                tensor: self,
                shape: shape.to_vec(),
                stride: stride_of(shape),
            });
        }
        Err(ReshapeError::NotContiguous)
    }

    pub fn materialize(&self) -> Tensor {
        let mut new_data = Vec::with_capacity(length_of(&self.shape));
        unsafe {
            new_data.set_len(length_of(&self.shape));
        }
        let mut new = Tensor {
            shape: self.shape.clone(),
            stride: stride_of(&self.shape),
            data: new_data,
        };
        let new_slice = new.data.as_mut_slice();
        let mut point = vec![0; self.shape.len()];
        let mut old_idx = 0;
        let mut new_idx = 0;
        'iterate: loop {
            unsafe {
                // todo: simd'd materialization
                *new_slice.get_unchecked_mut(new_idx) = *self.data.get_unchecked(old_idx);
            }
            for i in 0..self.shape.len() {
                if point[i] == self.shape[i] - 1 {
                    new_idx -= new.stride[i] * point[i];
                    old_idx -= self.stride[i] * point[i];
                    point[i] = 0;
                } else {
                    new_idx += new.stride[i];
                    old_idx += self.stride[i];
                    point[i] += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        new
    }

    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read_exact(&mut signature)?;
        if &signature != b"CPUTensr" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid tensor signature",
            ));
        }
        let mut ndimb = [0u8; 8];
        read.read_exact(&mut ndimb)?;
        let ndim = usize::from_le_bytes(ndimb);
        let mut shape = vec![0; ndim];
        for s in &mut shape {
            let mut dim = [0u8; 8];
            read.read_exact(&mut dim)?;
            *s = usize::from_le_bytes(dim);
        }
        let mut stride = vec![0; ndim];
        for s in &mut stride {
            let mut dim = [0u8; 8];
            read.read_exact(&mut dim)?;
            *s = usize::from_le_bytes(dim);
        }
        let size = length_of(shape.as_slice());
        let mut data = vec![0.0; size];
        for d in &mut data {
            let mut x = [0u8; 8];
            read.read_exact(&mut x)?;
            *d = f64::from_le_bytes(x);
        }

        Ok(Self {
            shape,
            stride,
            data,
        })
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"CPUTensr")?;
        write.write_all(&self.shape.len().to_le_bytes())?;
        for s in &self.shape {
            write.write_all(&s.to_le_bytes())?;
        }
        for s in &self.stride {
            write.write_all(&s.to_le_bytes())?;
        }
        for x in self.data.as_slice() {
            write.write_all(&x.to_le_bytes())?;
        }
        Ok(())
    }
}

pub struct TensorView<'a> {
    pub tensor: &'a Tensor,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

impl<'a> TensorView<'a> {
    pub fn t(&self, axes: &[usize]) -> Result<TensorView, InvalidShape> {
        if self.shape.len() != axes.len() || axes.iter().any(|i| *i >= self.shape.len()) {
            return Err(InvalidShape);
        }
        let axis_set = axes.iter().copied().collect::<HashSet<usize>>();
        if axis_set.len() != self.shape.len() || axis_set != (0..self.shape.len()).collect() {
            return Err(InvalidShape);
        }
        let old_shape = self.shape.to_vec();
        let old_stride = self.stride.to_vec();
        let dim = self.shape.len();
        let mut new_shape = Vec::with_capacity(dim);
        unsafe {
            new_shape.set_len(dim);
        }
        let mut new_stride = Vec::with_capacity(dim);
        unsafe {
            new_stride.set_len(dim);
        }
        for (i, j) in axes.iter().enumerate() {
            new_shape[i] = old_shape[*j];
            new_stride[i] = old_stride[*j];
        }
        Ok(TensorView {
            tensor: self.tensor,
            shape: new_shape,
            stride: new_stride,
        })
    }

    pub fn materialize(&self) -> Tensor {
        let mut new_data = Vec::with_capacity(length_of(&self.shape));
        unsafe {
            new_data.set_len(length_of(&self.shape));
        }
        let mut new = Tensor {
            shape: self.shape.clone(),
            stride: stride_of(&self.shape),
            data: new_data,
        };
        let new_slice = new.data.as_mut_slice();
        let mut point = vec![0; self.shape.len()];
        let mut old_idx = 0;
        let mut new_idx = 0;
        'iterate: loop {
            unsafe {
                // todo: simd'd materialization
                *new_slice.get_unchecked_mut(new_idx) = *self.tensor.data.get_unchecked(old_idx);
            }
            for i in 0..self.shape.len() {
                if point[i] == self.shape[i] - 1 {
                    new_idx -= new.stride[i] * point[i];
                    old_idx -= self.stride[i] * point[i];
                    point[i] = 0;
                } else {
                    new_idx += new.stride[i];
                    old_idx += self.stride[i];
                    point[i] += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        new
    }
}

pub struct TensorViewMut<'a> {
    pub tensor: &'a mut Tensor,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct InvalidShape;

pub fn stride_of(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

pub fn length_of(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[derive(Clone, Copy, Debug)]
pub struct NotView;

#[derive(Clone, Copy, Debug)]
pub enum ReshapeError {
    InvalidShape,
    NotContiguous,
}
fn is_row_major_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }
    if shape.len() != stride.len() {
        return false;
    }
    let mut expected = 1;
    for i in (0..shape.len()).rev() {
        if shape[i] != 1 && stride[i] != expected {
            return false;
        }
        expected = expected.saturating_mul(shape[i]);
    }
    true
}

pub trait TensorInit {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)>;
}

pub struct Tt(pub Vec<Tensor>);

impl TensorInit for Tt {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        if self.0.is_empty() {
            return None;
        }
        let base_shape = self.0.first().unwrap().shape.to_vec();
        if self.0.iter().any(|t| t.shape != base_shape) {
            return None;
        }
        let batch = self.0.len();
        let mut shape = base_shape.clone();
        shape.push(batch);

        if base_shape.is_empty() {
            let data = self
                .0
                .iter()
                .map(|t| *t.data.get(0).unwrap_or(&0.0))
                .collect();
            return Some((shape, data));
        }

        let sample_len = length_of(&base_shape);
        let mut data = Vec::with_capacity(sample_len * batch);
        let mut idx = vec![0usize; base_shape.len()];
        for step in 0..sample_len {
            for t in &self.0 {
                let mut lin = 0usize;
                for (axis, stride) in t.stride.iter().enumerate() {
                    lin += idx[axis] * stride;
                }
                data.push(*t.data.get(lin)?);
            }
            if step + 1 == sample_len {
                break;
            }
            for axis in (0..base_shape.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < base_shape[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
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

pub struct Scalar(pub f64);

impl TensorInit for Scalar {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        return Some((vec![], vec![self.0]));
    }
}

pub struct Vector(pub Vec<f64>);

impl TensorInit for Vector {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        return Some((vec![self.0.len()], self.0));
    }
}

pub struct Fill {
    pub shape: Vec<usize>,
    pub with: f64,
}

impl Fill {
    pub fn null(shape: Vec<usize>) -> Self {
        Self { shape, with: 0.0 }
    }
}

impl TensorInit for Fill {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)> {
        let data = vec![self.with; length_of(&self.shape)];
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
        data.resize_with(length_of(self.shape.as_slice()), self.with);
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
        let len = length_of(&self.shape);
        let mut data = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }
        Some((self.shape, data))
    }
}
