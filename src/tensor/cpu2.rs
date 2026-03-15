use std::io::{self, Read, Write};
use std::{borrow::Cow, collections::HashSet};

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub data: Vec<f64>,
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
        let mut base_shape = self.0.first().unwrap().shape.to_vec();
        if self.0.iter().any(|t| t.shape != base_shape) {
            return None;
        }
        let mut base_stride = self.0.first().unwrap().stride.to_vec();
        let batch = self.0.len();
        let mut shape = vec![self.0.len()];
        shape.extend_from_slice(&mut base_shape);
        let mut stride = stride_of(&shape);

        if base_shape.is_empty() {
            let data = self
                .0
                .iter()
                .map(|t| *t.data.get(0).unwrap_or(&0.0))
                .collect();
            return Some((shape, data));
        }

        let mut len = length_of(&shape);
        let mut data: Vec<f64> = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }
        let mut point = vec![0; base_shape.len() + 1];
        let mut idx = 0;
        let mut local_idx = 0;
        'iterate: loop {
            data[idx] = self.0[point[0]].data[local_idx];
            for i in 0..point.len() {
                if point[i] == shape[i] - 1 {
                    idx -= stride[i] * point[i];
                    if i > 0 {
                        local_idx -= base_stride[i - 1] * point[i];
                    }
                    point[i] = 0;
                } else {
                    point[i] += 1;
                    idx += stride[i];
                    if i > 0 {
                        local_idx += base_stride[i - 1];
                    }
                    continue 'iterate;
                }
            }
            break;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_1d(data: &[f64]) -> Tensor {
        Tensor::init(Vector(data.to_vec())).unwrap()
    }

    fn tensor_with(shape: Vec<usize>, data: &[f64]) -> Tensor {
        let mut t = Tensor::init(Fill { shape, with: 0.0 }).unwrap();
        t.data.as_mut_slice().clone_from_slice(data);
        t
    }

    #[test]
    fn tt_scalars() {
        let a = Tensor::init(Scalar(1.0)).unwrap();
        let b = Tensor::init(Scalar(2.0)).unwrap();
        let stacked = Tensor::init(Tt(vec![a, b])).unwrap();
        assert_eq!(stacked.shape, vec![2]);
        assert_eq!(stacked.data, vec![1.0, 2.0]);
    }

    #[test]
    fn tt_two_vectors() {
        let a = tensor_1d(&[1.0, 2.0, 3.0]);
        let b = tensor_1d(&[4.0, 5.0, 6.0]);
        let stacked = Tensor::init(Tt(vec![a, b])).unwrap();
        // batch-first: shape [batch, features]
        assert_eq!(stacked.shape, vec![2, 3]);
        // row-major: batch 0 then batch 1 contiguous
        assert_eq!(stacked.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tt_three_vectors() {
        let a = tensor_1d(&[1.0, 2.0]);
        let b = tensor_1d(&[3.0, 4.0]);
        let c = tensor_1d(&[5.0, 6.0]);
        let stacked = Tensor::init(Tt(vec![a, b, c])).unwrap();
        assert_eq!(stacked.shape, vec![3, 2]);
        assert_eq!(stacked.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn tt_two_matrices() {
        // Two 2x3 matrices stacked → [2, 2, 3]
        let a = tensor_with(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = tensor_with(vec![2, 3], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let stacked = Tensor::init(Tt(vec![a, b])).unwrap();
        assert_eq!(stacked.shape, vec![2, 2, 3]);
        // row-major batch-first: a's data then b's data contiguously
        assert_eq!(
            stacked.data,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }

    #[test]
    fn tt_single_tensor() {
        let a = tensor_1d(&[10.0, 20.0]);
        let stacked = Tensor::init(Tt(vec![a])).unwrap();
        assert_eq!(stacked.shape, vec![1, 2]);
        assert_eq!(stacked.data, vec![10.0, 20.0]);
    }

    #[test]
    fn tt_empty_returns_none() {
        assert!(Tensor::init(Tt(vec![])).is_none());
    }

    #[test]
    fn tt_mismatched_shapes_returns_none() {
        let a = tensor_1d(&[1.0, 2.0]);
        let b = tensor_1d(&[3.0, 4.0, 5.0]);
        assert!(Tensor::init(Tt(vec![a, b])).is_none());
    }

    #[test]
    fn tt_stride_is_row_major() {
        let a = tensor_1d(&[1.0, 2.0, 3.0]);
        let b = tensor_1d(&[4.0, 5.0, 6.0]);
        let stacked = Tensor::init(Tt(vec![a, b])).unwrap();
        // [2, 3] row-major → stride [3, 1]
        assert_eq!(stacked.stride, vec![3, 1]);
    }

    #[test]
    fn tt_indexing_matches_batch_first() {
        // Verify element-by-element that [batch][feature] indexing works
        let a = tensor_1d(&[10.0, 20.0, 30.0]);
        let b = tensor_1d(&[40.0, 50.0, 60.0]);
        let stacked = Tensor::init(Tt(vec![a, b])).unwrap();
        // batch=0, feat=0
        assert_eq!(stacked.data[0 * 3 + 0], 10.0);
        // batch=0, feat=2
        assert_eq!(stacked.data[0 * 3 + 2], 30.0);
        // batch=1, feat=0
        assert_eq!(stacked.data[1 * 3 + 0], 40.0);
        // batch=1, feat=2
        assert_eq!(stacked.data[1 * 3 + 2], 60.0);
    }
}
