use std::{borrow::Cow, collections::HashSet};

use arrow::compute::kernels::length::length;

#[derive(Clone)]
pub struct Tensor<'a> {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub data: Cow<'a, Vec<f64>>,
}

impl<'a> Tensor<'a> {
    pub fn init(init: impl TensorInit) -> Option<Self> {
        let (shape, data) = init.make()?;
        let stride = stride_of(&shape);
        Some(Self {
            shape,
            stride,
            data: Cow::Owned(data),
        })
    }

    pub fn reshape(&'a self, shape: &[usize]) -> Option<Tensor<'a>> {
        reshape_impl(
            Cow::Borrowed(&self.shape),
            Cow::Borrowed(&self.stride),
            Cow::Borrowed(&self.data),
            shape,
        )
    }

    pub fn into_reshape(self, shape: &[usize]) -> Option<Tensor<'a>> {
        reshape_impl(
            Cow::Owned(self.shape),
            Cow::Owned(self.stride),
            self.data,
            shape,
        )
    }

    pub fn transpose(&'a self, axes: &[usize]) -> Option<Tensor<'a>> {
        transpose_impl(&self.shape, &self.stride, Cow::Borrowed(&self.data), axes)
    }

    pub fn into_transpose(self, axes: &[usize]) -> Option<Tensor<'a>> {
        transpose_impl(&self.shape, &self.stride, self.data, axes)
    }

    pub fn materialize<'o>(&self) -> Tensor<'o> {
        let mut new_data = Vec::with_capacity(length_of(&self.shape));
        unsafe {
            new_data.set_len(length_of(&self.shape));
        }
        let mut new = Tensor {
            shape: self.shape.clone(),
            stride: stride_of(&self.shape),
            data: Cow::Owned(new_data),
        };
        let new_slice = new.data.to_mut().as_mut_slice();
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

    pub fn cloned_view<'o>(&self) -> Tensor<'o> {
        Tensor {
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            data: Cow::Owned(self.data.clone().into_owned()),
        }
    }
}

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

fn reshape_impl<'a>(
    shape: Cow<'a, Vec<usize>>,
    stride: Cow<'a, Vec<usize>>,
    with_data: Cow<'a, Vec<f64>>,
    to_shape: &[usize],
) -> Option<Tensor<'a>> {
    if length_of(&shape) != length_of(&to_shape) {
        return None;
    }
    if shape.is_empty() {
        // short circuit case for scalar so i dont have to deal
        // with it in general reshaping code
        // we can just return the same scalar since shape must be []
        // and stride is already []
        return Some(Tensor {
            shape: shape.into_owned(),
            stride: stride.into_owned(),
            data: with_data,
        });
    }

    let shape = shape.into_owned();
    let stride = stride.into_owned();
    if is_row_major_contiguous(&shape, &stride) {
        return Some(Tensor {
            shape: to_shape.to_vec(),
            stride: stride_of(to_shape),
            data: with_data,
        });
    }

    // Fall back to a row-major materialized view when stride isn't contiguous.
    let view = Tensor {
        shape,
        stride,
        data: with_data,
    };
    let mut materialized = view.materialize();
    materialized.shape = to_shape.to_vec();
    materialized.stride = stride_of(to_shape);
    Some(materialized)
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

fn transpose_impl<'a>(
    shape: &[usize],
    stride: &[usize],
    with_data: Cow<'a, Vec<f64>>,
    axes: &[usize],
) -> Option<Tensor<'a>> {
    if shape.len() != axes.len() || axes.iter().any(|i| *i >= shape.len()) {
        return None;
    }
    let axis_set = axes.iter().copied().collect::<HashSet<usize>>();
    if axis_set.len() != shape.len() || axis_set != (0..shape.len()).collect() {
        return None;
    }
    let old_shape = shape.to_vec();
    let old_stride = stride.to_vec();
    let mut new_shape = Vec::with_capacity(shape.len());
    unsafe {
        new_shape.set_len(shape.len());
    }
    let mut new_stride = Vec::with_capacity(shape.len());
    unsafe {
        new_stride.set_len(shape.len());
    }
    for (i, j) in axes.iter().enumerate() {
        new_shape[i] = old_shape[*j];
        new_stride[i] = old_stride[*j];
    }
    Some(Tensor {
        shape: new_shape,
        stride: new_stride,
        data: with_data,
    })
}

pub trait TensorInit {
    fn make(self) -> Option<(Vec<usize>, Vec<f64>)>;
}

pub struct Tt<'a>(pub Vec<Tensor<'a>>);

impl TensorInit for Tt<'_> {
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
            let data = self.0.iter().map(|t| *t.data.get(0).unwrap_or(&0.0)).collect();
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
    fn null(shape: Vec<usize>) -> Self {
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
