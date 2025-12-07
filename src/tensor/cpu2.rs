use std::{borrow::Cow, collections::HashSet};

use arrow::compute::kernels::length::length;

use crate::tensor::Tensor;

pub struct CPUTensor<'a> {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub data: Cow<'a, Vec<f64>>,
}

impl<'a> CPUTensor<'a> {
    pub fn init(init: impl TensorInit) -> Option<Self> {
        let (shape, data) = init.make()?;
        let stride = stride_of(&shape);
        Some(Self {
            shape,
            stride,
            data: Cow::Owned(data),
        })
    }

    pub fn reshape(&'a self, shape: &[usize]) -> Option<CPUTensor<'a>> {
        reshape_impl(
            Cow::Borrowed(&self.shape),
            Cow::Borrowed(&self.stride),
            Cow::Borrowed(&self.data),
            shape,
        )
    }

    pub fn into_reshape(self, shape: &[usize]) -> Option<CPUTensor<'a>> {
        reshape_impl(
            Cow::Owned(self.shape),
            Cow::Owned(self.stride),
            self.data,
            shape,
        )
    }

    pub fn transpose(&'a self, axes: &[usize]) -> Option<CPUTensor<'a>> {
        transpose_impl(&self.shape, &self.stride, Cow::Borrowed(&self.data), axes)
    }

    pub fn into_transpose(self, axes: &[usize]) -> Option<CPUTensor<'a>> {
        transpose_impl(&self.shape, &self.stride, self.data, axes)
    }

    pub fn materialize<'o>(&self) -> CPUTensor<'o> {
        let mut new_data = Vec::with_capacity(length_of(&self.shape));
        unsafe {
            new_data.set_len(length_of(&self.shape));
        }
        let mut new = CPUTensor {
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

    pub fn cloned_view<'o>(&self) -> CPUTensor<'o> {
        CPUTensor {
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            data: Cow::Owned(self.data.clone().into_owned()),
        }
    }
}

pub fn stride_of(shape: &[usize]) -> Vec<usize> {
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
) -> Option<CPUTensor<'a>> {
    if length_of(&shape) != length_of(&to_shape) {
        return None;
    }
    if shape.is_empty() {
        // short circuit case for scalar so i dont have to deal
        // with it in general reshaping code
        // we can just return the same scalar since shape must be []
        // and stride is already []
        return Some(CPUTensor {
            shape: shape.into_owned(),
            stride: stride.into_owned(),
            data: with_data,
        });
    }
    // see if stride is monotonically increasing. if so we can just recompute the strides
    // since we havent done any views
    let mut last = 0;
    let mut monotonic = true;
    for s in stride.iter() {
        if *s < last {
            monotonic = false;
            break;
        }
        last = *s;
    }
    if monotonic {
        let mut new_stride = Vec::with_capacity(to_shape.len());
        let mut mult = 1;
        for s in to_shape.iter() {
            new_stride.push(mult);
            mult *= *s;
        }
        return Some(CPUTensor {
            shape: to_shape.to_vec(),
            stride: new_stride,
            data: with_data,
        });
    }

    // so our stride isnt monotonically increasing, we transposed at some point
    // therefore we have to recompute the stride such that it still aligns with
    // the view we created.
    struct Block {
        len: usize,
        stride: usize,
    }

    let mut blocks = vec![];
    let mut block_len = 1;
    let mut block_stride = 1;
    for i in 0..shape.len() {
        if i > 0 && stride[i] == stride[i - 1] * shape[i - 1] {
            block_len *= shape[i];
        } else {
            if i > 0 {
                blocks.push(Block {
                    len: block_len,
                    stride: block_stride,
                });
            }
            block_len = shape[i];
            block_stride = stride[i];
        }
    }
    blocks.push(Block {
        len: block_len,
        stride: block_stride,
    });

    let mut block_iter = blocks.into_iter();
    let mut cur_blk = block_iter.next().unwrap();
    let mut cur_len = 1;
    let mut accd_stride = cur_blk.stride;
    let mut new_stride = Vec::with_capacity(to_shape.len());
    for (i, s) in to_shape.iter().enumerate() {
        cur_len *= *s;
        new_stride.push(accd_stride);
        accd_stride = accd_stride.saturating_mul(*s);
        if cur_len == cur_blk.len {
            cur_len = 1;
            if let Some(blk) = block_iter.next() {
                cur_blk = blk;
                accd_stride = cur_blk.stride;
            } else if i + 1 < to_shape.len() {
                // no more blocks. in this case the remaining axes
                // must be size 1. if not we will reject it later on
                accd_stride = 0;
            }
        } else if cur_len > cur_blk.len {
            return None;
        }
    }
    // we didnt use all of the blocks OR we couldnt fit the last dimensions
    // into a block, reject.
    // todo: if this causes issues later on add a materialization path
    if cur_len != 1 || block_iter.next().is_some() {
        return None;
    }
    Some(CPUTensor {
        shape: to_shape.to_vec(),
        stride: new_stride,
        data: with_data,
    })
}

fn transpose_impl<'a>(
    shape: &[usize],
    stride: &[usize],
    with_data: Cow<'a, Vec<f64>>,
    axes: &[usize],
) -> Option<CPUTensor<'a>> {
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
    Some(CPUTensor {
        shape: new_shape,
        stride: new_stride,
        data: with_data,
    })
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
