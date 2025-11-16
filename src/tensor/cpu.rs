use core::f64;
use std::{collections::HashSet, io::{self, Read, Write}, sync::{Arc, Mutex}};

use crate::tensor::{Autograd, DifferentiableTensor, Field, Fill, FillUninit, Tensor, TensorIO, TensorInit, TensorMut};


#[derive(Clone, PartialEq)]
pub struct CPUTensor {
    shape: Vec<usize>,
    stride: Vec<usize>,
    data: Vec<f64>,
}

struct Block {
    len: usize,
    stride: usize,
}

macro_rules! impl_arithmetic {
    ($self:expr, $op:tt, $other:expr) => {{
        if $self.shape == $other.shape && $self.stride == $other.stride {
            let mut new_data = Vec::with_capacity($self.data.len());
            unsafe { new_data.set_len($self.data.len()); }
            let new_slice = new_data.as_mut_slice();
            let lhs_slice = $self.data.as_slice();
            let rhs_slice = $other.data.as_slice();
            for i in 0..$self.data.len() {
                unsafe {
                    *new_slice.get_unchecked_mut(i) = *lhs_slice.get_unchecked(i) $op *rhs_slice.get_unchecked(i);
                }
            }
            return Some(Self {
                shape: $self.shape.clone(),
                stride: $self.stride.clone(),
                data: new_data,
            });
        }
        let k = $self.shape.len().max($other.shape.len());
        let mut new_shape = Vec::with_capacity(k);
        let mut lhs_strides = Vec::with_capacity(k);
        let mut rhs_strides = Vec::with_capacity(k);
        for i in 0..k {
            let lhs = $self.shape.get(i).cloned();
            let rhs = $other.shape.get(i).cloned();
            match (lhs, rhs) {
                (Some(l), Some(r)) if l == r => {
                    new_shape.push(l);
                    lhs_strides.push($self.stride[i]);
                    rhs_strides.push($other.stride[i]);
                },
                (Some(l), Some(r)) if l == 1 => {
                    new_shape.push(r);
                    lhs_strides.push(0);
                    rhs_strides.push($other.stride[i]);
                },
                (Some(l), Some(r)) if r == 1 => {
                    new_shape.push(l);
                    lhs_strides.push($self.stride[i]);
                    rhs_strides.push(0);
                },
                _ => return None
            }
        }

        let mut new = Self::tensor(Fill::null(new_shape)).unwrap();
        let mut new_point = vec![0; new.shape.len()];
        let mut new_ptr = new.data.as_mut_ptr();
        let mut lhs_ptr = $self.data.as_ptr();
        let mut rhs_ptr = $other.data.as_ptr();

        'iterate: loop {
            unsafe { *new_ptr = *lhs_ptr $op *rhs_ptr; }
            for i in 0..new_point.len() {
                let np_ref = unsafe { new_point.get_unchecked_mut(i) };
                let np = *np_ref;
                if np == unsafe { new.shape.get_unchecked(i) } - 1 {
                    new_ptr = unsafe { new_ptr.sub(*new.stride.get_unchecked(i) * np) };
                    lhs_ptr = unsafe { lhs_ptr.sub(*lhs_strides.get_unchecked(i) * np) };
                    rhs_ptr = unsafe { rhs_ptr.sub(*rhs_strides.get_unchecked(i) * np) };
                    *np_ref = 0;
                } else {
                    new_ptr = unsafe { new_ptr.add(*new.stride.get_unchecked(i)) };
                    lhs_ptr = unsafe { lhs_ptr.add(*lhs_strides.get_unchecked(i)) };
                    rhs_ptr = unsafe { rhs_ptr.add(*rhs_strides.get_unchecked(i)) };
                    *np_ref += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        Some(new)
    }};
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
            4 => {
                self.stride[0] * of[0]
                    + self.stride[1] * of[1]
                    + self.stride[2] * of[2]
                    + self.stride[3] * of[3]
            }
            5 => {
                self.stride[0] * of[0]
                    + self.stride[1] * of[1]
                    + self.stride[2] * of[2]
                    + self.stride[3] * of[3]
                    + self.stride[4] * of[4]
            }
            6 => {
                self.stride[0] * of[0]
                    + self.stride[1] * of[1]
                    + self.stride[2] * of[2]
                    + self.stride[3] * of[3]
                    + self.stride[4] * of[4]
                    + self.stride[5] * of[5]
            }
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

    fn blocks(&self) -> Vec<Block> {
        let mut blocks = vec![];
        let mut block_len = 1;
        let mut block_stride = 1;
        for i in 0..self.ndim() {
            if i > 0 && self.stride[i] == self.stride[i - 1] * self.shape[i - 1] {
                block_len *= self.shape[i];
            } else {
                if i > 0 {
                    blocks.push(Block {
                        len: block_len,
                        stride: block_stride,
                    });
                }
                block_len = self.shape[i];
                block_stride = self.stride[i];
            }
        }
        blocks.push(Block {
            len: block_len,
            stride: block_stride,
        });
        blocks
    }

    fn contiguous_within(&self, within: std::ops::Range<usize>) -> bool {
        if self.ndim() == 0 {
            return true;
        }
        if within.end - within.start <= 1 {
            // single dimension is trivially contiguous
            return true;
        }
        // want to avoid checking dimensions outside the within range
        for i in within.start + 1..within.end {
            if self.stride[i] != self.stride[i - 1] * self.shape[i - 1] {
                return false;
            }
        }
        true
    }

    fn view(&self, shape: &[usize]) -> Option<Vec<usize>> {
        let blocks = self.blocks();
        let mut block_iter = blocks.into_iter();
        let mut cur_blk = block_iter.next().unwrap();
        let mut cur_len = 1;
        let mut accd_stride = cur_blk.stride;
        let mut new_stride = Vec::with_capacity(shape.len());
        for (i, s) in shape.iter().enumerate() {
            cur_len *= *s;
            new_stride.push(accd_stride);
            accd_stride = accd_stride.saturating_mul(*s);
            if cur_len == cur_blk.len {
                cur_len = 1;
                if let Some(blk) = block_iter.next() {
                    cur_blk = blk;
                    accd_stride = cur_blk.stride;
                } else if i + 1 < shape.len() {
                    // no more blocks. in this case the remaining axes
                    // must be size 1. if not we will reject it later on
                    accd_stride = 0;
                }
            } else if cur_len > cur_blk.len {
                return None;
            }
        }
        // we didnt use all of the blocks OR we couldnt fit the last dimensions
        // into a block, reject
        if cur_len != 1 || block_iter.next().is_some() {
            return None;
        }
        Some(new_stride)
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

        Some(Self {
            data,
            shape,
            stride,
        })
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
        impl_arithmetic!(self, +, other)
    }

    fn sub(&self, other: &Self) -> Option<Self> {
        impl_arithmetic!(self, -, other)
    }

    fn mul(&self, other: &Self) -> Option<Self> {
        impl_arithmetic!(self, *, other)
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
        if self.contiguous_within((self.shape.len() - depth)..(self.shape.len())) && other.contiguous_within(0..depth) {
            // fast case: dense axk kxb matmul
            // TODO: implement block intersections so i can apply this method to arbitrary 
            // non-contiguous tensors
            let k = contraction_shape.iter().product::<usize>();
            let a = lhs_survivors.iter().product::<usize>();
            let b = rhs_survivors.iter().product::<usize>();
            let lhs_2strides = self.view(&[a, k])?;
            let rhs_2strides = other.view(&[k, b])?;
            let mut new = Self::tensor(unsafe { FillUninit::new(vec![a, b]) }).unwrap();

            // explicit slice definitions to signal to the compiler about aliasing
            let lhs_data = self.data.as_slice();
            let mut lhs_idx = 0;
            let rhs_data = other.data.as_slice();
            let mut rhs_idx = 0;
            let new_data_mut = new.data.as_mut_slice();
            let mut new_idx = 0;
            for _ in 0..a {
                for _ in 0..b {
                    let mut sum = 0.0;
                    for _ in 0..k {
                        unsafe {
                            sum += *lhs_data.get_unchecked(lhs_idx) * *rhs_data.get_unchecked(rhs_idx);
                        }
                        lhs_idx += lhs_2strides[1];
                        rhs_idx += rhs_2strides[0];
                    }
                    unsafe {
                        *new_data_mut.get_unchecked_mut(new_idx) = sum;
                    }
                    // rewind indices
                    lhs_idx -= lhs_2strides[1] * k;
                    rhs_idx -= rhs_2strides[0] * k;

                    new_idx += new.stride[1];
                    rhs_idx += rhs_2strides[1];
                }
                rhs_idx = 0;
                lhs_idx += lhs_2strides[0];
                new_idx -= new.stride[1] * b;
                new_idx += new.stride[0];
            }
            // reshape back to original shape
            return Some(new.reshape(&new_shape).unwrap());
        }

        let mut new = Self::tensor(unsafe { FillUninit::new(new_shape) }).unwrap();
        let mut new_point = vec![0; new.shape.len()];
        let mut new_ptr = new.data.as_mut_ptr();
        let mut lhs_ptr = self.data.as_ptr();
        let mut rhs_ptr = other.data.as_ptr();

        let contraction_magnitude: usize = contraction_shape.iter().product();
        let lhs_contraction_stride = &self.stride[self.ndim() - depth..];
        let mut lhs_contract_offsets: Vec<usize> = Vec::with_capacity(contraction_magnitude);
        let rhs_contraction_stride = &other.stride[..depth];
        let mut rhs_contract_offsets: Vec<usize> = Vec::with_capacity(contraction_magnitude);

        let mut contraction_point = vec![0; contraction_shape.len()];
        let mut lhs_contract_idx = 0;
        let mut rhs_contract_idx = 0;
        'precompute: loop {
            lhs_contract_offsets.push(lhs_contract_idx);
            rhs_contract_offsets.push(rhs_contract_idx);
            for i in 0..contraction_point.len() {
                let con_ref = unsafe { contraction_point.get_unchecked_mut(i) };
                let con = *con_ref;
                let csh = unsafe { *contraction_shape.get_unchecked(i) };
                let lst = unsafe { *lhs_contraction_stride.get_unchecked(i) };
                let rst = unsafe { *rhs_contraction_stride.get_unchecked(i) };
                if con == csh - 1 {
                    lhs_contract_idx -= lst * con;
                    rhs_contract_idx -= rst * con;
                    *con_ref = 0;
                } else {
                    lhs_contract_idx += lst;
                    rhs_contract_idx += rst;
                    *con_ref += 1;
                    continue 'precompute;
                }
            }
            break;
        }
        'iterate: loop {
            let mut sum = 0.0;
            let mut i = 0;
            let mut lhs_offset_ptr = lhs_contract_offsets.as_ptr();
            let mut rhs_offset_ptr = rhs_contract_offsets.as_ptr();

            while i < contraction_magnitude {
                sum += unsafe { *lhs_ptr.add(*lhs_offset_ptr) * *rhs_ptr.add(*rhs_offset_ptr) };
                i += 1;
                lhs_offset_ptr = unsafe { lhs_offset_ptr.add(1) };
                rhs_offset_ptr = unsafe { rhs_offset_ptr.add(1) };
            }
            unsafe {
                *new_ptr = sum;
            }

            for i in 0..new.ndim() {
                let npt_ref = unsafe { new_point.get_unchecked_mut(i) };
                let np = *npt_ref;
                let nsh = unsafe { *new.shape.get_unchecked(i) };
                let nst = unsafe { *new.stride.get_unchecked(i) };
                if np == nsh - 1 {
                    *npt_ref = 0;
                    new_ptr = unsafe { new_ptr.sub(nst * np) };
                    if i < lhs_survivor_len {
                        let lst = unsafe { *self.stride.get_unchecked(i) };
                        lhs_ptr = unsafe { lhs_ptr.sub(lst * np) };
                    }
                    if i >= lhs_survivor_len && rhs_survivor_len > 0 {
                        let rhs_axis = depth + (i - lhs_survivor_len);
                        let rst = unsafe { *other.stride.get_unchecked(rhs_axis) };
                        rhs_ptr = unsafe { rhs_ptr.sub(rst * np) };
                    }
                } else {
                    *npt_ref += 1;
                    new_ptr = unsafe { new_ptr.add(nst) };
                    if i < lhs_survivor_len {
                        let lst = unsafe { *self.stride.get_unchecked(i) };
                        lhs_ptr = unsafe { lhs_ptr.add(lst) }
                    }
                    if i >= lhs_survivor_len {
                        let rhs_axis = depth + (i - lhs_survivor_len);
                        let rst = unsafe { *other.stride.get_unchecked(rhs_axis) };
                        rhs_ptr = unsafe { rhs_ptr.add(rst) }
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
        let mut new_idx = 0;
        let mut old_idx = 0;
        'iterate: loop {
            let mut sum = 0.0;
            let mut sum_idx = old_idx;
            let end_idx = sum_idx + self.stride[0] * self.shape[0];
            while sum_idx < end_idx {
                sum += self.data[sum_idx];
                sum_idx += self.stride[0];
            }
            new.data[new_idx] = sum;
            for i in 0..new.ndim() {
                if new_point[i] == new.shape[i] - 1 {
                    new_idx -= new.stride[i] * new_point[i];
                    old_idx += self.stride[i + 1] * new_point[i];
                    new_point[i] = 0;
                } else {
                    new_idx += new.stride[i];
                    old_idx += self.stride[i + 1];
                    new_point[i] += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        new
    }

    fn ln(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.ln());
        self
    }

    fn exp(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.exp());
        self
    }

    fn tanh(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.tanh());
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

    fn cols_at(&self, indices: &Self) -> Option<Self> {
        if indices.ndim() != 1 {
            return None;
        }
        let mut new_shape = self.shape.clone();
        new_shape[1] = indices.shape[0];
        let mut new = Self::tensor(Fill { shape: new_shape, with: 0.0 }).unwrap();
        let mut new_point = vec![0; new.ndim()];
        let mut new_idx = 0;
        while new_point[1] < indices.shape[0] {
            let idx = indices.data[new_point[1]] as usize;
            let mut old_idx = idx * self.stride[1];
            'iterate: loop {
                new.data[new_idx] = self.data[old_idx];
                for i in 0..new.ndim() {
                    if i == 1 {
                        continue;
                    }
                    if new_point[i] == new.shape[i] - 1 {
                        if i == 1 {
                            break 'iterate;
                        }
                        new_idx -= new.stride[i] * new_point[i];
                        old_idx += self.stride[i] * new_point[i];
                        new_point[i] = 0;
                    } else {
                        new_idx += new.stride[i];
                        old_idx += self.stride[i];
                        new_point[i] += 1;
                        continue 'iterate;
                    }
                }
                break;
            }
            new_idx += new.stride[1];
            new_point[1] += 1;
        }
        Some(new)
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
        let mut field_point = [0, 0];
        let mut location_point = [-(field.padding as i64), -(field.padding as i64)];
        let mut new_idx = 0;
        let mut rel_idx = 0i64;
        let mut new_point = vec![0; new.shape.len()];
        let mut old_idx = 0usize;
        'iterate: loop {
            let x = location_point[0] + field_point[0] as i64;
            let y = location_point[1] + field_point[1] as i64;
            if x >= 0 && y >= 0 && x < self.shape[0] as i64 && y < self.shape[1] as i64 {
                *unsafe { new.data.get_unchecked_mut(new_idx) } =
                    unsafe { *self.data.get_unchecked(rel_idx as usize + old_idx) };
            }

            for i in 0..self.ndim() {
                if new_point[i] == new.shape[i] - 1 {
                    match i {
                        0 => {
                            rel_idx -= (field_point[0] * self.stride[0]
                                + field_point[1] * self.stride[1])
                                as i64;
                            field_point[0] = 0;
                            field_point[1] = 0;
                        }
                        1 => {
                            rel_idx -= (location_point[0] + field.padding as i64)
                                * self.stride[0] as i64
                                + (location_point[1] + field.padding as i64)
                                    * self.stride[1] as i64;
                            location_point[0] = -(field.padding as i64);
                            location_point[1] = -(field.padding as i64);
                        }
                        _ => {}
                    }
                    if i >= 2 {
                        old_idx -= self.stride[i] * new_point[i];
                    }
                    new_idx -= new.stride[i] * new_point[i];
                    new_point[i] = 0;
                } else {
                    match i {
                        0 => {
                            if field_point[0] == field.size - 1 {
                                rel_idx -= (field_point[0] * self.stride[0]) as i64;
                                rel_idx += self.stride[1] as i64;
                                field_point[0] = 0;
                                field_point[1] += 1;
                            } else {
                                rel_idx += self.stride[0] as i64;
                                field_point[0] += 1;
                            }
                        }
                        1 => {
                            if location_point[0] == ((locations - 1) * field.stride) as i64 {
                                rel_idx -= location_point[0] * self.stride[0] as i64;
                                rel_idx += (field.stride * self.stride[1]) as i64;
                                location_point[0] = -(field.padding as i64);
                                location_point[1] += field.stride as i64;
                            } else {
                                rel_idx += (field.stride * self.stride[0]) as i64;
                                location_point[0] += field.stride as i64;
                            }
                        }
                        _ => {}
                    }
                    if i >= 2 {
                        old_idx += self.stride[i]
                    }
                    new_idx += new.stride[i];
                    new_point[i] += 1;
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
            with: f64::NEG_INFINITY,
        })
        .unwrap();
        let mut new_point = vec![0; new_shape.len()];
        let mut new_idx = 0;
        let mut old_idx = 0;
        'iterate: loop {
            let mut max = f64::NEG_INFINITY;
            let mut max_idx = old_idx;
            let end_idx = old_idx + self.shape[0] * self.stride[0];
            while max_idx < end_idx {
                if self.data[max_idx] > max {
                    max = self.data[max_idx];
                }
                max_idx += self.stride[0];
            }
            new.data[new_idx] = max;
            for i in 0..new.ndim() {
                if new_point[i] == new.shape[i] - 1 {
                    new_idx -= new.stride[i] * new_point[i];
                    old_idx -= self.stride[i + 1] * new_point[i];
                    new_point[i] = 0;
                } else {
                    new_idx += new.stride[i];
                    old_idx += self.stride[i + 1];
                    new_point[i] += 1;
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
        if self.ndim() == 0 {
            // short circuit case for scalar so i dont have to deal
            // with it in general reshaping code
            // we can just return the same scalar since shape must be []
            // and stride is already []
            return Some(self);
        }
        // see if stride is monotonically increasing. if so we can just recompute the strides
        // since we havent done any views
        let mut last = 0;
        let mut monotonic = true;
        for s in &self.stride {
            if *s < last {
                monotonic = false;
                break;
            }
            last = *s;
        }
        if monotonic {
            self.shape = shape.to_vec();
            self.stride.clear();
            let mut mult = 1;
            for s in shape {
                self.stride.push(mult);
                mult *= *s;
            }
            return Some(self);
        }

        // so our stride isnt monotonically increasing, we transposed at some point
        // therefore we have to recompute the stride such that it still aligns with
        // the view we created.
        let new_stride = self.view(shape)?;
        self.shape = shape.to_vec();
        self.stride = new_stride;

        Some(self)
    }

    fn transpose(mut self, axes: &[usize]) -> Option<Self> {
        if self.shape.len() != axes.len() || axes.iter().any(|i| *i >= self.shape.len()) {
            return None;
        }
        let axis_set = axes.iter().copied().collect::<HashSet<usize>>();
        if axis_set.len() != self.ndim() || axis_set != (0..self.ndim()).collect() {
            return None;
        }
        let old_shape = self.shape.clone();
        let old_stride = self.stride.clone();
        for (i, j) in axes.iter().enumerate() {
            self.shape[i] = old_shape[*j];
            self.stride[i] = old_stride[*j];
        }
        Some(self)
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

    fn softmax(mut self) -> Option<Self> {        
        // technically `t` in activation can be more than just an activation vector but actually a batch of activation
        // vectors, so we need to compute numerical stability for each activation vector in the batch.
        // to vaguely generalize this we will assume all non-last dimensions are the activations and then the last dimension is the batch.
        let batch_size = self.shape.last().unwrap();
        let batch_stride = self.stride.last().unwrap();
        let mut point = vec![0; self.ndim() - 1];
        for i in 0..*batch_size { 
            let mut idx = i * batch_stride;
            let mut max = f64::MIN;
            'iterate: loop {
                if self.data[idx] > max {
                    max = self.data[idx];
                }
                for i in 0..self.ndim() - 1 {
                    if point[i] == self.shape[i] - 1 {
                        idx -= self.stride[i] * point[i];
                        point[i] = 0;
                    } else {
                        idx += self.stride[i];
                        point[i] += 1;
                        continue 'iterate;
                    }
                }
                break 'iterate;
            }
            idx = i * batch_stride;
            point.fill(0);
            let mut norm = 0.0;
            'iterate: loop {
                self.data[idx] = (self.data[idx] - max).exp();
                norm += self.data[idx];
                for i in 0..self.ndim() - 1 {
                    if point[i] == self.shape[i] - 1 {
                        idx -= self.stride[i] * point[i];
                        point[i] = 0;
                    } else {
                        idx += self.stride[i];
                        point[i] += 1;
                        continue 'iterate;
                    }
                }
                break 'iterate;
            }
            idx = i * batch_stride;
            point.fill(0);
            'iterate: loop {
                self.data[idx] /= norm;
                for i in 0..self.ndim() - 1 {
                    if point[i] == self.shape[i] - 1 {
                        idx -= self.stride[i] * point[i];
                        point[i] = 0;
                    } else {
                        idx += self.stride[i];
                        point[i] += 1;
                        continue 'iterate;
                    }
                }
                break 'iterate;
            }
        }
        Some(self)
    }
}

impl DifferentiableTensor for CPUTensor {
    type Autograd = CPUAutograd;
    fn autograd(self) -> Self::Autograd {
        CPUAutograd(AutogradNode::new(AutogradNodeData {
            tensor: self,
            edge: AutogradEdge {
                grad: Mutex::new(None),
                op: None,
            }
        }))
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
        let size = Self::len(shape.as_slice());
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

#[derive(Clone)]
pub struct CPUAutograd(AutogradNode);

impl Autograd<CPUTensor> for CPUAutograd {
    fn backward(self) {    
        let identity = 
            CPUTensor::tensor(Fill {
                shape: self.0.tensor.shape().to_vec(),
                with: 1.0,
            })
            .unwrap();
        self.0.edge.backward(identity);
    }

    fn into_grad(self) -> Option<CPUTensor> {
        Arc::into_inner(self.0).and_then(|c| c.edge.grad.into_inner().ok().flatten())
    }
}

impl Tensor for CPUAutograd {
    fn scalar(c: impl Into<f64>) -> Self {
        Self(AutogradNode::new(AutogradNodeData {
            tensor: CPUTensor::scalar(c),
            edge: AutogradEdge {
                grad: Mutex::new(None),
                op: None,
            }
        }))
    }

    fn vector(v: impl Into<Vec<f64>>) -> Option<Self> {
        Some(Self(AutogradNode::new(AutogradNodeData {
            tensor: CPUTensor::vector(v)?,
            edge: AutogradEdge {
                grad: Mutex::new(None),
                op: None,
            }
        })))
    }

    fn tensor(tv: impl TensorInit) -> Option<Self> {
        Some(Self(AutogradNode::new(AutogradNodeData {
            tensor: CPUTensor::tensor(tv)?,
            edge: AutogradEdge {
                grad: Mutex::new(None),
                op: None,
            }
        })))
    }

    fn ndim(&self) -> usize {
        self.0.tensor.ndim()
    }

    fn shape(&self) -> &[usize] {
        &self.0.tensor.shape()
    }

    fn get(&self, point: &[usize]) -> Option<&f64> {
        self.0.tensor.get(point)
    }

    fn iter(&self) -> impl Iterator<Item = &f64> {
        self.0.tensor.iter()
    }

    fn add(&self, other: &Self) -> Option<Self> {
        Operation::Add {
            lhs: self.0.clone(),
            rhs: other.0.clone(),
        }
        .forward()
    }

    fn sub(&self, other: &Self) -> Option<Self> {
        Operation::Sub {
            lhs: self.0.clone(),
            rhs: other.0.clone(),
        }
        .forward()
    }

    fn mul(&self, other: &Self) -> Option<Self> {
        Operation::Mul {
            lhs: self.0.clone(),
            rhs: other.0.clone(),
        }
        .forward()
    }

    fn dot(&self, other: &Self, depth: usize) -> Option<Self> {
        Operation::Dot {
            lhs: self.0.clone(),
            rhs: other.0.clone(),
            depth,
        }
        .forward()
    }

    fn sum(&self) -> Self {
        Operation::Sum { t: self.0.clone() }.forward().unwrap()
    }

    fn pow(self, i: i32) -> Self {
        Operation::Pow { t: self.0, i }.forward().unwrap()
    }

    fn ln(self) -> Self {
        Operation::Ln { t: self.0 }.forward().unwrap()
    }

    fn exp(self) -> Self {
        Operation::Exp { t: self.0 }.forward().unwrap()
    }

    fn tanh(self) -> Self {
        Operation::Exp { t: self.0.clone() }.forward().unwrap()
    }
    
    fn neg(self) -> Self {
        Operation::Neg { t: self.0 }.forward().unwrap()
    }

    fn max(self, y: f64) -> Self {
        Operation::Max { t: self.0, u: y }.forward().unwrap()
    }

    fn cols_at(&self, indices: &Self) -> Option<Self> {
        Operation::ColsAt {
            t: self.0.clone(),
            indices: indices.0.clone(),
        }
        .forward()
    }

    fn colify(&self, field: Field) -> Option<Self> {
        Operation::Colify {
            t: self.0.clone(),
            field,
        }
        .forward()
    }

    fn colmax(&self) -> Option<Self> {
        Operation::Colmax { t: self.0.clone() }.forward()
    }

    fn reshape(self, shape: &[usize]) -> Option<Self> {
        Operation::Reshape {
            t: self.0.clone(),
            shape: shape.to_vec(),
        }
        .forward()
    }

    fn transpose(self, axes: &[usize]) -> Option<Self> {
        Operation::Transpose {
            t: self.0,
            axes: axes.to_vec(),
        }
        .forward()
    }

    fn at_argmax(&self, of: &Self) -> Option<Self> {
        Operation::AtArgmax {
            t: self.0.clone(),
            of: of.0.clone(),
        }
        .forward()
    }

    fn softmax(self) -> Option<Self> {
        Operation::Softmax { t: self.0.clone() }.forward()
    }
}

type AutogradNode = Arc<AutogradNodeData>;

struct AutogradNodeData {
    tensor: CPUTensor,
    edge: AutogradEdge,
}

struct AutogradEdge {
    grad: Mutex<Option<CPUTensor>>,
    op: Option<Operation>
}

fn unravel(node: AutogradNode) -> UnraveledEdge {
    if Arc::strong_count(&node) == 1 {
        let owned=  Arc::try_unwrap(node).ok().unwrap();
        UnraveledEdge::Edge(owned.edge)
    } else {
        UnraveledEdge::Node(node.clone())
    }
}

fn unravel_tensor(node: AutogradNode) -> (CPUTensor, UnraveledEdge) {
    if Arc::strong_count(&node) == 1 {
        let owned=  Arc::try_unwrap(node).ok().unwrap();
        (owned.tensor, UnraveledEdge::Edge(owned.edge))
    } else {
        (node.tensor.clone(), UnraveledEdge::Node(node.clone()))
    }
}
    
enum UnraveledEdge {
    Edge(AutogradEdge),
    Node(AutogradNode),
}

impl UnraveledEdge {
    pub fn backward(self, grad: CPUTensor) {
        match self {
            Self::Edge(edge) => edge.take_backward(grad),
            Self::Node(node) => node.edge.backward(grad),
        }
    }
}

impl AutogradEdge {
    pub fn backward(&self, grad: CPUTensor) {
        if let Some(op) = &self.op {
            op.backward(grad);
            return;
        }
        let mut slf_grad = self.grad.lock().expect("mutex poisoned, should not happen");
        let grad_to_modify = slf_grad.take();
        slf_grad.replace(
            grad_to_modify
                .map(|ng| ng.add(&grad).unwrap())
                .unwrap_or(grad),
        );
    }

    pub fn take_backward(self, grad: CPUTensor) {
        if let Some(op) = self.op {
            op.take_backward(grad);
        }
        // in this case we have exclusive ownership of the edge, which also implies its not a node
        // that we want to save a grad to either, so we just forward the grad upwards.
    }
}

enum Operation {
    Add {
        lhs: AutogradNode,
        rhs: AutogradNode,
    },
    Sub {
        lhs: AutogradNode,
        rhs: AutogradNode,
    },
    Mul {
        lhs: AutogradNode,
        rhs: AutogradNode,
    },
    Dot {
        lhs: AutogradNode,
        rhs: AutogradNode,
        depth: usize,
    },
    Sum {
        t: AutogradNode,
    },
    Ln {
        t: AutogradNode,
    },
    Exp {
        t: AutogradNode,
    },
    Tanh {
        t: AutogradNode,
    },
    Pow {
        t: AutogradNode,
        i: i32,
    },
    Neg {
        t: AutogradNode,
    },
    Max {
        t: AutogradNode,
        u: f64,
    },
    ColsAt {
        t: AutogradNode,
        indices: AutogradNode,
    },
    Colify {
        t: AutogradNode,
        field: Field,
    },
    Colmax {
        t: AutogradNode,
    },
    Reshape {
        t: AutogradNode,
        shape: Vec<usize>,
    },
    Transpose {
        t: AutogradNode,
        axes: Vec<usize>,
    },
    AtArgmax {
        t: AutogradNode,
        of: AutogradNode,
    },
    Softmax {
        t: AutogradNode,
    }
}

impl Operation {
    #[inline(always)]
    fn arithmetic_backward(
        lhs: AutogradNode,
        rhs: AutogradNode,
        grad: CPUTensor,
        op: impl Fn(&f64, &f64) -> (f64, f64),
    ) {
        // partially assisted via codex
        let k = lhs.tensor.ndim().max(rhs.tensor.ndim());
        let mut lhs_strides = Vec::with_capacity(k);
        let mut rhs_strides = Vec::with_capacity(k);

        for axis in 0..k {
            let lhs_dim = lhs.tensor.shape.get(axis).copied().unwrap_or(1);
            let rhs_dim = rhs.tensor.shape.get(axis).copied().unwrap_or(1);

            if lhs_dim == rhs_dim {
                lhs_strides.push(lhs.tensor.stride[axis]);
                rhs_strides.push(rhs.tensor.stride[axis]);
            } else if lhs_dim == 1 {
                lhs_strides.push(0);
                rhs_strides.push(rhs.tensor.stride[axis]);
            } else if rhs_dim == 1 {
                lhs_strides.push(lhs.tensor.stride[axis]);
                rhs_strides.push(0);
            } else {
                unreachable!();
            }
        }

        let mut lhs_grad = CPUTensor { shape: lhs.tensor.shape.clone(), stride: lhs.tensor.stride.clone(), data: vec![0.0; lhs.tensor.data.len()] };
        let mut rhs_grad = CPUTensor { shape: rhs.tensor.shape.clone(), stride: rhs.tensor.stride.clone(), data: vec![0.0; rhs.tensor.data.len()] };

        let mut grad_point = vec![0; grad.ndim()];
        let mut grad_ptr = grad.data.as_ptr();
        let mut lhs_ptr = lhs.tensor.data.as_ptr();
        let mut lhs_grad_ptr = lhs_grad.data.as_mut_ptr();
        let mut rhs_ptr = rhs.tensor.data.as_ptr();
        let mut rhs_grad_ptr = rhs_grad.data.as_mut_ptr();

        'iterate: loop {
            let lhs_val = unsafe { *lhs_ptr };
            let rhs_val = unsafe { *rhs_ptr };
            let (lhs_scale, rhs_scale) = op(&lhs_val, &rhs_val);
            let upstream = unsafe { *grad_ptr };

            unsafe { *lhs_grad_ptr += lhs_scale * upstream; }
            unsafe { *rhs_grad_ptr += rhs_scale * upstream; }

            for axis in 0..grad.ndim() {
                if grad_point[axis] == grad.shape()[axis] - 1 {
                    grad_ptr = unsafe { grad_ptr.sub(grad.stride[axis] * grad_point[axis]) };
                    lhs_ptr = unsafe { lhs_ptr.sub(lhs_strides[axis] * grad_point[axis]) };
                    lhs_grad_ptr = unsafe { lhs_grad_ptr.sub(lhs_strides[axis] * grad_point[axis]) };
                    rhs_ptr = unsafe { rhs_ptr.sub(rhs_strides[axis] * grad_point[axis]) };
                    rhs_grad_ptr = unsafe { rhs_grad_ptr.sub(rhs_strides[axis] * grad_point[axis]) };
                    grad_point[axis] = 0;
                } else {
                    grad_point[axis] += 1;
                    grad_ptr = unsafe { grad_ptr.add(grad.stride[axis]) };
                    lhs_ptr = unsafe { lhs_ptr.add(lhs_strides[axis]) };
                    lhs_grad_ptr = unsafe { lhs_grad_ptr.add(lhs_strides[axis]) };
                    rhs_ptr = unsafe { rhs_ptr.add(rhs_strides[axis]) };
                    rhs_grad_ptr = unsafe { rhs_grad_ptr.add(rhs_strides[axis]) };
                    continue 'iterate;
                }
            }

            break;
        }

        rhs.edge.backward(rhs_grad);
        lhs.edge.backward(lhs_grad);
    }

    fn add_backward(lhs: AutogradNode, rhs: AutogradNode, grad: CPUTensor) {
        Self::arithmetic_backward(lhs, rhs, grad, |_, _| (1.0, 1.0));
    }

    fn sub_backward(lhs: AutogradNode, rhs: AutogradNode, grad: CPUTensor) {
        Self::arithmetic_backward(lhs, rhs, grad, |_, _| (1.0, -1.0));
    }

    fn mul_backward(lhs: AutogradNode, rhs: AutogradNode, grad: CPUTensor) {
        Self::arithmetic_backward(lhs, rhs, grad, |lhs, rhs| (*rhs, *lhs));
    }

    fn dot_backward(lhs: AutogradNode, rhs: AutogradNode, depth: usize, grad: CPUTensor) {
        let (lhs_tensor, lhs_edge) = unravel_tensor(lhs);
        let (rhs_tensor, rhs_edge) = unravel_tensor(rhs);
        let lhs_shift = rhs_tensor.ndim() - depth;
        let mut rhs_axes: Vec<usize> = (0..rhs_tensor.ndim()).collect();
        rhs_axes.rotate_right(lhs_shift);
        lhs_edge.backward(
            grad.dot(&rhs_tensor.transpose(&rhs_axes).unwrap(), lhs_shift)
                .unwrap(),
        );

        let rhs_shift = lhs_tensor.ndim() - depth;
        let mut lhs_axes: Vec<usize> = (0..lhs_tensor.ndim()).collect();
        lhs_axes.rotate_left(rhs_shift);
        rhs_edge.backward(
            lhs_tensor
                .transpose(&lhs_axes)
                .unwrap()
                .dot(&grad, rhs_shift)
                .unwrap(),
        );
    }

    fn sum_backward(t: AutogradNode, grad: CPUTensor) {
        let mut t_grad = CPUTensor::tensor(Fill {
            shape: t.tensor.shape().to_vec(),
            with: 0.0,
        })
        .unwrap();
        let mut t_grad_point = vec![0; t_grad.shape().len()];
        let mut grad_point = vec![0; grad.shape().len()];
        'iterate: loop {
            t_grad_point[1..].copy_from_slice(&grad_point);
            for i in 0..t.tensor.shape()[0] {
                t_grad_point[0] = i;
                *t_grad.get_mut(&t_grad_point).unwrap() = *grad.get(&grad_point).unwrap();
            }
            for (p, s) in grad_point.iter_mut().zip(grad.shape().iter()) {
                if *p == *s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        t.edge.backward(t_grad);
    }

    fn ln_backward(t: AutogradNode, grad: CPUTensor) {
        let (t_tensor, t_edge) = unravel_tensor(t);
        t_edge.backward(t_tensor.pow(-1).mul(&grad).unwrap());
    }

    fn exp_backward(t: AutogradNode, grad: CPUTensor) {
        let (t_tensor, t_edge) = unravel_tensor(t);
        t_edge.backward(t_tensor.exp().mul(&grad).unwrap());
    }

    fn tanh_backward(t: AutogradNode, grad: CPUTensor) {
        let (mut t_tensor, t_edge) = unravel_tensor(t);
        t_tensor.data.iter_mut()
            .for_each(|x| {
                *x = x.asinh().powi(2);
            });
        t_edge.backward(t_tensor.mul(&grad).unwrap());
    }

    fn pow_backward(t: AutogradNode, i: i32, grad: CPUTensor) {
        let (t_tensor, t_edge) = unravel_tensor(t);
        t_edge.backward(
            t_tensor
                .pow(i - 1)
                .mul(&Tensor::scalar(i as f64))
                .unwrap()
                .mul(&grad)
                .unwrap(),
        );
    }

    fn neg_backward(t: AutogradNode, grad: CPUTensor) {
        let t_edge = unravel(t);
        t_edge.backward(grad.neg());
    }

    fn max_backward(t: AutogradNode, u: f64, grad: CPUTensor) {
        let (mut t_tensor, t_edge) = unravel_tensor(t);
        t_tensor.iter_mut()
            .for_each(|x| *x = if *x >= u { 1.0 } else { 0.0 });
        t_edge.backward(t_tensor.mul(&grad).unwrap());
    }

    fn cols_at_backward(t: AutogradNode, indices: AutogradNode, grad: CPUTensor) {
        let mut t_grad = CPUTensor { shape: t.tensor.shape.clone(), stride: t.tensor.stride.clone(), data: vec![0.0; t.tensor.data.len()] };
        let mut grad_point = vec![0; grad.ndim()];
        let mut grad_idx = 0;
        while grad_point[1] < indices.tensor.shape[0] {
            let idx = indices.tensor.data[grad_point[1]] as usize;
            let mut t_grad_idx = idx * t.tensor.stride[1];
            'iterate: loop {
                t_grad.data[t_grad_idx] += grad.data[grad_idx];
                for i in 0..grad.ndim() {
                    if i == 1 {
                        continue;
                    }
                    if grad_point[i] == grad.shape[i] - 1 {
                        grad_idx -= grad.stride[i] * grad_point[i];
                        t_grad_idx -= t.tensor.stride[i] * grad_point[i];
                        grad_point[i] = 0;
                    } else {
                        grad_idx += grad.stride[i];
                        t_grad_idx += t.tensor.stride[i];
                        grad_point[i] += 1;
                        continue 'iterate;
                    }
                }
                break;
            }
            grad_idx += grad.stride[1];
            grad_point[1] += 1;
        }
        t.edge.backward(t_grad);
    }

    fn colify_backward(t: AutogradNode, field: Field, grad: CPUTensor) {
        let mut t_grad = CPUTensor { shape: t.tensor.shape.clone(), stride: t.tensor.stride.clone(), data: vec![0.0; t.tensor.data.len()] };
        let locations = field
            .locations_on(*t.tensor.shape().first().unwrap())
            .unwrap();
        let new_shape = grad.shape();
        let mut new_point = vec![0; new_shape.len()];
        let mut field_point = [0, 0];
        let mut location_point = [-(field.padding as i64), -(field.padding as i64)];
        let mut new_idx = 0;
        let mut rel_idx = 0i64;
        let mut old_idx = 0usize;
        'iterate: loop {
            let x = location_point[0] + field_point[0] as i64;
            let y = location_point[1] + field_point[1] as i64;
            if x >= 0 && y >= 0 && x < t.tensor.shape[0] as i64 && y < t.tensor.shape[1] as i64 {
                t_grad.data[rel_idx as usize + old_idx] += grad.data[new_idx];
            }

            for i in 0..t.tensor.ndim() {
                if new_point[i] == grad.shape[i] - 1 {
                    match i {
                        0 => {
                            rel_idx -= (field_point[0] * t.tensor.stride[0]
                                + field_point[1] * t.tensor.stride[1])
                                as i64;
                            field_point[0] = 0;
                            field_point[1] = 0;
                        }
                        1 => {
                            rel_idx -= (location_point[0] + field.padding as i64)
                                * t.tensor.stride[0] as i64
                                + (location_point[1] + field.padding as i64)
                                    * t.tensor.stride[1] as i64;
                            location_point[0] = -(field.padding as i64);
                            location_point[1] = -(field.padding as i64);
                        }
                        _ => {}
                    }
                    if i >= 2 {
                        old_idx -= t.tensor.stride[i] * new_point[i];
                    }
                    new_idx -= grad.stride[i] * new_point[i];
                    new_point[i] = 0;
                } else {
                    match i {
                        0 => {
                            if field_point[0] == field.size - 1 {
                                rel_idx -= (field_point[0] * t.tensor.stride[0]) as i64;
                                rel_idx += t.tensor.stride[1] as i64;
                                field_point[0] = 0;
                                field_point[1] += 1;
                            } else {
                                rel_idx += t.tensor.stride[0] as i64;
                                field_point[0] += 1;
                            }
                        }
                        1 => {
                            if location_point[0] == ((locations - 1) * field.stride) as i64 {
                                rel_idx -= location_point[0] * t.tensor.stride[0] as i64;
                                rel_idx += (field.stride * t.tensor.stride[1]) as i64;
                                location_point[0] = -(field.padding as i64);
                                location_point[1] += field.stride as i64;
                            } else {
                                rel_idx += (field.stride * t.tensor.stride[0]) as i64;
                                location_point[0] += field.stride as i64;
                            }
                        }
                        _ => {}
                    }
                    if i >= 2 {
                        old_idx += t.tensor.stride[i]
                    }
                    new_idx += grad.stride[i];
                    new_point[i] += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        t.edge.backward(t_grad);
    }

    fn colmax_backward(t: AutogradNode, grad: CPUTensor) {
        let mut t_grad = CPUTensor { shape: t.tensor.shape.clone(), stride: t.tensor.stride.clone(), data: vec![0.0; t.tensor.data.len()] };
        let mut grad_point = vec![0; grad.ndim()];
        let mut grad_idx = 0;
        let mut t_idx = 0;
        'iterate: loop {
            let mut max_idx = 0;
            let mut max_value = f64::NEG_INFINITY;
            for i in 0..t.tensor.shape()[0] {
                let cur_value = t.tensor.data[t_idx + i * t.tensor.stride[0]];
                if cur_value > max_value {
                    max_idx = i;
                    max_value = cur_value;
                }
            }
            t_grad.data[t_idx + max_idx * t_grad.stride[0]] = grad.data[grad_idx];
            for i in 0..grad.ndim() {
                if grad_point[i] == grad.shape()[i] - 1 {
                    grad_idx -= grad.stride[i] * grad_point[i];
                    t_idx -= t.tensor.stride[i + 1] * grad_point[i];
                    grad_point[i] = 0;
                } else {
                    grad_idx += grad.stride[i];
                    t_idx += t.tensor.stride[i + 1];
                    grad_point[i] += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        t.edge.backward(t_grad);
    }

    fn reshape_backward(t: AutogradNode, grad: CPUTensor) {
        let reshaped = grad.reshape(&t.tensor.shape()).unwrap();
        unravel(t).backward(reshaped)
    }

    fn transpose_backward(t: AutogradNode, axes: &[usize], grad: CPUTensor) {
        let mut rev_axes = vec![0; axes.len()];
        for i in 0..axes.len() {
            rev_axes[axes[i]] = i;
        }
        let transposed = grad.transpose(&rev_axes).unwrap();
        unravel(t).backward(transposed);
    }

    fn at_argmax_backward(at: AutogradNode, of: &AutogradNode, grad: CPUTensor) {
        let mut t_grad = CPUTensor::tensor(Fill {
            shape: at.tensor.shape().to_vec(),
            with: 0.0,
        })
        .unwrap();
        let mut t_grad_point = vec![0; t_grad.shape().len()];
        let mut grad_point = vec![0; grad.shape().len()];
        'iterate: loop {
            let mut argmax = 0;
            let mut max = f64::NEG_INFINITY;
            t_grad_point[1..].copy_from_slice(&grad_point);
            for i in 0..at.tensor.shape()[0] {
                t_grad_point[0] = i;
                let x = *of.tensor.get(&t_grad_point).unwrap();
                if x > max {
                    argmax = i;
                    max = x;
                }
            }
            t_grad_point[0] = argmax;
            *t_grad.get_mut(&t_grad_point).unwrap() = *grad.get(&grad_point).unwrap();
            for (p, s) in grad_point.iter_mut().zip(grad.shape().iter()) {
                if *p == *s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        at.edge.backward(t_grad);
    }

    fn softmax_backwards(t: AutogradNode, grad: CPUTensor) {
        let (t_tensor, t_edge)= unravel_tensor(t);
        // technically `t` in activation can be more than just an activation vector but actually a batch of activation
        // vectors, so we need to compute numerical stability for each activation vector in the batch.
        // to vaguely generalize this we will assume all non-last dimensions are the activations and then the last dimension is the batch.
        let orig_shape = t_tensor.shape().to_vec();
        let batch = *t_tensor.shape().last().unwrap();
        let act_prod = t_tensor.data.len() / batch;
        let softmax = t_tensor.softmax().unwrap().reshape(&[act_prod, batch]).unwrap();
        let grad = grad.reshape(&[act_prod, batch]).unwrap();
        let s_mul_g = softmax.mul(&grad).unwrap().sum().reshape(&[1, batch]).unwrap();
        let g_mul_sum = grad.sub(&s_mul_g).unwrap();
        let upstream = g_mul_sum.mul(&softmax).unwrap().reshape(&orig_shape).unwrap();
        t_edge.backward(upstream);
    }

    fn forward(self) -> Option<CPUAutograd> {
        let tensor = match &self {
            Self::Add { lhs, rhs } => lhs.tensor.add(&rhs.tensor),
            Self::Sub { lhs, rhs } => lhs.tensor.sub(&rhs.tensor),
            Self::Mul { lhs, rhs } => lhs.tensor.mul(&rhs.tensor),
            Self::Dot { lhs, rhs, depth } => lhs.tensor.dot(&rhs.tensor, *depth),
            Self::Sum { t } => Some(t.tensor.sum()),
            Self::Ln { t } => Some(t.tensor.clone().ln()),
            Self::Exp { t } => Some(t.tensor.clone().exp()),
            Self::Tanh { t } => Some(t.tensor.clone().tanh()),
            Self::Pow { t, i } => Some(t.tensor.clone().pow(*i)),
            Self::Neg { t } => Some(t.tensor.clone().neg()),
            Self::Max { t, u } => Some(t.tensor.clone().max(*u)),
            Self::ColsAt { t, indices } => t.tensor.cols_at(&indices.tensor),
            Self::Colify { t, field } => t.tensor.colify(*field),
            Self::Colmax { t } => t.tensor.colmax(),
            Self::Reshape { t, shape } => t.tensor.clone().reshape(shape),
            Self::Transpose { t, axes } => t.tensor.clone().transpose(&axes),
            Self::AtArgmax { t, of } => t.tensor.at_argmax(&of.tensor),
            Self::Softmax { t } => t.tensor.clone().softmax()
        };
        tensor.map(|tensor| {
            CPUAutograd(Arc::new(AutogradNodeData {
                tensor,
                edge: AutogradEdge {
                    grad: Mutex::new(None),
                    op: Some(self),
                }
            }))
        })
    }
 
    fn backward(&self, grad: CPUTensor) {
        match self {
            Self::Add { lhs, rhs } => {
                Self::add_backward(lhs.clone(), rhs.clone(), grad);
            }
            Self::Sub { lhs, rhs } => {
                Self::sub_backward(lhs.clone(), rhs.clone(), grad);
            }
            Self::Mul { lhs, rhs } => {
                Self::mul_backward(lhs.clone(), rhs.clone(), grad);
            }
            Self::Dot { lhs, rhs, depth } => {
                Self::dot_backward(lhs.clone(), rhs.clone(), *depth, grad);
            }
            Self::Sum { t } => {
                Self::sum_backward(t.clone(), grad);
            }
            Self::Ln { t } => {
                Self::ln_backward(t.clone(), grad);
            }
            Self::Exp { t } => {
                Self::exp_backward(t.clone(), grad);
            }
            Self::Tanh { t } => {
                Self::tanh_backward(t.clone(), grad);
            }
            Self::Pow { t, i } => {
                Self::pow_backward(t.clone(), *i, grad);
            }
            Self::Neg { t } => {
                Self::neg_backward(t.clone(), grad);
            }
            Self::Max { t, u } => {
                Self::max_backward(t.clone(), *u, grad);
            }
            Self::ColsAt { t, indices } => {
                Self::cols_at_backward(t.clone(), indices.clone(), grad);
            }
            Self::Colify { t, field } => {
                Self::colify_backward(t.clone(), *field, grad);
            }
            Self::Colmax { t } => {
                Self::colmax_backward(t.clone(), grad);
            }
            Self::Reshape { t, .. } => {
                Self::reshape_backward(t.clone(), grad);
            }
            Self::Transpose { t, axes } => {
                Self::transpose_backward(t.clone(), &axes, grad);
            }
            Self::AtArgmax { t, of } => {
                Self::at_argmax_backward(t.clone(), of, grad);
            },
            Self::Softmax { t } => {
                Self::softmax_backwards(t.clone(), grad);
            }
        }
    }

    fn take_backward(self, grad: CPUTensor) {
        match self {
            Self::Add { lhs, rhs } => {
                Self::add_backward(lhs, rhs, grad);
            }
            Self::Sub { lhs, rhs } => {
                Self::sub_backward(lhs, rhs, grad);
            }
            Self::Mul { lhs, rhs } => {
                Self::mul_backward(lhs, rhs, grad);
            }
            Self::Dot { lhs, rhs, depth } => {
                Self::dot_backward(lhs, rhs, depth, grad);
            }
            Self::Sum { t } => {
                Self::sum_backward(t, grad);
            }
            Self::Ln { t } => {
                Self::ln_backward(t, grad);
            }
            Self::Exp { t } => {
                Self::exp_backward(t, grad);
            }
            Self::Tanh { t } => {
                Self::tanh_backward(t, grad);
            }
            Self::Pow { t, i } => {
                Self::pow_backward(t, i, grad);
            }
            Self::Neg { t } => {
                Self::neg_backward(t, grad);
            }
            Self::Max { t, u } => {
                Self::max_backward(t, u, grad);
            }
            Self::ColsAt { t, indices } => {
                Self::cols_at_backward(t, indices, grad);
            }
            Self::Colify { t, field } => {
                Self::colify_backward(t, field, grad);
            }
            Self::Colmax { t } => {
                Self::colmax_backward(t, grad);
            }
            Self::Reshape { t, .. } => {
                Self::reshape_backward(t, grad);
            }
            Self::Transpose { t, axes } => {
                Self::transpose_backward(t, &axes, grad);
            }
            Self::AtArgmax { t, of } => {
                Self::at_argmax_backward(t, &of, grad);
            },
            Self::Softmax { t } => {
                Self::softmax_backwards(t, grad);
            }
        }
    }
}

// ai generated tests -- ai generated tests -- ai generated tests

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_with_data(shape: Vec<usize>, values: &[f64]) -> CPUTensor {
        let mut tensor = CPUTensor::tensor(Fill { shape, with: 0.0 }).unwrap();
        assert_eq!(tensor.data.len(), values.len());
        tensor.data.clone_from_slice(values);
        tensor
    }

    #[test]
    fn transpose_swaps_axes_for_matrices() {
        let tensor = tensor_with_data(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let transposed = tensor.clone().transpose(&[1, 0]).unwrap();

        assert_eq!(transposed.shape(), &[3, 2]);

        for i in 0..3 {
            for j in 0..2 {
                assert_eq!(
                    transposed.get(&[i, j]).unwrap(),
                    tensor.get(&[j, i]).unwrap()
                );
            }
        }
    }

    #[test]
    fn transpose_reorders_higher_rank_axes() {
        let mut values = Vec::with_capacity(24);
        for i in 0..24 {
            values.push(i as f64);
        }
        let tensor = tensor_with_data(vec![2, 3, 4], &values);
        let transposed = tensor.clone().transpose(&[2, 0, 1]).unwrap();

        assert_eq!(transposed.shape(), &[4, 2, 3]);

        for a in 0..4 {
            for b in 0..2 {
                for c in 0..3 {
                    assert_eq!(
                        transposed.get(&[a, b, c]).unwrap(),
                        tensor.get(&[b, c, a]).unwrap()
                    );
                }
            }
        }
    }

    #[test]
    fn transpose_rejects_axis_length_mismatch() {
        let tensor = CPUTensor::tensor(Fill {
            shape: vec![2, 2],
            with: 1.0,
        })
        .unwrap();
        assert!(tensor.transpose(&[0]).is_none());
    }

    #[test]
    fn cols_at_selects_specified_columns() {
        let tensor = tensor_with_data(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let indices = CPUTensor::vector(vec![2.0, 0.0]).unwrap();

        let selected = tensor.cols_at(&indices).unwrap();

        assert_eq!(selected.shape(), &[2, 2]);
        assert_eq!(selected.get(&[0, 0]).unwrap(), tensor.get(&[0, 2]).unwrap());
        assert_eq!(selected.get(&[1, 0]).unwrap(), tensor.get(&[1, 2]).unwrap());
        assert_eq!(selected.get(&[0, 1]).unwrap(), tensor.get(&[0, 0]).unwrap());
        assert_eq!(selected.get(&[1, 1]).unwrap(), tensor.get(&[1, 0]).unwrap());
    }

    #[test]
    fn cols_at_backward_accumulates_gradients() {
        let tensor = tensor_with_data(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).autograd();
        let indices = CPUAutograd::vector(vec![2.0, 0.0]).unwrap();
        let indices_for_grad = indices.clone();

        let selected = tensor.cols_at(&indices).unwrap();
        selected.backward();

        let grad = tensor.into_grad().expect("expected gradient for input tensor");
        assert_eq!(grad.shape(), &[2, 3]);
        assert_eq!(grad.data, vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);

        assert!(
            indices_for_grad.into_grad().is_none(),
            "indices tensor should not accumulate gradient"
        );
    }

    #[test]
    fn softmax_respects_non_contiguous_layouts() {
        // Arrange input so last dimension (batch) is contiguous but activation values are strided.
        let base = tensor_with_data(vec![3, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let transposed = base.transpose(&[1, 0]).unwrap();

        // Expected softmax per batch over the two activation values differing by 3.0.
        let exp_small = 1.0_f64;
        let exp_large = 3.0_f64.exp();
        let denom = exp_small + exp_large;
        let expected_first = exp_small / denom;
        let expected_second = exp_large / denom;

        let result = transposed.clone().softmax().unwrap();
        let expected = vec![
            expected_first,
            expected_first,
            expected_first,
            expected_second,
            expected_second,
            expected_second,
        ];

        for (idx, (actual, expected)) in result.data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-12,
                "mismatch at flat index {}: expected {}, got {}",
                idx,
                expected,
                actual
            );
        }
    }

    #[test]
    fn softmax_backward_with_unit_upstream_grad_is_zero() {
        // The gradient of sum(softmax(x)) w.r.t x should be zero because outputs sum to one.
        let logits = tensor_with_data(vec![2, 1], &[1.0, 2.0]).autograd();
        let output = logits.clone().softmax().unwrap();

        output.backward();
        let grad = logits.into_grad().expect("expected gradient for logits");

        for (i, g) in grad.data.iter().enumerate() {
            assert!(
                g.abs() < 1e-12,
                "expected zero gradient at position {}, got {}",
                i,
                g
            );
        }
    }
}
