use std::fmt::{Debug, Formatter};
use std::sync::LazyLock;

use rand_distr::{Distribution, Normal};

static NORMAL: LazyLock<Normal<f64>> = std::sync::LazyLock::new(|| Normal::new(0.0, 1.0).unwrap());

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub m: usize,
    pub n: usize,
}

impl Shape {
    pub fn vector(n: usize) -> Self {
        Self { m: n, n: 1 }
    }
}

impl Debug for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write![f, "{}x{}", self.m, self.n]
    }
}

#[derive(Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    shape: Shape,
}

impl Matrix {
    pub fn null(shape: Shape) -> Self {
        let mut data = Vec::new();
        data.resize(shape.m * shape.n, 0f64);
        Self { data, shape }
    }

    pub fn new(shape: Shape, mut fill: impl FnMut(usize) -> f64) -> Self {
        let mut data = Vec::new();
        let mut i = 0;
        data.resize_with(shape.m * shape.n, || {
            let res = fill(i);
            i += 1;
            res
        });
        Self { shape, data }
    }

    pub fn noisy(shape: Shape) -> Self {
        Self::new(shape, |_i| NORMAL.sample(&mut rand::rng()))
    }

    pub fn argmax(&self) -> usize {
        let mut maxi = 0;
        for (i, v) in self.data.iter().enumerate() {
            if *v > self.data[maxi] {
                maxi = i;
            }
        }
        return maxi;
    }

    pub fn vector(data: Vec<f64>) -> Self {
        Self {
            shape: Shape::vector(data.len()),
            data,
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn length(&self) -> f64 {
        self.data.iter().map(|n| n * n).sum()
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.shape.n + j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.shape.n + j] = value;
    }

    pub fn add(mut self, rhs: &Self) -> Self {
        self.add_assign(rhs);
        self
    }

    pub fn add_assign(&mut self, rhs: &Self) {
        if self.shape != rhs.shape {
            panic!("shape mismatch for element-wise addition: {:?} != {:?}", self.shape, rhs.shape);
        }
        unsafe { self.add_assign_unchecked(rhs) }
    }

    pub unsafe fn add_assign_unchecked(&mut self, rhs: &Self) {
        for i in 0..self.data.len() {
            unsafe { *self.data.get_unchecked_mut(i) += rhs.data.get_unchecked(i) }
        }
    }

    pub fn sub(mut self, rhs: &Self) -> Self {
        self.sub_assign(rhs);
        self
    }

    pub fn sub_assign(&mut self, rhs: &Self) {
        if self.shape != rhs.shape {
            panic!("shape mismatch for element-wise subtraction: {:?} != {:?}", self.shape, rhs.shape);
        }
        unsafe { self.sub_assign_unchecked(rhs); }
    }

    pub unsafe fn sub_assign_unchecked(&mut self, rhs: &Self) {
        for i in 0..self.data.len() {
            unsafe { *self.data.get_unchecked_mut(i) -= rhs.data.get_unchecked(i) }
        }
    }

    pub fn mul(mut self, rhs: &Self) -> Self {
        if self.shape != rhs.shape {
            panic!("shape mismatch for element-wise multiplication: {:?} != {:?}", self.shape, rhs.shape);
        }
        unsafe { self.mul_assign_unchecked(&rhs) };
        self
    }

    pub unsafe fn mul_assign_unchecked(&mut self, rhs: &Self) {
        for i in 0..self.data.len() {
            unsafe { *self.data.get_unchecked_mut(i) *= rhs.data.get_unchecked(i) }
        }
    }

    pub fn dot(mut self, rhs: &Self) -> Self {
        self.dot_assign(rhs);
        self
    }

    pub fn dot_assign(&mut self, rhs: &Self) {
        if self.shape.n != rhs.shape.m {
            panic!(
                "shape mismatch for matrix multiplication: {:?} is incompatible with {:?}, {} != {}",
                self.shape, rhs.shape, self.shape.n, rhs.shape.m
            );
        }
        unsafe { self.dot_assign_unchecked(rhs); }
    }

    pub unsafe fn dot_assign_unchecked(&mut self, rhs: &Self) {
        let old = self.clone();
        self.shape.n = rhs.shape.n;
        self.data.resize(self.shape.m * self.shape.n, 0f64);
        for i in 0..self.shape.m {
            for j in 0..self.shape.n {
                let mut sum = 0f64;
                for k in 0..old.shape.n {
                    sum += old.get(i, k) * rhs.get(k, j)
                }
                self.set(i, j, sum);
            }
        }
    }

    pub fn transpose(self) -> Self {
        let mut new = Self::null(Shape {
            m: self.shape.n,
            n: self.shape.m,
        });
        let mut data = Vec::new();
        data.resize(self.data.len(), 0f64);
        for i in 0..new.shape.m {
            for j in 0..new.shape.n {
                new.set(i, j, self.get(j, i));
            }
        }
        new
    }

    pub fn scale(self, c: f64) -> Self {
        self.apply(|n| c * n)
    }

    pub fn apply(mut self, transform: impl Fn(f64) -> f64) -> Self {
        for v in &mut self.data {
            *v = transform(*v);
        }
        self
    }

    pub fn apply_indexed(mut self, transform: impl Fn(usize, usize, f64) -> f64) -> Self {
        let mut i = 0;
        let mut j = 0;
        for v in &mut self.data {
            *v = transform(i, j, *v);
            j += 1;
            if j >= self.shape.n {
                i += 1;
                j = 0;
            } 
        }
        self
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.shape.m {
            write![f, "["]?;
            for j in 0..self.shape.n {
                write![f, "{}", self.get(i, j)]?;
                if j < self.shape.n - 1 {
                    write![f, " "]?;
                }
            }
            write![f, "]"]?;
            if i < self.shape.m - 1 {
                write![f, "\n"]?;
            }
        }
        Ok(())
    }
}

