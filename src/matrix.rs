use std::ops::Range;
use std::ops::{Add, AddAssign, Mul, MulAssign, RangeBounds, RangeTo, Sub, SubAssign};
use std::fmt::{Debug, Formatter};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub m: usize,
    pub n: usize
}

impl Shape {
    pub fn square(n: usize) -> Self {
        Self { m: n, n }
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
    shape: Shape
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

    pub fn identity(n: usize) -> Self {
        Self::new(Shape::square(n), |i| {
            if i % (n + 1) == 0 {
                1f64
            } else {
                0f64
            }
        })
    }

    pub fn noisy(shape: Shape, range: Range<f64>) -> Self {
        Self::new(shape, |i| rand::random_range(range.clone()))
    }

    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.shape.n + j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.shape.n + j] = value;
    }


    pub unsafe fn add_assign_unchecked(&mut self, rhs: &Self) {
        for i in 0..self.data.len() {
            unsafe {
                *self.data.get_unchecked_mut(i) += rhs.data.get_unchecked(i)
            }
        }
    }

    pub unsafe fn sub_assign_unchecked(&mut self, rhs: &Self) {
        for i in 0..self.data.len() {
            unsafe {
                *self.data.get_unchecked_mut(i) -= rhs.data.get_unchecked(i)
            }
        }
    }

    pub unsafe fn mul_assign_unchecked(&mut self, rhs: &Self) {
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
        let mut new = Self::null(Shape { m: self.shape.n, n: self.shape.m });
        let mut data = Vec::new();
        data.resize(self.data.len(), 0f64);
        for i in 0..new.shape.m {
            for j in 0..new.shape.n {
                new.set(i, j, self.get(j, i));
            }
        }
        new
    }

    pub fn apply(&mut self, transform: impl Fn(f64) -> f64) {
        for v in &mut self.data {
            *v = transform(*v);
        }
    }

    fn assert_equal_shape(&self, other: &Self) {
        if self.shape != other.shape {
            panic!("shape mismatch for addition: {:?} != {:?}", self.shape.m, self.shape.n);
        }
    }

    fn assert_equal_n_m(&self, rhs: &Self) {
        if self.shape.n != rhs.shape.m {
            panic!("shape mismatch: {:?} is incompatible with {:?}, {} != {}", 
            self.shape, rhs.shape, self.shape.n, rhs.shape.m);
        }
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

impl Add for Matrix {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign for Matrix {
    fn add_assign(&mut self, rhs: Self) {
        self.assert_equal_shape(&rhs);
        unsafe { self.add_assign_unchecked(&rhs); }
    }
}

impl Sub for Matrix {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl SubAssign for Matrix {
    fn sub_assign(&mut self, rhs: Self) {
        self.assert_equal_shape(&rhs);
        unsafe { self.sub_assign_unchecked(&rhs) }
    }
}

impl Mul for Matrix {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl MulAssign for Matrix {
    fn mul_assign(&mut self, rhs: Self) {
        self.assert_equal_n_m(&rhs);
        unsafe { self.mul_assign_unchecked(&rhs) }
    }
}
