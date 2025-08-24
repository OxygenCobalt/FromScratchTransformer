use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign};
use std::fmt::{Debug, Formatter};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    pub m: usize,
    pub n: usize
}

impl Dimensions {
    pub fn square(n: usize) -> Self {
        Self { m: n, n }
    }
}

impl Debug for Dimensions {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write![f, "{}x{}", self.m, self.n]
    }
}

#[derive(Clone, PartialEq)]
pub struct Matrix {
    data: Vec<f64>,
    dimensions: Dimensions
}

impl Matrix {
    pub fn null(dimensions: Dimensions) -> Self {
        let mut data = Vec::new();
        data.resize(dimensions.m * dimensions.n, 0f64);
        Self { data, dimensions }
    }

    pub fn new(dimensions: Dimensions, mut fill: impl FnMut(usize) -> f64) -> Self {
        let mut data = Vec::new();
        let mut i = 0;
        data.resize_with(dimensions.m * dimensions.n, || {
            let res = fill(i);
            i += 1;
            res
        });
        Self { dimensions, data }
    }

    pub fn identity(n: usize) -> Self {
        Self::new(Dimensions::square(n), |i| {
            if i % (n + 1) == 0 {
                1f64
            } else {
                0f64
            }
        })
    }

    pub fn transpose(self) -> Self {
        let mut new = Self::null(Dimensions { m: self.dimensions.n, n: self.dimensions.m });
        let mut data = Vec::new();
        data.resize(self.data.len(), 0f64);
        for i in 0..new.dimensions.m {
            for j in 0..new.dimensions.n {
                new.set(i, j, self.get(j, i));
            }
        }
        new
    }

    pub fn inc(dimensions: Dimensions) -> Self {
        Self::new(dimensions, |i| i as f64)
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.dimensions.n + j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.data[i * self.dimensions.n + j] = value;
    }

    pub unsafe fn add_assign_unchecked(&mut self, rhs: Self) {
        for i in 0..self.data.len() {
            unsafe {
                *self.data.get_unchecked_mut(i) += rhs.data.get_unchecked(i)
            }
        }
    }

    pub unsafe fn sub_assign_unchecked(&mut self, rhs: Self) {
        for i in 0..self.data.len() {
            unsafe {
                *self.data.get_unchecked_mut(i) -= rhs.data.get_unchecked(i)
            }
        }
    }

    pub unsafe fn mul_assign_unchecked(&mut self, rhs: Self) {
        let old = self.clone();
        self.dimensions.n = rhs.dimensions.n;
        self.data.resize(self.dimensions.m * self.dimensions.n, 0f64);
        for i in 0..self.dimensions.m {
            for j in 0..self.dimensions.n {
                let mut sum = 0f64;
                for k in 0..old.dimensions.n {
                    sum += old.get(i, k) * rhs.get(k, j)
                }
                self.set(i, j, sum);
            }
        }
    }

    fn assert_equal_dimensions(&self, other: &Self) {
        if self.dimensions != other.dimensions {
            panic!("dimension mismatch for addition: {:?} != {:?}", self.dimensions.m, self.dimensions.n);
        }
    }

    fn assert_equal_n_m(&self, rhs: &Self) {
        if self.dimensions.n != rhs.dimensions.m {
            panic!("dimension mismatch: {:?} is incompatible with {:?}, {} != {}", 
            self.dimensions, rhs.dimensions, self.dimensions.n, rhs.dimensions.m);
        }
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.dimensions.m {
            write![f, "["]?;
            for j in 0..self.dimensions.n {
                write![f, "{}", self.get(i, j)]?;
                if j < self.dimensions.n - 1 {
                    write![f, " "]?;
                }
            }
            write![f, "]"]?;
            if i < self.dimensions.m - 1 {
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
        self.assert_equal_dimensions(&rhs);
        unsafe { self.add_assign_unchecked(rhs); }
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
        self.assert_equal_dimensions(&rhs);
        unsafe { self.sub_assign_unchecked(rhs) }
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
        unsafe { self.mul_assign_unchecked(rhs) }
    }
}