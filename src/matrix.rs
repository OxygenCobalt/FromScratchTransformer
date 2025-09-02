use std::fmt::{Debug, Formatter};
use std::sync::LazyLock;
use std::io::{self, Read, Write};

use rand_distr::{Distribution, Normal};

static NORMAL: LazyLock<Normal<f64>> = std::sync::LazyLock::new(|| Normal::new(0.0, 1.0).unwrap());

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

    pub fn new(shape: Shape, mut fill: impl FnMut(usize, usize) -> f64) -> Self {
        let mut data = Vec::new();
        let mut n = 0;
        data.resize_with(shape.m * shape.n, || {
            let i = n / shape.n;
            let j = n % shape.n;
            let res = fill(i, j);
            n += 1;
            res
        });
        Self { shape, data }
    }

    pub fn noisy(shape: Shape) -> Self {
        Self::new(shape, |_, _| NORMAL.sample(&mut rand::rng()))
    }

    pub fn scalar(c: f64) -> Self {
        Self { data: vec![c], shape: Shape { m: 1, n: 1 } }
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

    pub fn get(&self, i: usize, j: usize) -> f64 {
        let idx = self.index_of(i, j);
        self.data[idx]
    }

    pub unsafe fn get_unchecked(&self, i: usize, j: usize) -> f64 {
        let idx = self.index_of(i, j);
        unsafe { *self.data.get_unchecked(idx) }
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

    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        let idx = self.index_of(i, j);
        self.data[idx] = value;
    }

    pub unsafe fn set_unchecked(&mut self, i: usize, j: usize, value: f64) {
        let idx = self.index_of(i, j);
        unsafe { *self.data.get_unchecked_mut(idx) = value }
    }

    fn index_of(&self, i: usize, j: usize) -> usize {
        i * self.shape.n + j
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

    pub fn scale(self, c: f64) -> Self {
        self.apply(|n| c * n)
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
                    sum += unsafe { old.get_unchecked(i, k) } * unsafe { rhs.get_unchecked(k, j) }
                }
                unsafe { self.set_unchecked(i, j, sum) };
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
                unsafe { new.set_unchecked(i, j, self.get_unchecked(j, i)); }
            }
        }
        new
    }

    pub fn apply(mut self, transform: impl Fn(f64) -> f64) -> Self {
        for v in &mut self.data {
            *v = transform(*v);
        }
        self
    }

    pub fn flatten(&self) -> impl Iterator<Item=&f64> {
        self.data.iter()
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"Matrix\0\0")?;
        write.write_all(&self.shape.m.to_le_bytes())?;
        write.write(&self.shape.n.to_le_bytes())?;
        write.write(&self.data.iter().map(|v| v.to_le_bytes()).flatten().collect::<Vec<u8>>())?;
        Ok(())
    }

    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read(&mut signature)?;
        if &signature != b"Matrix\0\0" {
            return Err(io::Error::new(io::ErrorKind::Other, "invalid matrix signature"));
        }
        let mut nb = [0u8; 8];
        read.read(&mut nb)?;
        let mut mb = [0u8; 8];
        read.read_exact(&mut mb)?;

        let shape = Shape { m: usize::from_le_bytes(nb), n: usize::from_le_bytes(mb) };

        let mut data = Vec::new();
        for _ in 0..(shape.m * shape.n) {
            let mut fb = [0u8; 8];
            read.read_exact(&mut fb)?;
            data.push(f64::from_le_bytes(fb));
        }

        Ok(Self {
            shape,
            data
        })
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
