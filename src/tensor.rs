pub trait Tensor where Self: Sized + Clone {
    fn scalar(c: impl Into<f64>) -> Self;
    fn vector(v: impl Into<Vec<f64>>) -> Option<Self>;
    fn tensor(f: impl FnMut() -> f64, shape: &[usize]) -> Option<Self>;
    fn shape(&self) -> &[usize];
    fn get(&self, point: &[usize]) -> Option<&f64>;
    fn add(self, other: &Self) -> Option<Self>;
    fn sub(self, other: &Self) -> Option<Self>;
    fn mul(self, other: &Self) -> Option<Self>;
    fn dot(self, other: &Self, axes: (&[usize], &[usize])) -> Option<Self>;
    fn norm(self, i: u8) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn inv(self) -> Self;
    fn neg(self) -> Self;
    fn max(self, y: f64) -> Self;
}

pub trait SharpTensor: Tensor {
    fn get_mut(&mut self, point: &[usize]) -> Option<&mut f64>;
    fn tranpose(self, axes: &[usize]) -> Option<Self>;
}

#[derive(Clone, PartialEq)]
pub struct CPUTensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

impl CPUTensor {
    fn len(shape: &[usize]) -> usize {
        shape.iter().cloned().reduce(|a, v| a * v).unwrap()
    }

    fn point_index(&self, of: &[usize]) -> Option<usize> {
        if of.len() != self.shape.len() {
            return None;
        }
        let mut idx = 0;
        let mut mult = 1;
        for (i, p) in of.iter().enumerate().rev() {
            idx += mult * p;
            mult *= self.shape[i];
        }
        Some(idx)
    }

    fn element_op(mut self, other: &Self, op: impl Fn(&mut f64, &f64)  -> ()) -> Option<Self> {
        if self.shape == other.shape {
            self.data.iter_mut().zip(other.data.iter()).for_each(|(rhs, lhs)| op(rhs, lhs))
        } else {
            match other.shape.len() {
                // dont know generalized broadcasting yet, these are the only cases i need right now
                0 => self.data.iter_mut().for_each(|rhs| op(rhs, other.get(&[]).unwrap())),
                1 if self.shape.len() == 2 && self.shape[0] == other.shape[0] => {
                    for i in 0..other.shape[0] {
                        for j in 0..self.shape[1] {
                            op(self.get_mut(&[i, j]).unwrap(), other.get(&[i]).unwrap())
                        }
                    }
                },
                _ => return None
            }
        }
        Some(self)
    }
}

impl Tensor for CPUTensor {
    fn scalar(c: impl Into<f64>) -> Self {
        Self { shape: Default::default(), data: vec![c.into()] }
    }

    fn vector(v: impl Into<Vec<f64>>) -> Option<Self> {
        let data =  v.into();
        if data.is_empty() {
            return None;
        }
        Some(Self { shape: vec![data.len()], data })
    }

    fn tensor(f: impl FnMut() -> f64, shape: &[usize]) -> Option<Self> {
        if shape.is_empty() {
            return None;
        }
        let mut data = Vec::new();
        data.resize_with(Self::len(shape), f);
        Some(Self { data, shape: shape.to_vec() })
    }
    
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn get(&self, point: &[usize]) -> Option<&f64> {
        self.point_index(point).and_then(|i| self.data.get(i))
    }

    fn add(self, other: &Self) -> Option<Self>{
        self.element_op(other, |rhs, lhs| *rhs += lhs)
    }

    fn sub(self, other: &Self) -> Option<Self> {
        self.element_op(other, |rhs, lhs| *rhs -= lhs)
    }

    fn mul(self, other: &Self) -> Option<Self> {
        self.element_op(other, |rhs, lhs| *rhs *= lhs)
    }

    fn dot(self, other: &Self, axes: (&[usize], &[usize])) -> Option<Self> {
        if axes.0.is_empty() || axes.0.iter().any(|i| *i > self.shape.len()) {
            return None;
        }
        if axes.1.is_empty() || axes.1.iter().any(|i| *i > self.shape.len()) {
            return None;
        }
        if axes.0.len() != axes.1.len() {
            return None;
        }
        if axes.0.iter().zip(axes.1.iter()).any(|(l, r)| self.shape[*l] != other.shape[*r]) {
            return None;
        }
        let new_shape: Vec<usize> = 
            self.shape.iter().enumerate().filter(|(i, _)| !axes.0.contains(i)).map(|(_, a)| a)
                .chain(other.shape.iter().enumerate().filter(|(i, _)| !axes.1.contains(i)).map(|(_, a)| a))
                .cloned()
                .collect();

        let mut new = Self {
            data: vec![0.0; Self::len(new_shape.as_slice())],
            shape: new_shape,
        };

        let mut point = vec![0; new.shape.len()];
        'iterate: loop {
            fn recursive(point: Vec<usize>, sum_vars: Vec<usize>, axes: (&[usize], &[usize]), new: &mut CPUTensor, lhs: &CPUTensor, rhs: &CPUTensor) {
                if sum_vars.len() == axes.0.len() {
                    let mut lhs_point: Vec<usize> = point.clone();
                    let zipped_lhs_point: Vec<usize> = (0..lhs.shape.len())
                        .map(|i| axes.0.iter().position(|j| i == *j)
                        .map(|k| sum_vars[k]).unwrap_or_else(|| lhs_point.remove(0))
                    ).collect();
                    let mut rhs_point = point.clone();
                    let zipped_rhs_point: Vec<usize> = (0..lhs.shape.len())
                        .map(|i| axes.1.iter().position(|j| i == *j)
                        .map(|k| sum_vars[k]).unwrap_or_else(|| rhs_point.remove(0))).collect();
                    *new.get_mut(point.as_slice()).unwrap() += *lhs.get(&zipped_lhs_point).unwrap() * *rhs.get(&zipped_rhs_point).unwrap()
                } else {
                    for i in 0..lhs.shape[axes.0.len()] {
                        let mut next = sum_vars.clone();
                        next.push(i);
                        recursive(point.clone(), next, axes, new, lhs, rhs);
                    }
                }
            }

            recursive(point.clone(), Vec::new(), axes, &mut new, &self, other);
            
            for (p, s) in point.iter_mut().zip(new.shape.iter()).rev() {
                let o = *p;
                *p = (*p + 1) % s;
                if *p > o {
                    continue 'iterate;
                }
            }

            break
        }
        
        Some(new)
    }

    fn norm(self, i: u8) -> Self {
        Self::scalar(self.data.iter().map(|x| x.abs().powi(i.into())).sum::<f64>())
    }

    fn ln(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = x.ln());
        self
    }

    fn exp(mut self) -> Self  {
        self.data.iter_mut().for_each(|x| *x = x.exp());
        self
    }

    fn inv(mut self) -> Self {
        self.data.iter_mut().for_each(|x| *x = 1.0 / *x);
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
}

impl SharpTensor for CPUTensor {
    fn tranpose(self, axes: &[usize]) -> Option<Self> {
        if self.shape.len() != axes.len() || axes.iter().any(|i| *i > self.shape.len()) {
            return None;
        }
        let new_shape: Vec<usize> = axes.iter().map(|j| self.shape[*j]).collect();
        let mut new = Self {
            data: vec![0.0; Self::len(new_shape.as_slice())],
            shape: new_shape,
        };

        let mut point = vec![0; self.shape.len()];
        'iterate: loop {
            let new_point: Vec<usize> = axes
                .iter()
                .map(|i| point[*i]).collect();
            *new.get_mut(&new_point).unwrap() = 
                *self.get(&point.as_slice()).unwrap();

            for (p, s) in point.iter_mut().zip(self.shape.iter()).rev() {
                let o = *p;
                *p = (*p + 1) % s;
                if *p > o {
                    continue 'iterate;
                }
            }

            break
        }
        
        Some(new)
    }

    fn get_mut(&mut self, point: &[usize]) -> Option<&mut f64> {
        self.point_index(point).and_then(|i| self.data.get_mut(i))
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::SmallRng, RngCore, SeedableRng};

    use crate::tensor::{CPUTensor, Tensor, SharpTensor};

    #[test]
    fn get() {
        let mut rng = SmallRng::from_seed([0; 32]);
        let tensor = CPUTensor::tensor(|| { rng.next_u32() } as f64, &[3, 3]).unwrap();
        assert_eq!(tensor.get(&[1, 2]).cloned(), Some(88327031.0));
    }

    #[test]
    fn set() {
        let mut rng = SmallRng::from_seed([0; 32]);
        let mut tensor = CPUTensor::tensor(|| { rng.next_u32() } as f64, &[3, 3]).unwrap();
        *tensor.get_mut(&[1, 2]).unwrap() = 3.0;
        assert_eq!(tensor.get(&[1, 2]).cloned(), Some(3.0));
    }


    #[test]
    fn transpose() {
        let mut rng = SmallRng::from_seed([0; 32]);
        let mut tensor = CPUTensor::tensor(|| { rng.next_u32() } as f64, &[3, 3, 3]).unwrap();
        assert_eq!(tensor.get(&[2, 1, 0]).cloned(), Some(560731428.0));
        tensor = tensor.tranpose(&[1, 0, 2]).unwrap();
        assert_eq!(tensor.get(&[1, 2, 0]).cloned(), Some(560731428.0));
    }

    #[test]
    fn dot() {
        let mut rng = SmallRng::from_seed([0; 32]);
        let mut a = CPUTensor::tensor(|| 0.5, &[3, 3]).unwrap();
        let mut b = CPUTensor::tensor(|| 0.5, &[3, 3]).unwrap();
        let res = a.dot(&b, (&[1], &[1])).unwrap();
        dbg!(res.get(&[1, 1]));
        assert!(false)
    }
}