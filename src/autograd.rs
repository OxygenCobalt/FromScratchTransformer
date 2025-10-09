use core::f64;
use std::{cell::RefCell, rc::Rc};

use crate::tensor::{Field, Fill, Generate, TensorInit};

use super::tensor::{Tensor, TensorMut};

#[derive(Clone)]
pub struct Autograd<T: TensorMut>(AutogradNode<T>);

impl<T: TensorMut> Autograd<T> {
    pub fn new(matrix: T) -> Self {
        Self(Rc::new(Computation {
            tensor: matrix,
            grad: RefCell::new(None),
            op: None,
        }))
    }

    pub fn backward(&self) {
        self.0.backward_init();
    }

    pub fn into_grad(self) -> Option<T> {
        Rc::into_inner(self.0).and_then(|c| c.grad.into_inner())
    }
}

impl<T: TensorMut> Tensor for Autograd<T> {
    fn scalar(c: impl Into<f64>) -> Self {
        Self::new(T::scalar(c))
    }

    fn vector(v: impl Into<Vec<f64>>) -> Option<Self> {
        T::vector(v).map(Self::new)
    }

    fn tensor(tv: impl TensorInit) -> Option<Self> {
        T::tensor(tv).map(Self::new)
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

    fn neg(self) -> Self {
        Operation::Neg { t: self.0 }.forward().unwrap()
    }

    fn max(self, y: f64) -> Self {
        Operation::Max { t: self.0, u: y }.forward().unwrap()
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

    fn transpose(&self, axes: &[usize]) -> Option<Self> {
        Operation::Transpose {
            t: self.0.clone(),
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
}

type AutogradNode<T> = Rc<Computation<T>>;

struct Computation<T: TensorMut> {
    tensor: T,
    grad: RefCell<Option<T>>,
    op: Option<Operation<T>>,
}

impl<T: TensorMut> Computation<T> {
    fn backward_init(&self) {
        self.backward(T::scalar(1.0));
    }

    pub fn backward(&self, grad: T) {
        if let Some(op) = &self.op {
            op.backward(&grad);
        }
        let new_grad = match self.grad.borrow().as_ref() {
            Some(existing) => Some(existing.clone().add(&grad).unwrap()),
            None => Some(grad),
        };
        *self.grad.borrow_mut() = new_grad;
    }
}

enum Operation<T: TensorMut> {
    Add {
        lhs: AutogradNode<T>,
        rhs: AutogradNode<T>,
    },
    Sub {
        lhs: AutogradNode<T>,
        rhs: AutogradNode<T>,
    },
    Mul {
        lhs: AutogradNode<T>,
        rhs: AutogradNode<T>,
    },
    Dot {
        lhs: AutogradNode<T>,
        rhs: AutogradNode<T>,
        depth: usize,
    },
    Sum {
        t: AutogradNode<T>,
    },
    Ln {
        t: AutogradNode<T>,
    },
    Exp {
        t: AutogradNode<T>,
    },
    Pow {
        t: AutogradNode<T>,
        i: i32,
    },
    Neg {
        t: AutogradNode<T>,
    },
    Max {
        t: AutogradNode<T>,
        u: f64,
    },
    Colify {
        t: AutogradNode<T>,
        field: Field,
    },
    Colmax {
        t: AutogradNode<T>,
    },
    Reshape {
        t: AutogradNode<T>,
        shape: Vec<usize>,
    },
    Transpose {
        t: AutogradNode<T>,
        axes: Vec<usize>,
    },
    AtArgmax {
        t: AutogradNode<T>,
        of: AutogradNode<T>,
    },
}

impl<T: TensorMut> Operation<T> {
    fn arithmetic_backward(
        lhs: &AutogradNode<T>,
        rhs: &AutogradNode<T>,
        grad: &T,
        op: impl Fn(&f64, &f64) -> (f64, f64),
    ) {
        let mut lhs_grad = T::tensor(Fill {
            shape: lhs.tensor.shape().to_vec(),
            with: 0.0,
        })
        .unwrap();
        let mut rhs_grad = T::tensor(Fill {
            shape: rhs.tensor.shape().to_vec(),
            with: 0.0,
        })
        .unwrap();

        let mut new_point = vec![0; grad.ndim()];
        let mut lhs_point = vec![0; lhs.tensor.ndim()];
        let mut rhs_point = vec![0; rhs.tensor.ndim()];
        'iterate: loop {
            for (i, v) in new_point.iter().enumerate() {
                if i < lhs_point.len() {
                    lhs_point[i] = v % lhs.tensor.shape()[i];
                }
                if i < rhs_point.len() {
                    rhs_point[i] = v % rhs.tensor.shape()[i];
                }
            }
            let (lhs_lgrad, rhs_lgrad) = op(
                lhs.tensor.get(&lhs_point).unwrap(),
                rhs.tensor.get(&rhs_point).unwrap(),
            );
            let bgrad = *grad.get(&new_point).unwrap();
            *lhs_grad.get_mut(&lhs_point).unwrap() += lhs_lgrad * bgrad;
            *rhs_grad.get_mut(&rhs_point).unwrap() += rhs_lgrad * bgrad;

            for (p, s) in new_point.iter_mut().zip(grad.shape()) {
                if *p == *s - 1 {
                    *p = 0;
                } else {
                    *p += 1;
                    continue 'iterate;
                }
            }
            break;
        }
        rhs.backward(rhs_grad);
        lhs.backward(lhs_grad);
    }

    fn forward(self) -> Option<Autograd<T>> {
        let tensor = match &self {
            Self::Add { lhs, rhs } => lhs.tensor.add(&rhs.tensor),
            Self::Sub { lhs, rhs } => lhs.tensor.sub(&rhs.tensor),
            Self::Mul { lhs, rhs } => lhs.tensor.mul(&rhs.tensor),
            Self::Dot { lhs, rhs, depth } => lhs.tensor.dot(&rhs.tensor, *depth),
            Self::Sum { t } => Some(t.tensor.sum()),
            Self::Ln { t } => Some(t.tensor.clone().ln()),
            Self::Exp { t } => Some(t.tensor.clone().exp()),
            Self::Pow { t, i } => Some(t.tensor.clone().pow(*i)),
            Self::Neg { t } => Some(t.tensor.clone().neg()),
            Self::Max { t, u } => Some(t.tensor.clone().max(*u)),
            Self::Colify { t, field } => t.tensor.colify(*field),
            Self::Colmax { t } => t.tensor.colmax(),
            Self::Reshape { t, shape } => t.tensor.clone().reshape(shape),
            Self::Transpose { t, axes } => t.tensor.transpose(&axes),
            Self::AtArgmax { t, of } => t.tensor.at_argmax(&of.tensor),
        };
        tensor.map(|tensor| {
            Autograd(Rc::new(Computation {
                tensor,
                grad: RefCell::new(None),
                op: Some(self),
            }))
        })
    }

    fn backward(&self, grad: &T) {
        match self {
            Self::Add { lhs, rhs } => {
                Self::arithmetic_backward(lhs, rhs, grad, |_, _| (1.0, 1.0));
            }
            Self::Sub { lhs, rhs } => {
                Self::arithmetic_backward(lhs, rhs, grad, |_, _| (1.0, -1.0));
            }
            Self::Mul { lhs, rhs } => {
                Self::arithmetic_backward(lhs, rhs, grad, |lhs, rhs| (*rhs, *lhs));
            }
            Self::Dot { lhs, rhs, depth } => {
                let lhs_shift = rhs.tensor.ndim() - depth;
                let mut rhs_axes: Vec<usize> = (0..rhs.tensor.ndim()).collect();
                rhs_axes.rotate_right(lhs_shift);
                lhs.backward(
                    grad.dot(&rhs.tensor.transpose(&rhs_axes).unwrap(), lhs_shift)
                        .unwrap(),
                );

                let rhs_shift = lhs.tensor.ndim() - depth;
                let mut lhs_axes: Vec<usize> = (0..lhs.tensor.ndim()).collect();
                lhs_axes.rotate_left(rhs_shift);
                rhs.backward(
                    lhs.tensor
                        .transpose(&lhs_axes)
                        .unwrap()
                        .dot(grad, rhs_shift)
                        .unwrap(),
                );
            }
            Self::Sum { t } => {
                let mut t_grad = T::tensor(Fill {
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
                t.backward(t_grad);
            }
            Self::Ln { t } => {
                t.backward(t.tensor.clone().pow(-1).mul(grad).unwrap());
            }
            Self::Exp { t } => {
                t.backward(t.tensor.clone().exp().mul(grad).unwrap());
            }
            Self::Pow { t, i } => {
                t.backward(
                    t.tensor
                        .clone()
                        .pow(i - 1)
                        .mul(&Tensor::scalar(*i as f64))
                        .unwrap()
                        .mul(grad)
                        .unwrap(),
                );
            }
            Self::Neg { t } => {
                t.backward(grad.clone().neg());
            }
            Self::Max { t, u } => {
                let mut loc = t.tensor.clone();
                loc.iter_mut()
                    .for_each(|x| *x = if *x >= *u { 1.0 } else { 0.0 });
                t.backward(loc.mul(grad).unwrap());
            }
            Self::Colify { t, field } => {
                let mut t_grad = T::tensor(Fill {
                    shape: t.tensor.shape().to_vec(),
                    with: 0.0,
                })
                .unwrap();
                let locations = field
                    .locations_on(*t.tensor.shape().first().unwrap())
                    .unwrap();
                let new_shape = grad.shape();
                let mut new_point = vec![0; new_shape.len()];
                let mut old_point = vec![0; t.tensor.ndim()];
                'iterate: loop {
                    let field_idx = new_point[0];
                    let location_idx = new_point[1];
                    let x = -(field.padding as i64) + (((location_idx % locations) * field.stride) + (field_idx % field.size)) as i64;
                    let y = -(field.padding as i64) + (((location_idx / locations) * field.stride) + (field_idx / field.size)) as i64;
                    if x >= 0 && y >= 0 && x < t.tensor.shape()[0] as i64 && y < t.tensor.shape()[1] as i64 {
                        old_point[0] = x as usize;
                        old_point[1] = y as usize;
                        old_point[2..].copy_from_slice(&new_point[2..]);
                        *t_grad.get_mut(&old_point).unwrap() += *grad.get(&new_point).unwrap()
                    }
                    for (p, s) in new_point.iter_mut().zip(grad.shape().iter()) {
                        if *p == *s - 1 {
                            *p = 0;
                        } else {
                            *p += 1;
                            continue 'iterate;
                        }
                    }
                    break;
                }
                t.backward(t_grad);
            }
            Self::Colmax { t } => {
                let mut t_grad = T::tensor(Fill {
                    shape: t.tensor.shape().to_vec(),
                    with: 0.0,
                })
                .unwrap();
                let mut grad_point = vec![0; grad.ndim()];
                let mut column_point = vec![0; t.tensor.ndim()];
                'iterate: loop {
                    column_point[1..].copy_from_slice(&grad_point);
                    let mut max_idx = 0;
                    let mut max_value = f64::NEG_INFINITY;
                    for i in 0..t.tensor.shape()[0] {
                        column_point[0] = i;
                        let cur_value = t.tensor.get(&column_point).unwrap();
                        if *cur_value > max_value {
                            max_idx = i;
                            max_value = *cur_value;
                        }
                    }
                    column_point[0] = max_idx;
                    // max point, frwd here
                    *t_grad.get_mut(&column_point).unwrap() = *grad.get(&grad_point).unwrap();
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
                t.backward(t_grad);
            }
            Self::Reshape { t, shape } => {
                t.backward(grad.clone().reshape(t.tensor.shape()).unwrap())
            }
            Self::Transpose { t, axes } => {
                let mut rev_axes = vec![0; axes.len()];
                for i in 0..axes.len() {
                    rev_axes[axes[i]] = i;
                }
                t.backward(grad.transpose(&rev_axes).unwrap());
            }
            Self::AtArgmax { t, of } => {
                let mut t_grad = T::tensor(Fill {
                    shape: t.tensor.shape().to_vec(),
                    with: 0.0,
                })
                .unwrap();
                let mut t_grad_point = vec![0; t_grad.shape().len()];
                let mut grad_point = vec![0; grad.shape().len()];
                'iterate: loop {
                    let mut argmax = 0;
                    let mut max = f64::NEG_INFINITY;
                    t_grad_point[1..].copy_from_slice(&grad_point);
                    for i in 0..t.tensor.shape()[0] {
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
                t.backward(t_grad);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{CPUTensor, Field, Fill};

}
