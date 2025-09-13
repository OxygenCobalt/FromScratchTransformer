use std::{cell::RefCell, rc::Rc};

use crate::tensor::{Fill, TensorInit};

use super::tensor::{SharpTensor, Tensor};

#[derive(Clone)]
pub struct Autograd<T: SharpTensor>(AutogradNode<T>);

impl<T: SharpTensor> Autograd<T> {
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

impl<T: SharpTensor> Tensor for Autograd<T> {
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
}

type AutogradNode<T> = Rc<Computation<T>>;

struct Computation<T: SharpTensor> {
    tensor: T,
    grad: RefCell<Option<T>>,
    op: Option<Operation<T>>,
}

impl<T: SharpTensor> Computation<T> {
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

enum Operation<T: SharpTensor> {
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
}

impl<T: SharpTensor> Operation<T> {
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
        'iterate: loop {
            let lhs_point: Vec<usize> = new_point
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < lhs.tensor.ndim())
                .map(|(i, x)| x % lhs.tensor.shape()[i])
                .collect();
            let rhs_point: Vec<usize> = new_point
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < rhs.tensor.ndim())
                .map(|(i, x)| x % rhs.tensor.shape()[i])
                .collect();
            let (lhs_lgrad, rhs_lgrad) = op(
                lhs.tensor.get(&lhs_point).unwrap(),
                rhs.tensor.get(&rhs_point).unwrap(),
            );
            let bgrad = *grad.get(&new_point).unwrap();
            *lhs_grad.get_mut(&lhs_point).unwrap() += lhs_lgrad * bgrad;
            *rhs_grad.get_mut(&rhs_point).unwrap() += rhs_lgrad * bgrad;

            for (p, s) in new_point.iter_mut().zip(grad.shape()) {
                let o = *p;
                *p = (*p + 1) % s;
                if *p > o {
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
                    grad
                        .dot(&rhs.tensor.tranpose(&rhs_axes).unwrap(), lhs_shift)
                        .unwrap(),
                );

                let rhs_shift = lhs.tensor.ndim() - depth;
                let mut lhs_axes: Vec<usize> = (0..lhs.tensor.ndim()).collect();
                lhs_axes.rotate_left(rhs_shift);
                rhs.backward(
                    lhs.tensor
                        .tranpose(&lhs_axes)
                        .unwrap()
                        .dot(grad, rhs_shift)
                        .unwrap(),
                );
            }
            Self::Sum { t } => t.backward(
                T::tensor(Fill {
                    shape: t.tensor.shape().to_vec(),
                    with: *grad.get(&[]).unwrap(),
                })
                .unwrap(),
            ),
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
        }
    }
}
