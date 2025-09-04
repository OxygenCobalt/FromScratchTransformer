

use std::{cell::RefCell, rc::Rc};

use super::matrix::Matrix;

pub struct Autograd(AutogradNode);

impl Autograd {
    pub fn new(matrix: Matrix) -> Self {
        Self(Rc::new(
            RefCell::new(Computation { matrix, grad: None, op: None })
        ))
    }

    pub fn add(&self, rhs: &Autograd) -> Self {
        self.execute(Add { lhs: self.0.clone(), rhs: rhs.0.clone() })
    }

    pub fn dot(&self, rhs: &Autograd) -> Self {
        self.execute(Dot { lhs: self.0.clone(), rhs: rhs.0.clone() })
    }

    pub fn execute_with(&self, factory: impl OperationFactory) -> Self {
        self.execute_impl(factory.new(self.0.clone()))
    }

    fn execute(&self, op: impl Operation + 'static) -> Self {
        self.execute_impl(Box::new(op))
    }

    fn execute_impl(&self, op: Box<dyn Operation + 'static>) -> Self {
        Self(
            Rc::new(RefCell::new(Computation { 
                matrix: op.f(),
                grad: None,
                op: Some(op)
            }))
        )
    }

    pub fn backward(&mut self) {
        self.0.borrow_mut().backward_init();
    }

    pub fn into_grad(self) -> Option<Matrix> {
        Rc::into_inner(self.0).and_then(|c| c.into_inner().grad)
    }
}

pub type AutogradNode = Rc<RefCell<Computation>>;

pub struct Computation {
    pub matrix: Matrix,
    grad: Option<Matrix>,
    op: Option<Box<dyn Operation>>
}

impl Computation {
    fn backward_init(&mut self) {
        self.backward(Matrix::scalar(1.0));
    }

    pub fn backward(&mut self, grad: Matrix) {
        if let Some(op) = self.op.as_mut() {
            op.df(&grad);
        }
        match self.grad.as_mut() {
            Some(existing) => existing.add_assign(&grad),
            None => self.grad = Some(grad)
        };
    }
}

pub trait Operation {
    fn f(&self) -> Matrix;
    fn df(&mut self, grad: &Matrix);
}

struct Add {
    lhs: AutogradNode,
    rhs: AutogradNode
}

impl Operation for Add {    
    fn f(&self) -> Matrix {
        // add = lhs + rhs
        self.lhs.borrow().matrix.clone().add(&self.rhs.borrow().matrix)
    }

    fn df(&mut self, grad: &Matrix) {
        // df/dlhs = df/dmul
        // df/drhs = df/dmul
        self.lhs.borrow_mut().backward(grad.clone());
        self.rhs.borrow_mut().backward(grad.clone());
    }
}

struct Dot {
    lhs: AutogradNode,
    rhs: AutogradNode
}

impl Operation for Dot {
    fn f(&self) -> Matrix {
        // dot = lhs * rhs
        self.lhs.borrow().matrix.clone().dot(&self.rhs.borrow().matrix)
    }

    fn df(&mut self, grad: &Matrix) {
        // df/dlhs = df/ddot * rhs^T
        // df/drhs = lhs^T * df/ddot
        self.lhs.borrow_mut().backward(grad.clone().dot(&self.rhs.borrow().matrix.clone().transpose()));
        self.rhs.borrow_mut().backward(self.lhs.borrow().matrix.clone().transpose().dot(&grad));
    }
}

pub trait OperationFactory {
    fn new(self, this: AutogradNode) -> Box<dyn Operation>;
}