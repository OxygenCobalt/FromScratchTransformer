

use std::{cell::RefCell, rc::Rc};

use crate::{matrix::Matrix};

pub struct Autograd(Rc<RefCell<ComputationNode>>);

impl Autograd {
    pub fn new(matrix: Matrix) -> Self {
        Self(Rc::new(
            RefCell::new(ComputationNode { matrix, grad: None, parent: None })
        ))
    }

    pub fn add(&self, rhs: &Autograd) -> Self {
        self.execute(Add { lhs: self.0.clone(), rhs: rhs.0.clone() })
    }

    pub fn sub(&self, rhs: &Autograd) -> Self {
        self.execute(Sub { lhs: self.0.clone(), rhs: rhs.0.clone() })
    }

    pub fn mul(&self, rhs: &Autograd) -> Self {
        self.execute(Mul { lhs: self.0.clone(), rhs: rhs.0.clone() })
    }

    pub fn dot(&self, rhs: &Autograd) -> Self {
        self.execute(Dot { lhs: self.0.clone(), rhs: rhs.0.clone() })
    }

    pub fn execute_with<F: Function + 'static>(&self, factory: impl Fn(Rc<RefCell<ComputationNode>>) -> F) -> Self {
        let func = factory(self.0.clone());
        self.execute(func)
    }

    fn execute(&self, function: impl Function + 'static) -> Self {
        Self(
            Rc::new(RefCell::new(ComputationNode { 
                matrix:  function.f(),
                grad: None,
                parent: Some(Box::new(function))
            }))
        )
    }

    pub fn backward(&mut self) {
        self.0.borrow_mut().backward_init();
    }

    pub fn grad(&self) -> Option<Matrix> {
        self.0.borrow().grad.clone()
    }

    pub fn matrix(&self) -> Matrix {
        self.0.borrow().matrix.clone()
    }
}

pub struct ComputationNode {
    pub matrix: Matrix,
    grad: Option<Matrix>,
    parent: Option<Box<dyn Function>>
}

impl ComputationNode {
    fn backward_init(&mut self) {
        self.backward(Matrix::scalar(1.0, self.matrix.shape()));
    }
    pub fn backward(&mut self, grad: Matrix) {
        if let Some(parent) = self.parent.as_mut() {
            parent.df(&grad);
        }
        self.grad = Some(grad);
    }
}

pub trait Function {
    fn f(&self) -> Matrix;
    fn df(&mut self, grad: &Matrix);
}

struct Add {
    lhs: Rc<RefCell<ComputationNode>>,
    rhs: Rc<RefCell<ComputationNode>>
}

impl Function for Add {    
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

struct Sub {
    lhs: Rc<RefCell<ComputationNode>>,
    rhs: Rc<RefCell<ComputationNode>>
}

impl Function for Sub {
    fn f(&self) -> Matrix {
        // sub = lhs - rhs
        self.lhs.borrow().matrix.clone().add(&self.rhs.borrow().matrix)
    }

    fn df(&mut self, grad: &Matrix) {
        // df/dlhs = df/dmul
        // df/drhs = -df/dmul
        self.lhs.borrow_mut().backward(grad.clone());
        self.rhs.borrow_mut().backward(grad.clone().scale(-1.0));
    }
}


struct Mul {
    lhs: Rc<RefCell<ComputationNode>>,
    rhs: Rc<RefCell<ComputationNode>>
}

impl Function for Mul {
    fn f(&self) -> Matrix {
        // mul = lhs ⊙ rhs
        self.lhs.borrow().matrix.clone().mul(&self.rhs.borrow().matrix)
    }

    fn df(&mut self, grad: &Matrix) {
        // df/dlhs = rhs ⊙ df/dmul;
        // df/drhs = lhs ⊗ df/dmul
        self.lhs.borrow_mut().backward(self.rhs.borrow().matrix.clone().mul(grad));
        self.rhs.borrow_mut().backward(self.lhs.borrow().matrix.clone().mul(grad));
    }
}

struct Dot {
    lhs: Rc<RefCell<ComputationNode>>,
    rhs: Rc<RefCell<ComputationNode>>
}

impl Function for Dot {
    fn f(&self) -> Matrix {
        // dot = lhs * rhs
        self.lhs.borrow().matrix.clone().dot(&self.rhs.borrow().matrix)
    }

    fn df(&mut self, grad: &Matrix) {
        // df/dlhs = df/ddot ⊗ rhs   
        // df/drhs = lhs^T ⊗ df/ddot
        self.lhs.borrow_mut().backward(grad.clone().outer(&self.rhs.borrow().matrix));
        self.rhs.borrow_mut().backward(self.lhs.borrow().matrix.clone().transpose().dot(&grad));
    }
}
