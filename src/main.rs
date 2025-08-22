use crate::matrix::{Dimensions, Matrix};

mod matrix;

fn main() {
    let mut mat = Matrix::inc(Dimensions { n: 10000, m: 10000 }) + Matrix::identity(10000);
    // mat.transpose();
    // println!("{:?}", mat);
    // mat += Matrix::identity(10000);
}
