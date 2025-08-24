use crate::matrix::{Dimensions, Matrix};

mod matrix;

fn main() {
    let mut a = Matrix::inc(Dimensions::square(10000));
    let mut b = Matrix::inc(Dimensions::square(10000));
    // println!("then");

    // for i in 0..=100 {
    //     let mut mat = Matrix::inc(Dimensions { n: 10, m: 10 });
    //     let transposed = mat.clone().transpose();
    //     mat.transpose_assign(i);
    //     println!("i={}, correct? {:?}", i, mat == transposed);
    // }
    // println!("transpose");
    // println!("{:?}", mat);
    // mat += Matrix::identity(10000);
}
