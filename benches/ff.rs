#![feature(test, portable_simd)]
extern crate test;

use rand::Rng;
use test::{black_box, Bencher};

use nn::{
    ml::{activation::Activation, layers::ff::FeedForward},
    tensor::cpu2::{CPUTensor, Fill},
};

fn random_tensor(shape: Vec<usize>) -> CPUTensor<'static> {
    let mut tensor = CPUTensor::init(Fill { shape, with: 0.0 }).expect("tensor init");
    let mut rng = rand::rng();
    for v in tensor.data.to_mut().iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    tensor
}

fn tensor_from_data(shape: &[usize], data: &[f64]) -> CPUTensor<'static> {
    let mut tensor = CPUTensor::init(Fill {
        shape: shape.to_vec(),
        with: 0.0,
    })
    .expect("tensor init");
    tensor.data.to_mut().clone_from_slice(data);
    tensor
}

#[bench]
fn forward_512_to_8096_batch16(b: &mut Bencher) {
    let ff = FeedForward::new(vec![512], 8_096, Activation::ReLU);
    let input = random_tensor(vec![512, 16]);
    let input_shape = input.shape.clone();
    let input_buf = input.data.clone();
    b.iter(|| {
        let in_t = tensor_from_data(&input_shape, input_buf.as_ref());
        let out = ff.forward(in_t);
        black_box(out);
    });
}

fn backward_512_to_8096_impl_batch16<'a>(ff: &mut FeedForward<CPUTensor<'a>>, input_shape: &[usize], input_buf: &[f64], grad_shape: &[usize], grad_buf: &[f64], lr: f64) -> CPUTensor<'a> {
    let in_t = tensor_from_data(&input_shape, input_buf.as_ref());
    let grad_t = tensor_from_data(&grad_shape, grad_buf.as_ref());
    ff.backward(lr, in_t, grad_t)
}

#[bench]
fn backward_512_to_8096_batch16(b: &mut Bencher) {
    let mut ff = FeedForward::new(vec![512], 8_096, Activation::ReLU);
    let input = random_tensor(vec![512, 16]);
    let grad = random_tensor(vec![8_096, 16]);
    let input_shape = input.shape.clone();
    let input_buf = input.data.clone();
    let grad_shape = grad.shape.clone();
    let grad_buf = grad.data.clone();
    let lr = 1e-3;
    b.iter(|| {
        let out = backward_512_to_8096_impl_batch16(&mut ff, &input_shape, &input_buf, &grad_shape, &grad_buf, lr);
        black_box(out);
    });
}
