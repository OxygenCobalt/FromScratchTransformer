#![feature(test, portable_simd)]
extern crate test;

use rand::Rng;
use test::{black_box, Bencher};

use nn::{
    ml::{
        activation::Activation,
        layers::ff::FeedForward as FusedFeedForward,
        nn::FeedForward as BaselineFeedForward,
    },
    tensor::{
        cpu::CPUTensor as CpuTensor,
        cpu2::{CPUTensor as Cpu2Tensor, Fill as Cpu2Fill},
        Autograd, DifferentiableTensor, Fill as CpuFill, Tensor, TensorMut,
    },
};

fn random_cpu2_tensor(shape: Vec<usize>) -> Cpu2Tensor<'static> {
    let mut tensor = Cpu2Tensor::init(Cpu2Fill { shape, with: 0.0 }).expect("tensor init");
    let mut rng = rand::rng();
    for v in tensor.data.to_mut().iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    tensor
}

fn cpu2_tensor_from_data(shape: &[usize], data: &[f64]) -> Cpu2Tensor<'static> {
    let mut tensor = Cpu2Tensor::init(Cpu2Fill {
        shape: shape.to_vec(),
        with: 0.0,
    })
    .expect("tensor init");
    tensor.data.to_mut().clone_from_slice(data);
    tensor
}

fn bench_fused_forward(b: &mut Bencher, features: usize, neurons: usize, batch: usize) {
    let ff = FusedFeedForward::new(vec![features], neurons, Activation::ReLU);
    let input = random_cpu2_tensor(vec![features, batch]);
    let input_shape = input.shape.clone();
    let input_buf = input.data.clone();
    b.iter(|| {
        let in_t = cpu2_tensor_from_data(&input_shape, input_buf.as_ref());
        let out = ff.forward(in_t);
        black_box(out);
    });
}

#[bench]
fn fused_forward_512_to_8096_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 16);
}

#[bench]
fn fused_forward_512_to_8096_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 1);
}

#[bench]
fn fused_forward_512_to_8096_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 64);
}

#[bench]
fn fused_forward_64_to_256_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 1);
}

#[bench]
fn fused_forward_64_to_256_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 16);
}

fn fused_backward_impl<'a>(
    ff: &mut FusedFeedForward<Cpu2Tensor<'a>>,
    input_shape: &[usize],
    input_buf: &[f64],
    activations_shape: &[usize],
    activations_buf: &[f64],
    grad_shape: &[usize],
    grad_buf: &[f64],
    lr: f64,
) -> Cpu2Tensor<'a> {
    let in_t = cpu2_tensor_from_data(input_shape, input_buf.as_ref());
    let activations_out = cpu2_tensor_from_data(activations_shape, activations_buf.as_ref());
    let grad_t = cpu2_tensor_from_data(grad_shape, grad_buf.as_ref());
    ff.backward(lr, in_t, activations_out, grad_t)
}

fn bench_fused_backward(b: &mut Bencher, features: usize, neurons: usize, batch: usize) {
    let mut ff = FusedFeedForward::new(vec![features], neurons, Activation::ReLU);
    let input = random_cpu2_tensor(vec![features, batch]);
    let grad = random_cpu2_tensor(vec![neurons, batch]);
    let activations_out = ff.forward(input.cloned_view());
    let input_shape = input.shape.clone();
    let input_buf = input.data.clone();
    let activations_shape = activations_out.shape.clone();
    let activations_buf = activations_out.data.clone();
    let grad_shape = grad.shape.clone();
    let grad_buf = grad.data.clone();
    let lr = 1e-3;
    b.iter(|| {
        let out = fused_backward_impl(
            &mut ff,
            &input_shape,
            &input_buf,
            &activations_shape,
            &activations_buf,
            &grad_shape,
            &grad_buf,
            lr,
        );
        black_box(out);
    });
}

#[bench]
fn fused_backward_512_to_8096_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 16);
}

#[bench]
fn fused_backward_512_to_8096_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 1);
}

#[bench]
fn fused_backward_512_to_8096_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 64);
}

#[bench]
fn fused_backward_64_to_256_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 1);
}

#[bench]
fn fused_backward_64_to_256_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 16);
}

fn random_cpu_tensor(shape: Vec<usize>) -> CpuTensor {
    let mut tensor = CpuTensor::tensor(CpuFill { shape, with: 0.0 }).expect("tensor init");
    let mut rng = rand::rng();
    for v in tensor.iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    tensor
}

fn cpu_tensor_from_data(shape: &[usize], data: &[f64]) -> CpuTensor {
    let mut tensor = CpuTensor::tensor(CpuFill {
        shape: shape.to_vec(),
        with: 0.0,
    })
    .expect("tensor init");
    for (dst, src) in tensor.iter_mut().zip(data.iter()) {
        *dst = *src;
    }
    tensor
}

fn nn_backward_impl(
    ff: &mut BaselineFeedForward<CpuTensor>,
    input_shape: &[usize],
    input_buf: &[f64],
    grad_shape: &[usize],
    grad_buf: &[f64],
    lr: f64,
) -> CpuTensor {
    let auto_ff = ff.autograd();
    let input = cpu_tensor_from_data(input_shape, input_buf).into_autograd();
    let input_handle = input.clone();
    let grad = cpu_tensor_from_data(grad_shape, grad_buf);
    let out = auto_ff.forward(input);
    out.backward_with_grad(grad);
    let grad_ff = auto_ff.into_grad().expect("ff grad");
    let input_grad = input_handle.into_grad().expect("input grad");
    ff.descend(lr, &grad_ff).expect("descend");
    input_grad
}

#[bench]
fn nn_backward_512_to_8096_batch16(b: &mut Bencher) {
    let mut ff = BaselineFeedForward::new(vec![512], 8_096, Activation::ReLU);
    let input = random_cpu_tensor(vec![512, 16]);
    let grad = random_cpu_tensor(vec![8_096, 16]);
    let input_shape = input.shape().to_vec();
    let input_buf: Vec<f64> = input.iter().cloned().collect();
    let grad_shape = grad.shape().to_vec();
    let grad_buf: Vec<f64> = grad.iter().cloned().collect();
    let lr = 1e-3;
    b.iter(|| {
        let out = nn_backward_impl(&mut ff, &input_shape, &input_buf, &grad_shape, &grad_buf, lr);
        black_box(out);
    });
}

#[bench]
fn nn_forward_512_to_8096_batch16(b: &mut Bencher) {
    let ff = BaselineFeedForward::new(vec![512], 8_096, Activation::ReLU);
    let input = random_cpu_tensor(vec![512, 16]);
    let input_shape = input.shape().to_vec();
    let input_buf: Vec<f64> = input.iter().cloned().collect();
    b.iter(|| {
        let in_t = cpu_tensor_from_data(&input_shape, &input_buf);
        let out = ff.forward(in_t);
        black_box(out);
    });
}
