#![feature(test, portable_simd)]
extern crate test;

use rand::Rng;
use test::{black_box, Bencher};

use nn::{
    ml::{activation::Activation, nn::FeedForward as BaselineFeedForward},
    tensor::{
        Autograd, DifferentiableTensor, Fill as CpuFill, Tensor, TensorMut,
        cpu::CPUTensor as CpuTensor,
    },
};

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

fn bench_nn_backward(b: &mut Bencher, features: usize, neurons: usize, batch: usize) {
    let mut ff = BaselineFeedForward::new(vec![features], neurons, Activation::ReLU);
    let input = random_cpu_tensor(vec![features, batch]);
    let grad = random_cpu_tensor(vec![neurons, batch]);
    let input_shape = input.shape().to_vec();
    let input_buf: Vec<f64> = input.iter().cloned().collect();
    let grad_shape = grad.shape().to_vec();
    let grad_buf: Vec<f64> = grad.iter().cloned().collect();
    let lr = 1e-3;
    b.iter(|| {
        let out = nn_backward_impl(
            &mut ff,
            &input_shape,
            &input_buf,
            &grad_shape,
            &grad_buf,
            lr,
        );
        black_box(out);
    });
}

fn bench_nn_forward(b: &mut Bencher, features: usize, neurons: usize, batch: usize) {
    let ff = BaselineFeedForward::new(vec![features], neurons, Activation::ReLU);
    let input = random_cpu_tensor(vec![features, batch]);
    let input_shape = input.shape().to_vec();
    let input_buf: Vec<f64> = input.iter().cloned().collect();
    b.iter(|| {
        let in_t = cpu_tensor_from_data(&input_shape, &input_buf);
        let out = ff.forward(in_t);
        black_box(out);
    });
}

#[bench]
fn nn_backward_512_to_8096_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 512, 8_096, 1);
}

#[bench]
fn nn_backward_512_to_8096_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 512, 8_096, 8);
}

#[bench]
fn nn_backward_512_to_8096_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 512, 8_096, 10);
}

#[bench]
fn nn_backward_512_to_8096_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 512, 8_096, 16);
}

#[bench]
fn nn_backward_512_to_8096_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 512, 8_096, 32);
}

#[bench]
fn nn_backward_512_to_8096_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 512, 8_096, 64);
}

#[bench]
fn nn_backward_512_to_8096_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 512, 8_096, 100);
}

#[bench]
fn nn_backward_256_to_4096_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 256, 4_096, 1);
}

#[bench]
fn nn_backward_256_to_4096_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 256, 4_096, 8);
}

#[bench]
fn nn_backward_256_to_4096_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 256, 4_096, 10);
}

#[bench]
fn nn_backward_256_to_4096_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 256, 4_096, 16);
}

#[bench]
fn nn_backward_256_to_4096_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 256, 4_096, 32);
}

#[bench]
fn nn_backward_256_to_4096_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 256, 4_096, 64);
}

#[bench]
fn nn_backward_256_to_4096_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 256, 4_096, 100);
}

#[bench]
fn nn_backward_128_to_1024_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 128, 1_024, 1);
}

#[bench]
fn nn_backward_128_to_1024_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 128, 1_024, 8);
}

#[bench]
fn nn_backward_128_to_1024_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 128, 1_024, 10);
}

#[bench]
fn nn_backward_128_to_1024_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 128, 1_024, 16);
}

#[bench]
fn nn_backward_128_to_1024_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 128, 1_024, 32);
}

#[bench]
fn nn_backward_128_to_1024_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 128, 1_024, 64);
}

#[bench]
fn nn_backward_128_to_1024_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 128, 1_024, 100);
}

#[bench]
fn nn_backward_64_to_256_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 64, 256, 1);
}

#[bench]
fn nn_backward_64_to_256_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 64, 256, 8);
}

#[bench]
fn nn_backward_64_to_256_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 64, 256, 10);
}

#[bench]
fn nn_backward_64_to_256_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 64, 256, 16);
}

#[bench]
fn nn_backward_64_to_256_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 64, 256, 32);
}

#[bench]
fn nn_backward_64_to_256_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 64, 256, 64);
}

#[bench]
fn nn_backward_64_to_256_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 64, 256, 100);
}

#[bench]
fn nn_backward_8096_to_512_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 8_096, 512, 1);
}

#[bench]
fn nn_backward_8096_to_512_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 8_096, 512, 8);
}

#[bench]
fn nn_backward_8096_to_512_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 8_096, 512, 10);
}

#[bench]
fn nn_backward_8096_to_512_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 8_096, 512, 16);
}

#[bench]
fn nn_backward_8096_to_512_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 8_096, 512, 32);
}

#[bench]
fn nn_backward_8096_to_512_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 8_096, 512, 64);
}

#[bench]
fn nn_backward_8096_to_512_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 8_096, 512, 100);
}

#[bench]
fn nn_backward_4096_to_256_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 4_096, 256, 1);
}

#[bench]
fn nn_backward_4096_to_256_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 4_096, 256, 8);
}

#[bench]
fn nn_backward_4096_to_256_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 4_096, 256, 10);
}

#[bench]
fn nn_backward_4096_to_256_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 4_096, 256, 16);
}

#[bench]
fn nn_backward_4096_to_256_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 4_096, 256, 32);
}

#[bench]
fn nn_backward_4096_to_256_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 4_096, 256, 64);
}

#[bench]
fn nn_backward_4096_to_256_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 4_096, 256, 100);
}

#[bench]
fn nn_backward_1024_to_128_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 1_024, 128, 1);
}

#[bench]
fn nn_backward_1024_to_128_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 1_024, 128, 8);
}

#[bench]
fn nn_backward_1024_to_128_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 1_024, 128, 10);
}

#[bench]
fn nn_backward_1024_to_128_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 1_024, 128, 16);
}

#[bench]
fn nn_backward_1024_to_128_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 1_024, 128, 32);
}

#[bench]
fn nn_backward_1024_to_128_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 1_024, 128, 64);
}

#[bench]
fn nn_backward_1024_to_128_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 1_024, 128, 100);
}

#[bench]
fn nn_backward_256_to_64_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 256, 64, 1);
}

#[bench]
fn nn_backward_256_to_64_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 256, 64, 8);
}

#[bench]
fn nn_backward_256_to_64_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 256, 64, 10);
}

#[bench]
fn nn_backward_256_to_64_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 256, 64, 16);
}

#[bench]
fn nn_backward_256_to_64_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 256, 64, 32);
}

#[bench]
fn nn_backward_256_to_64_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 256, 64, 64);
}

#[bench]
fn nn_backward_256_to_64_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 256, 64, 100);
}

#[bench]
fn nn_backward_10_to_100_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 10, 100, 1);
}

#[bench]
fn nn_backward_10_to_100_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 10, 100, 8);
}

#[bench]
fn nn_backward_10_to_100_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 10, 100, 10);
}

#[bench]
fn nn_backward_10_to_100_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 10, 100, 16);
}

#[bench]
fn nn_backward_10_to_100_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 10, 100, 32);
}

#[bench]
fn nn_backward_10_to_100_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 10, 100, 64);
}

#[bench]
fn nn_backward_10_to_100_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 10, 100, 100);
}

#[bench]
fn nn_backward_100_to_10_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 100, 10, 1);
}

#[bench]
fn nn_backward_100_to_10_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 100, 10, 8);
}

#[bench]
fn nn_backward_100_to_10_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 100, 10, 10);
}

#[bench]
fn nn_backward_100_to_10_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 100, 10, 16);
}

#[bench]
fn nn_backward_100_to_10_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 100, 10, 32);
}

#[bench]
fn nn_backward_100_to_10_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 100, 10, 64);
}

#[bench]
fn nn_backward_100_to_10_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 100, 10, 100);
}

#[bench]
fn nn_backward_1000_to_10000_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 1_000, 10_000, 1);
}

#[bench]
fn nn_backward_1000_to_10000_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 1_000, 10_000, 8);
}

#[bench]
fn nn_backward_1000_to_10000_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 1_000, 10_000, 10);
}

#[bench]
fn nn_backward_1000_to_10000_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 1_000, 10_000, 16);
}

#[bench]
fn nn_backward_1000_to_10000_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 1_000, 10_000, 32);
}

#[bench]
fn nn_backward_1000_to_10000_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 1_000, 10_000, 64);
}

#[bench]
fn nn_backward_1000_to_10000_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 1_000, 10_000, 100);
}

#[bench]
fn nn_backward_10000_to_1000_batch1(b: &mut Bencher) {
    bench_nn_backward(b, 10_000, 1_000, 1);
}

#[bench]
fn nn_backward_10000_to_1000_batch8(b: &mut Bencher) {
    bench_nn_backward(b, 10_000, 1_000, 8);
}

#[bench]
fn nn_backward_10000_to_1000_batch10(b: &mut Bencher) {
    bench_nn_backward(b, 10_000, 1_000, 10);
}

#[bench]
fn nn_backward_10000_to_1000_batch16(b: &mut Bencher) {
    bench_nn_backward(b, 10_000, 1_000, 16);
}

#[bench]
fn nn_backward_10000_to_1000_batch32(b: &mut Bencher) {
    bench_nn_backward(b, 10_000, 1_000, 32);
}

#[bench]
fn nn_backward_10000_to_1000_batch64(b: &mut Bencher) {
    bench_nn_backward(b, 10_000, 1_000, 64);
}

#[bench]
fn nn_backward_10000_to_1000_batch100(b: &mut Bencher) {
    bench_nn_backward(b, 10_000, 1_000, 100);
}

#[bench]
fn nn_forward_512_to_8096_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 512, 8_096, 1);
}

#[bench]
fn nn_forward_512_to_8096_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 512, 8_096, 8);
}

#[bench]
fn nn_forward_512_to_8096_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 512, 8_096, 10);
}

#[bench]
fn nn_forward_512_to_8096_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 512, 8_096, 16);
}

#[bench]
fn nn_forward_512_to_8096_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 512, 8_096, 32);
}

#[bench]
fn nn_forward_512_to_8096_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 512, 8_096, 64);
}

#[bench]
fn nn_forward_512_to_8096_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 512, 8_096, 100);
}

#[bench]
fn nn_forward_256_to_4096_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 256, 4_096, 1);
}

#[bench]
fn nn_forward_256_to_4096_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 256, 4_096, 8);
}

#[bench]
fn nn_forward_256_to_4096_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 256, 4_096, 10);
}

#[bench]
fn nn_forward_256_to_4096_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 256, 4_096, 16);
}

#[bench]
fn nn_forward_256_to_4096_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 256, 4_096, 32);
}

#[bench]
fn nn_forward_256_to_4096_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 256, 4_096, 64);
}

#[bench]
fn nn_forward_256_to_4096_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 256, 4_096, 100);
}

#[bench]
fn nn_forward_128_to_1024_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 128, 1_024, 1);
}

#[bench]
fn nn_forward_128_to_1024_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 128, 1_024, 8);
}

#[bench]
fn nn_forward_128_to_1024_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 128, 1_024, 10);
}

#[bench]
fn nn_forward_128_to_1024_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 128, 1_024, 16);
}

#[bench]
fn nn_forward_128_to_1024_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 128, 1_024, 32);
}

#[bench]
fn nn_forward_128_to_1024_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 128, 1_024, 64);
}

#[bench]
fn nn_forward_128_to_1024_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 128, 1_024, 100);
}

#[bench]
fn nn_forward_64_to_256_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 64, 256, 1);
}

#[bench]
fn nn_forward_64_to_256_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 64, 256, 8);
}

#[bench]
fn nn_forward_64_to_256_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 64, 256, 10);
}

#[bench]
fn nn_forward_64_to_256_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 64, 256, 16);
}

#[bench]
fn nn_forward_64_to_256_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 64, 256, 32);
}

#[bench]
fn nn_forward_64_to_256_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 64, 256, 64);
}

#[bench]
fn nn_forward_64_to_256_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 64, 256, 100);
}

#[bench]
fn nn_forward_8096_to_512_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 8_096, 512, 1);
}

#[bench]
fn nn_forward_8096_to_512_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 8_096, 512, 8);
}

#[bench]
fn nn_forward_8096_to_512_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 8_096, 512, 10);
}

#[bench]
fn nn_forward_8096_to_512_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 8_096, 512, 16);
}

#[bench]
fn nn_forward_8096_to_512_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 8_096, 512, 32);
}

#[bench]
fn nn_forward_8096_to_512_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 8_096, 512, 64);
}

#[bench]
fn nn_forward_8096_to_512_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 8_096, 512, 100);
}

#[bench]
fn nn_forward_4096_to_256_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 4_096, 256, 1);
}

#[bench]
fn nn_forward_4096_to_256_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 4_096, 256, 8);
}

#[bench]
fn nn_forward_4096_to_256_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 4_096, 256, 10);
}

#[bench]
fn nn_forward_4096_to_256_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 4_096, 256, 16);
}

#[bench]
fn nn_forward_4096_to_256_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 4_096, 256, 32);
}

#[bench]
fn nn_forward_4096_to_256_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 4_096, 256, 64);
}

#[bench]
fn nn_forward_4096_to_256_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 4_096, 256, 100);
}

#[bench]
fn nn_forward_1024_to_128_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 1_024, 128, 1);
}

#[bench]
fn nn_forward_1024_to_128_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 1_024, 128, 8);
}

#[bench]
fn nn_forward_1024_to_128_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 1_024, 128, 10);
}

#[bench]
fn nn_forward_1024_to_128_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 1_024, 128, 16);
}

#[bench]
fn nn_forward_1024_to_128_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 1_024, 128, 32);
}

#[bench]
fn nn_forward_1024_to_128_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 1_024, 128, 64);
}

#[bench]
fn nn_forward_1024_to_128_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 1_024, 128, 100);
}

#[bench]
fn nn_forward_256_to_64_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 256, 64, 1);
}

#[bench]
fn nn_forward_256_to_64_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 256, 64, 8);
}

#[bench]
fn nn_forward_256_to_64_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 256, 64, 10);
}

#[bench]
fn nn_forward_256_to_64_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 256, 64, 16);
}

#[bench]
fn nn_forward_256_to_64_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 256, 64, 32);
}

#[bench]
fn nn_forward_256_to_64_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 256, 64, 64);
}

#[bench]
fn nn_forward_256_to_64_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 256, 64, 100);
}

// Power-of-10 shapes that do not align with SIMD lanes.
#[bench]
fn nn_forward_10_to_100_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 10, 100, 1);
}

#[bench]
fn nn_forward_10_to_100_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 10, 100, 8);
}

#[bench]
fn nn_forward_10_to_100_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 10, 100, 10);
}

#[bench]
fn nn_forward_10_to_100_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 10, 100, 16);
}

#[bench]
fn nn_forward_10_to_100_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 10, 100, 32);
}

#[bench]
fn nn_forward_10_to_100_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 10, 100, 64);
}

#[bench]
fn nn_forward_10_to_100_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 10, 100, 100);
}

#[bench]
fn nn_forward_100_to_10_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 100, 10, 1);
}

#[bench]
fn nn_forward_100_to_10_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 100, 10, 8);
}

#[bench]
fn nn_forward_100_to_10_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 100, 10, 10);
}

#[bench]
fn nn_forward_100_to_10_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 100, 10, 16);
}

#[bench]
fn nn_forward_100_to_10_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 100, 10, 32);
}

#[bench]
fn nn_forward_100_to_10_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 100, 10, 64);
}

#[bench]
fn nn_forward_100_to_10_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 100, 10, 100);
}

#[bench]
fn nn_forward_1000_to_10000_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 1_000, 10_000, 1);
}

#[bench]
fn nn_forward_1000_to_10000_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 1_000, 10_000, 8);
}

#[bench]
fn nn_forward_1000_to_10000_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 1_000, 10_000, 10);
}

#[bench]
fn nn_forward_1000_to_10000_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 1_000, 10_000, 16);
}

#[bench]
fn nn_forward_1000_to_10000_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 1_000, 10_000, 32);
}

#[bench]
fn nn_forward_1000_to_10000_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 1_000, 10_000, 64);
}

#[bench]
fn nn_forward_1000_to_10000_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 1_000, 10_000, 100);
}

#[bench]
fn nn_forward_10000_to_1000_batch1(b: &mut Bencher) {
    bench_nn_forward(b, 10_000, 1_000, 1);
}

#[bench]
fn nn_forward_10000_to_1000_batch8(b: &mut Bencher) {
    bench_nn_forward(b, 10_000, 1_000, 8);
}

#[bench]
fn nn_forward_10000_to_1000_batch10(b: &mut Bencher) {
    bench_nn_forward(b, 10_000, 1_000, 10);
}

#[bench]
fn nn_forward_10000_to_1000_batch16(b: &mut Bencher) {
    bench_nn_forward(b, 10_000, 1_000, 16);
}

#[bench]
fn nn_forward_10000_to_1000_batch32(b: &mut Bencher) {
    bench_nn_forward(b, 10_000, 1_000, 32);
}

#[bench]
fn nn_forward_10000_to_1000_batch64(b: &mut Bencher) {
    bench_nn_forward(b, 10_000, 1_000, 64);
}

#[bench]
fn nn_forward_10000_to_1000_batch100(b: &mut Bencher) {
    bench_nn_forward(b, 10_000, 1_000, 100);
}
