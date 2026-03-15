#![feature(test, portable_simd)]
extern crate test;

use rand::Rng;
use test::{Bencher, black_box};

use nn::{
    ml::axons::{
        act::{Activation, ActivationFn},
        ff::FeedForward as FusedFeedForward,
    },
    tensor::cpu2::{Fill as Cpu2Fill, Tensor as Cpu2Tensor},
};

fn random_cpu2_tensor(shape: Vec<usize>) -> Cpu2Tensor {
    let mut tensor = Cpu2Tensor::init(Cpu2Fill { shape, with: 0.0 }).expect("tensor init");
    let mut rng = rand::rng();
    for v in tensor.data.iter_mut() {
        *v = rng.random::<f64>() * 2.0 - 1.0;
    }
    tensor
}

fn cpu2_tensor_from_data(shape: &[usize], data: &[f64]) -> Cpu2Tensor {
    let mut tensor = Cpu2Tensor::init(Cpu2Fill {
        shape: shape.to_vec(),
        with: 0.0,
    })
    .expect("tensor init");
    tensor.data.clone_from_slice(data);
    tensor
}

fn bench_fused_forward(b: &mut Bencher, features: usize, neurons: usize, batch: usize) {
    let ff = FusedFeedForward::new(vec![features], neurons);
    let act = Activation::new(vec![neurons], ActivationFn::ReLU, 0.0);
    let input = random_cpu2_tensor(vec![batch, features]);
    let input_shape = input.shape.clone();
    let input_buf = input.data.clone();
    b.iter(|| {
        let in_t = cpu2_tensor_from_data(&input_shape, input_buf.as_ref());
        let out = act.forward(ff.forward(in_t));
        black_box(out);
    });
}

#[bench]
fn fused_forward_512_to_8096_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 1);
}

#[bench]
fn fused_forward_512_to_8096_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 8);
}

#[bench]
fn fused_forward_512_to_8096_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 10);
}

#[bench]
fn fused_forward_512_to_8096_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 16);
}

#[bench]
fn fused_forward_512_to_8096_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 32);
}

#[bench]
fn fused_forward_512_to_8096_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 64);
}

#[bench]
fn fused_forward_512_to_8096_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 512, 8_096, 100);
}

#[bench]
fn fused_forward_256_to_4096_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 256, 4_096, 1);
}

#[bench]
fn fused_forward_256_to_4096_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 256, 4_096, 8);
}

#[bench]
fn fused_forward_256_to_4096_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 256, 4_096, 10);
}

#[bench]
fn fused_forward_256_to_4096_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 256, 4_096, 16);
}

#[bench]
fn fused_forward_256_to_4096_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 256, 4_096, 32);
}

#[bench]
fn fused_forward_256_to_4096_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 256, 4_096, 64);
}

#[bench]
fn fused_forward_256_to_4096_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 256, 4_096, 100);
}

#[bench]
fn fused_forward_128_to_1024_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 128, 1_024, 1);
}

#[bench]
fn fused_forward_128_to_1024_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 128, 1_024, 8);
}

#[bench]
fn fused_forward_128_to_1024_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 128, 1_024, 10);
}

#[bench]
fn fused_forward_128_to_1024_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 128, 1_024, 16);
}

#[bench]
fn fused_forward_128_to_1024_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 128, 1_024, 32);
}

#[bench]
fn fused_forward_128_to_1024_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 128, 1_024, 64);
}

#[bench]
fn fused_forward_128_to_1024_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 128, 1_024, 100);
}

#[bench]
fn fused_forward_64_to_256_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 1);
}

#[bench]
fn fused_forward_64_to_256_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 8);
}

#[bench]
fn fused_forward_64_to_256_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 10);
}

#[bench]
fn fused_forward_64_to_256_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 16);
}

#[bench]
fn fused_forward_64_to_256_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 32);
}

#[bench]
fn fused_forward_64_to_256_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 64);
}

#[bench]
fn fused_forward_64_to_256_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 64, 256, 100);
}

#[bench]
fn fused_forward_8096_to_512_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 8_096, 512, 1);
}

#[bench]
fn fused_forward_8096_to_512_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 8_096, 512, 8);
}

#[bench]
fn fused_forward_8096_to_512_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 8_096, 512, 10);
}

#[bench]
fn fused_forward_8096_to_512_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 8_096, 512, 16);
}

#[bench]
fn fused_forward_8096_to_512_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 8_096, 512, 32);
}

#[bench]
fn fused_forward_8096_to_512_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 8_096, 512, 64);
}

#[bench]
fn fused_forward_8096_to_512_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 8_096, 512, 100);
}

#[bench]
fn fused_forward_4096_to_256_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 4_096, 256, 1);
}

#[bench]
fn fused_forward_4096_to_256_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 4_096, 256, 8);
}

#[bench]
fn fused_forward_4096_to_256_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 4_096, 256, 10);
}

#[bench]
fn fused_forward_4096_to_256_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 4_096, 256, 16);
}

#[bench]
fn fused_forward_4096_to_256_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 4_096, 256, 32);
}

#[bench]
fn fused_forward_4096_to_256_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 4_096, 256, 64);
}

#[bench]
fn fused_forward_4096_to_256_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 4_096, 256, 100);
}

#[bench]
fn fused_forward_1024_to_128_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 1_024, 128, 1);
}

#[bench]
fn fused_forward_1024_to_128_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 1_024, 128, 8);
}

#[bench]
fn fused_forward_1024_to_128_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 1_024, 128, 10);
}

#[bench]
fn fused_forward_1024_to_128_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 1_024, 128, 16);
}

#[bench]
fn fused_forward_1024_to_128_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 1_024, 128, 32);
}

#[bench]
fn fused_forward_1024_to_128_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 1_024, 128, 64);
}

#[bench]
fn fused_forward_1024_to_128_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 1_024, 128, 100);
}

#[bench]
fn fused_forward_256_to_64_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 256, 64, 1);
}

#[bench]
fn fused_forward_256_to_64_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 256, 64, 8);
}

#[bench]
fn fused_forward_256_to_64_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 256, 64, 10);
}

#[bench]
fn fused_forward_256_to_64_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 256, 64, 16);
}

#[bench]
fn fused_forward_256_to_64_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 256, 64, 32);
}

#[bench]
fn fused_forward_256_to_64_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 256, 64, 64);
}

#[bench]
fn fused_forward_256_to_64_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 256, 64, 100);
}

// Power-of-10 shapes that do not align with SIMD lanes.
#[bench]
fn fused_forward_10_to_100_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 10, 100, 1);
}

#[bench]
fn fused_forward_10_to_100_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 10, 100, 8);
}

#[bench]
fn fused_forward_10_to_100_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 10, 100, 10);
}

#[bench]
fn fused_forward_10_to_100_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 10, 100, 16);
}

#[bench]
fn fused_forward_10_to_100_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 10, 100, 32);
}

#[bench]
fn fused_forward_10_to_100_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 10, 100, 64);
}

#[bench]
fn fused_forward_10_to_100_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 10, 100, 100);
}

#[bench]
fn fused_forward_100_to_10_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 100, 10, 1);
}

#[bench]
fn fused_forward_100_to_10_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 100, 10, 8);
}

#[bench]
fn fused_forward_100_to_10_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 100, 10, 10);
}

#[bench]
fn fused_forward_100_to_10_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 100, 10, 16);
}

#[bench]
fn fused_forward_100_to_10_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 100, 10, 32);
}

#[bench]
fn fused_forward_100_to_10_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 100, 10, 64);
}

#[bench]
fn fused_forward_100_to_10_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 100, 10, 100);
}

#[bench]
fn fused_forward_1000_to_10000_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 1_000, 10_000, 1);
}

#[bench]
fn fused_forward_1000_to_10000_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 1_000, 10_000, 8);
}

#[bench]
fn fused_forward_1000_to_10000_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 1_000, 10_000, 10);
}

#[bench]
fn fused_forward_1000_to_10000_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 1_000, 10_000, 16);
}

#[bench]
fn fused_forward_1000_to_10000_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 1_000, 10_000, 32);
}

#[bench]
fn fused_forward_1000_to_10000_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 1_000, 10_000, 64);
}

#[bench]
fn fused_forward_1000_to_10000_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 1_000, 10_000, 100);
}

#[bench]
fn fused_forward_10000_to_1000_batch1(b: &mut Bencher) {
    bench_fused_forward(b, 10_000, 1_000, 1);
}

#[bench]
fn fused_forward_10000_to_1000_batch8(b: &mut Bencher) {
    bench_fused_forward(b, 10_000, 1_000, 8);
}

#[bench]
fn fused_forward_10000_to_1000_batch10(b: &mut Bencher) {
    bench_fused_forward(b, 10_000, 1_000, 10);
}

#[bench]
fn fused_forward_10000_to_1000_batch16(b: &mut Bencher) {
    bench_fused_forward(b, 10_000, 1_000, 16);
}

#[bench]
fn fused_forward_10000_to_1000_batch32(b: &mut Bencher) {
    bench_fused_forward(b, 10_000, 1_000, 32);
}

#[bench]
fn fused_forward_10000_to_1000_batch64(b: &mut Bencher) {
    bench_fused_forward(b, 10_000, 1_000, 64);
}

#[bench]
fn fused_forward_10000_to_1000_batch100(b: &mut Bencher) {
    bench_fused_forward(b, 10_000, 1_000, 100);
}

fn fused_backward_impl(
    ff: &mut FusedFeedForward,
    act: &mut Activation,
    input_shape: &[usize],
    input_buf: &[f64],
    ff_out_shape: &[usize],
    ff_out_buf: &[f64],
    grad_shape: &[usize],
    grad_buf: &[f64],
    lr: f64,
) -> Cpu2Tensor {
    let in_t = cpu2_tensor_from_data(input_shape, input_buf.as_ref());
    let ff_out = cpu2_tensor_from_data(ff_out_shape, ff_out_buf.as_ref());
    let grad_t = cpu2_tensor_from_data(grad_shape, grad_buf.as_ref());
    let ff_grad = act.backward(lr, ff_out, grad_t);
    ff.backward(lr, in_t, ff_grad)
}

fn bench_fused_backward(b: &mut Bencher, features: usize, neurons: usize, batch: usize) {
    let mut ff = FusedFeedForward::new(vec![features], neurons);
    let mut act = Activation::new(vec![neurons], ActivationFn::ReLU, 0.0);
    let input = random_cpu2_tensor(vec![batch, features]);
    let ff_out = ff.forward(cpu2_tensor_from_data(&input.shape, input.data.as_ref()));
    let grad = random_cpu2_tensor(vec![batch, neurons]);
    let input_shape = input.shape.clone();
    let input_buf = input.data.clone();
    let ff_out_shape = ff_out.shape.clone();
    let ff_out_buf = ff_out.data.clone();
    let grad_shape = grad.shape.clone();
    let grad_buf = grad.data.clone();
    let lr = 1e-3;
    b.iter(|| {
        let out = fused_backward_impl(
            &mut ff,
            &mut act,
            &input_shape,
            &input_buf,
            &ff_out_shape,
            &ff_out_buf,
            &grad_shape,
            &grad_buf,
            lr,
        );
        black_box(out);
    });
}

#[bench]
fn fused_backward_512_to_8096_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 1);
}

#[bench]
fn fused_backward_512_to_8096_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 8);
}

#[bench]
fn fused_backward_512_to_8096_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 10);
}

#[bench]
fn fused_backward_512_to_8096_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 16);
}

#[bench]
fn fused_backward_512_to_8096_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 32);
}

#[bench]
fn fused_backward_512_to_8096_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 64);
}

#[bench]
fn fused_backward_512_to_8096_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 512, 8_096, 100);
}

#[bench]
fn fused_backward_256_to_4096_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 256, 4_096, 1);
}

#[bench]
fn fused_backward_256_to_4096_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 256, 4_096, 8);
}

#[bench]
fn fused_backward_256_to_4096_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 256, 4_096, 10);
}

#[bench]
fn fused_backward_256_to_4096_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 256, 4_096, 16);
}

#[bench]
fn fused_backward_256_to_4096_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 256, 4_096, 32);
}

#[bench]
fn fused_backward_256_to_4096_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 256, 4_096, 64);
}

#[bench]
fn fused_backward_256_to_4096_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 256, 4_096, 100);
}

#[bench]
fn fused_backward_128_to_1024_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 128, 1_024, 1);
}

#[bench]
fn fused_backward_128_to_1024_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 128, 1_024, 8);
}

#[bench]
fn fused_backward_128_to_1024_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 128, 1_024, 10);
}

#[bench]
fn fused_backward_128_to_1024_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 128, 1_024, 16);
}

#[bench]
fn fused_backward_128_to_1024_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 128, 1_024, 32);
}

#[bench]
fn fused_backward_128_to_1024_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 128, 1_024, 64);
}

#[bench]
fn fused_backward_128_to_1024_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 128, 1_024, 100);
}

#[bench]
fn fused_backward_64_to_256_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 1);
}

#[bench]
fn fused_backward_64_to_256_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 8);
}

#[bench]
fn fused_backward_64_to_256_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 10);
}

#[bench]
fn fused_backward_64_to_256_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 16);
}

#[bench]
fn fused_backward_64_to_256_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 32);
}

#[bench]
fn fused_backward_64_to_256_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 64);
}

#[bench]
fn fused_backward_64_to_256_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 64, 256, 100);
}

#[bench]
fn fused_backward_8096_to_512_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 8_096, 512, 1);
}

#[bench]
fn fused_backward_8096_to_512_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 8_096, 512, 8);
}

#[bench]
fn fused_backward_8096_to_512_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 8_096, 512, 10);
}

#[bench]
fn fused_backward_8096_to_512_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 8_096, 512, 16);
}

#[bench]
fn fused_backward_8096_to_512_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 8_096, 512, 32);
}

#[bench]
fn fused_backward_8096_to_512_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 8_096, 512, 64);
}

#[bench]
fn fused_backward_8096_to_512_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 8_096, 512, 100);
}

#[bench]
fn fused_backward_4096_to_256_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 4_096, 256, 1);
}

#[bench]
fn fused_backward_4096_to_256_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 4_096, 256, 8);
}

#[bench]
fn fused_backward_4096_to_256_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 4_096, 256, 10);
}

#[bench]
fn fused_backward_4096_to_256_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 4_096, 256, 16);
}

#[bench]
fn fused_backward_4096_to_256_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 4_096, 256, 32);
}

#[bench]
fn fused_backward_4096_to_256_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 4_096, 256, 64);
}

#[bench]
fn fused_backward_4096_to_256_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 4_096, 256, 100);
}

#[bench]
fn fused_backward_1024_to_128_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 1_024, 128, 1);
}

#[bench]
fn fused_backward_1024_to_128_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 1_024, 128, 8);
}

#[bench]
fn fused_backward_1024_to_128_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 1_024, 128, 10);
}

#[bench]
fn fused_backward_1024_to_128_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 1_024, 128, 16);
}

#[bench]
fn fused_backward_1024_to_128_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 1_024, 128, 32);
}

#[bench]
fn fused_backward_1024_to_128_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 1_024, 128, 64);
}

#[bench]
fn fused_backward_1024_to_128_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 1_024, 128, 100);
}

#[bench]
fn fused_backward_256_to_64_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 256, 64, 1);
}

#[bench]
fn fused_backward_256_to_64_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 256, 64, 8);
}

#[bench]
fn fused_backward_256_to_64_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 256, 64, 10);
}

#[bench]
fn fused_backward_256_to_64_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 256, 64, 16);
}

#[bench]
fn fused_backward_256_to_64_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 256, 64, 32);
}

#[bench]
fn fused_backward_256_to_64_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 256, 64, 64);
}

#[bench]
fn fused_backward_256_to_64_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 256, 64, 100);
}

#[bench]
fn fused_backward_10_to_100_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 10, 100, 1);
}

#[bench]
fn fused_backward_10_to_100_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 10, 100, 8);
}

#[bench]
fn fused_backward_10_to_100_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 10, 100, 10);
}

#[bench]
fn fused_backward_10_to_100_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 10, 100, 16);
}

#[bench]
fn fused_backward_10_to_100_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 10, 100, 32);
}

#[bench]
fn fused_backward_10_to_100_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 10, 100, 64);
}

#[bench]
fn fused_backward_10_to_100_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 10, 100, 100);
}

#[bench]
fn fused_backward_100_to_10_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 100, 10, 1);
}

#[bench]
fn fused_backward_100_to_10_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 100, 10, 8);
}

#[bench]
fn fused_backward_100_to_10_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 100, 10, 10);
}

#[bench]
fn fused_backward_100_to_10_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 100, 10, 16);
}

#[bench]
fn fused_backward_100_to_10_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 100, 10, 32);
}

#[bench]
fn fused_backward_100_to_10_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 100, 10, 64);
}

#[bench]
fn fused_backward_100_to_10_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 100, 10, 100);
}

#[bench]
fn fused_backward_1000_to_10000_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 1_000, 10_000, 1);
}

#[bench]
fn fused_backward_1000_to_10000_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 1_000, 10_000, 8);
}

#[bench]
fn fused_backward_1000_to_10000_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 1_000, 10_000, 10);
}

#[bench]
fn fused_backward_1000_to_10000_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 1_000, 10_000, 16);
}

#[bench]
fn fused_backward_1000_to_10000_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 1_000, 10_000, 32);
}

#[bench]
fn fused_backward_1000_to_10000_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 1_000, 10_000, 64);
}

#[bench]
fn fused_backward_1000_to_10000_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 1_000, 10_000, 100);
}

#[bench]
fn fused_backward_10000_to_1000_batch1(b: &mut Bencher) {
    bench_fused_backward(b, 10_000, 1_000, 1);
}

#[bench]
fn fused_backward_10000_to_1000_batch8(b: &mut Bencher) {
    bench_fused_backward(b, 10_000, 1_000, 8);
}

#[bench]
fn fused_backward_10000_to_1000_batch10(b: &mut Bencher) {
    bench_fused_backward(b, 10_000, 1_000, 10);
}

#[bench]
fn fused_backward_10000_to_1000_batch16(b: &mut Bencher) {
    bench_fused_backward(b, 10_000, 1_000, 16);
}

#[bench]
fn fused_backward_10000_to_1000_batch32(b: &mut Bencher) {
    bench_fused_backward(b, 10_000, 1_000, 32);
}

#[bench]
fn fused_backward_10000_to_1000_batch64(b: &mut Bencher) {
    bench_fused_backward(b, 10_000, 1_000, 64);
}

#[bench]
fn fused_backward_10000_to_1000_batch100(b: &mut Bencher) {
    bench_fused_backward(b, 10_000, 1_000, 100);
}
