use std::io;

use rand_distr::{Distribution, Normal};

use crate::tensor::cpu2::{Fill, Generate, Tensor};

pub struct Embeddings {
    c: Tensor,
}

impl Embeddings {
    pub fn new(size: usize, vocab: usize) -> Self {
        let xavier = Normal::new(0.0, 1.0 / (size as f64).sqrt()).unwrap();
        Self {
            c: Tensor::init(Generate {
                shape: vec![size, vocab],
                with: || xavier.sample(&mut rand::rng()),
            })
            .unwrap(),
        }
    }

    pub fn forward(&self, a_in: Tensor) -> Tensor {
        let ctx = a_in.shape[0];
        let batch = a_in.shape[1];
        let dim = self.c.shape[0];
        let mut new_shape = vec![dim, ctx, batch];
        let mut new = Tensor::init(Fill {
            shape: new_shape,
            with: 0.0,
        })
        .unwrap();
        let mut a_in_idx = 0;
        let mut new_idx = 0;
        for b in 0..batch {
            for c in 0..ctx {
                let tok = a_in.data[a_in_idx] as usize;
                let mut emb_idx = self.c.stride[1] * tok;
                for n in 0..dim {
                    new.data[new_idx] = self.c.data[emb_idx];
                    emb_idx += self.c.stride[0];
                    new_idx += new.stride[0];
                }
                a_in_idx += a_in.stride[0];
                new_idx -= new.stride[0] * new.shape[0];
                new_idx += new.stride[1];
            }
            a_in_idx -= a_in.stride[0] * a_in.shape[0];
            a_in_idx += a_in.stride[1];
            new_idx -= new.stride[1] * new.shape[1];
            new_idx += new.stride[2];
        }
        new
    }

    pub fn backward(&mut self, c: f64, a_in: Tensor, grad: Tensor) -> Tensor {
        let ctx = a_in.shape[0];
        let batch = *a_in.shape.get(1).unwrap_or(&1);
        let dim = self.c.shape[0];
        let mut a_in_idx = 0;
        let mut new_idx = 0;
        for _ in 0..batch {
            for _ in 0..ctx {
                let tok = a_in.data[a_in_idx] as usize;
                let mut emb_idx = self.c.stride[1] * tok;
                for _ in 0..dim {
                    self.c.data[emb_idx] -= c * grad.data[new_idx];
                    emb_idx += self.c.stride[0];
                    new_idx += grad.stride[0];
                }
                a_in_idx += a_in.stride[0];
                new_idx -= grad.stride[0] * grad.shape[0];
                new_idx += grad.stride[1];
            }
            a_in_idx -= a_in.stride[0] * a_in.shape[0];
            a_in_idx += *a_in.stride.get(1).unwrap_or(&0);
            new_idx -= grad.stride[1] * grad.shape[1];
            new_idx += grad.stride[2];
        }
        a_in
    }

    pub fn read(read: &mut impl io::Read) -> io::Result<Self> {
        let c = Tensor::read(read)?;
        Ok(Self { c })
    }

    pub fn write(&self, write: &mut impl io::Write) -> io::Result<()> {
        self.c.write(write)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_with_data(shape: Vec<usize>, values: &[f64]) -> Tensor {
        let mut tensor = Tensor::init(Fill { shape, with: 0.0 }).expect("tensor init");
        assert_eq!(
            tensor.data.len(),
            values.len(),
            "shape/data length mismatch"
        );
        tensor.data.as_mut_slice().clone_from_slice(values);
        tensor
    }

    fn emb_with_table(shape: Vec<usize>, values: &[f64]) -> Embeddings {
        Embeddings {
            c: tensor_with_data(shape, values),
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: actual {} vs expected {}",
            actual.len(),
            expected.len()
        );
        for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "mismatch at index {}: expected {}, got {} (diff {})",
                idx,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    #[test]
    fn forward_selects_embeddings_in_input_order() {
        let emb = emb_with_table(
            vec![2, 4],
            &[
                0.0, 1.0, 2.0, 3.0, //
                10.0, 11.0, 12.0, 13.0,
            ],
        );
        let input = tensor_with_data(vec![1, 3], &[2.0, 0.0, 3.0]);

        let out = emb.forward(input);

        assert_eq!(out.shape, vec![1, 3, 2]);
        let expected = vec![
            2.0, 12.0, // c=0: tok2
            0.0, 10.0, // c=1: tok0
            3.0, 13.0, // c=2: tok3
        ];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_handles_repeated_token_ids() {
        let emb = emb_with_table(
            vec![3, 3],
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0,
            ],
        );
        let input = tensor_with_data(vec![1, 4], &[1.0, 1.0, 0.0, 1.0]);

        let out = emb.forward(input);

        assert_eq!(out.shape, vec![1, 4, 3]);
        let expected = vec![
            2.0, 5.0, 8.0, // c=0: tok1
            2.0, 5.0, 8.0, // c=1: tok1
            1.0, 4.0, 7.0, // c=2: tok0
            2.0, 5.0, 8.0, // c=3: tok1
        ];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn backward_updates_only_selected_columns() {
        let mut emb = emb_with_table(
            vec![2, 4],
            &[
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0,
            ],
        );
        let input = tensor_with_data(vec![1, 3], &[2.0, 0.0, 3.0]);
        // grad shape [batch=1, ctx=3, dim=2]
        let grad = tensor_with_data(
            vec![1, 3, 2],
            &[
                0.5, -2.0,   // c=0: tok2
                -0.25, 0.75, // c=1: tok0
                1.0, 0.5,    // c=2: tok3
            ],
        );

        let lr = 0.1;
        let input_back = emb.backward(lr, input.clone(), grad);

        assert_eq!(input_back.shape, input.shape);
        assert_close(input_back.data.as_slice(), input.data.as_slice(), 1e-12);

        let expected = vec![
            1.0 - lr * -0.25,
            2.0,
            3.0 - lr * 0.5,
            4.0 - lr * 1.0,
            5.0 - lr * 0.75,
            6.0,
            7.0 - lr * -2.0,
            8.0 - lr * 0.5,
        ];
        assert_close(emb.c.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn backward_accumulates_updates_for_repeated_token_ids() {
        let mut emb = emb_with_table(
            vec![2, 3],
            &[
                10.0, 20.0, 30.0, //
                40.0, 50.0, 60.0,
            ],
        );
        let input = tensor_with_data(vec![1, 3], &[1.0, 1.0, 1.0]);
        // grad shape [batch=1, ctx=3, dim=2]
        let grad = tensor_with_data(
            vec![1, 3, 2],
            &[
                0.1, 1.0,   // c=0: tok1
                0.2, -3.0,  // c=1: tok1
                -0.4, 2.0,  // c=2: tok1
            ],
        );

        let lr = 0.5;
        emb.backward(lr, input, grad);

        let expected = vec![
            10.0,
            20.0 - lr * (0.1 + 0.2 - 0.4),
            30.0,
            40.0,
            50.0 - lr * (1.0 - 3.0 + 2.0),
            60.0,
        ];
        assert_close(emb.c.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn backward_with_zero_lr_leaves_table_unchanged() {
        let mut emb = emb_with_table(
            vec![2, 3],
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
        );
        let input = tensor_with_data(vec![1, 2], &[0.0, 2.0]);
        // grad shape [batch=1, ctx=2, dim=2]
        let grad = tensor_with_data(vec![1, 2, 2], &[1.0, 3.0, 2.0, 4.0]);
        let before = emb.c.data.clone();

        emb.backward(0.0, input, grad);

        assert_close(emb.c.data.as_slice(), before.as_slice(), 1e-12);
    }

    #[test]
    fn forward_batched_selects_embeddings_per_batch() {
        // table shape [dim=2, vocab=4]:
        //   tok0=[0,10], tok1=[1,11], tok2=[2,12], tok3=[3,13]
        let emb = emb_with_table(
            vec![2, 4],
            &[
                0.0, 1.0, 2.0, 3.0, //
                10.0, 11.0, 12.0, 13.0,
            ],
        );
        // input shape [batch=2, ctx=3]
        // batch 0 tokens: [2, 0, 3]
        // batch 1 tokens: [1, 2, 0]
        let input = tensor_with_data(vec![2, 3], &[2.0, 0.0, 3.0, 1.0, 2.0, 0.0]);

        let out = emb.forward(input);

        // output shape [batch=2, ctx=3, dim=2]
        // out[b, c, d] = emb_table[d, token(b,c)]
        assert_eq!(out.shape, vec![2, 3, 2]);
        #[rustfmt::skip]
        let expected = vec![
            // b=0
            2.0, 12.0,   // c=0: tok2
            0.0, 10.0,   // c=1: tok0
            3.0, 13.0,   // c=2: tok3
            // b=1
            1.0, 11.0,   // c=0: tok1
            2.0, 12.0,   // c=1: tok2
            0.0, 10.0,   // c=2: tok0
        ];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn forward_batched_with_shared_tokens_across_batches() {
        // table shape [dim=2, vocab=3]:
        //   tok0=[1,4], tok1=[2,5], tok2=[3,6]
        let emb = emb_with_table(
            vec![2, 3],
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
        );
        // input shape [batch=3, ctx=2]
        // batch 0: [0, 1], batch 1: [1, 0], batch 2: [2, 2]
        let input = tensor_with_data(vec![3, 2], &[0.0, 1.0, 1.0, 0.0, 2.0, 2.0]);

        let out = emb.forward(input);

        assert_eq!(out.shape, vec![3, 2, 2]);
        #[rustfmt::skip]
        let expected = vec![
            // b=0: tokens [0, 1]
            1.0, 4.0, // c=0: tok0
            2.0, 5.0, // c=1: tok1
            // b=1: tokens [1, 0]
            2.0, 5.0, // c=0: tok1
            1.0, 4.0, // c=1: tok0
            // b=2: tokens [2, 2]
            3.0, 6.0, // c=0: tok2
            3.0, 6.0, // c=1: tok2
        ];
        assert_close(out.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn backward_batched_updates_embedding_table() {
        // table shape [dim=2, vocab=3]
        let mut emb = emb_with_table(
            vec![2, 3],
            &[
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0,
            ],
        );
        // input shape [batch=2, ctx=2]
        // batch 0: [0, 2], batch 1: [1, 0]
        let input = tensor_with_data(vec![2, 2], &[0.0, 2.0, 1.0, 0.0]);
        // grad shape [batch=2, ctx=2, dim=2]
        let grad = tensor_with_data(
            vec![2, 2, 2],
            &[
                0.1, 0.5, // b=0, c=0: tok0
                0.3, 0.7, // b=0, c=1: tok2
                0.2, 0.6, // b=1, c=0: tok1
                0.4, 0.8, // b=1, c=1: tok0
            ],
        );

        let lr = 1.0;
        let input_back = emb.backward(lr, input.clone(), grad);

        // backward should pass input through
        assert_eq!(input_back.shape, input.shape);
        assert_close(input_back.data.as_slice(), input.data.as_slice(), 1e-12);

        // Token usage:
        //   tok0: (b=0,c=0) and (b=1,c=1) → accumulated
        //   tok1: (b=1,c=0)
        //   tok2: (b=0,c=1)
        //
        // emb[0,0] -= lr * (g[b0,c0,d0] + g[b1,c1,d0]) = 1.0 * (0.1 + 0.4)
        // emb[0,1] -= lr * g[b1,c0,d0]                  = 1.0 * 0.2
        // emb[0,2] -= lr * g[b0,c1,d0]                  = 1.0 * 0.3
        // emb[1,0] -= lr * (g[b0,c0,d1] + g[b1,c1,d1]) = 1.0 * (0.5 + 0.8)
        // emb[1,1] -= lr * g[b1,c0,d1]                  = 1.0 * 0.6
        // emb[1,2] -= lr * g[b0,c1,d1]                  = 1.0 * 0.7
        let expected = vec![
            1.0 - 0.5,
            2.0 - 0.2,
            3.0 - 0.3, //
            4.0 - 1.3,
            5.0 - 0.6,
            6.0 - 0.7,
        ];
        assert_close(emb.c.data.as_slice(), &expected, 1e-12);
    }

    #[test]
    fn backward_batched_accumulates_across_all_positions_and_batches() {
        // table shape [dim=2, vocab=2]
        let mut emb = emb_with_table(
            vec![2, 2],
            &[
                10.0, 20.0, //
                30.0, 40.0,
            ],
        );
        // input shape [batch=2, ctx=2], every position uses token 0
        let input = tensor_with_data(vec![2, 2], &[0.0, 0.0, 0.0, 0.0]);
        // grad shape [batch=2, ctx=2, dim=2]
        let grad = tensor_with_data(
            vec![2, 2, 2],
            &[
                1.0, 5.0, // b=0, c=0
                3.0, 7.0, // b=0, c=1
                2.0, 6.0, // b=1, c=0
                4.0, 8.0, // b=1, c=1
            ],
        );

        let lr = 0.1;
        emb.backward(lr, input, grad);

        // tok0 used at all 4 positions → sum all grads per dim
        // emb[0,0] -= 0.1 * (1+3+2+4) = 0.1 * 10 = 1.0
        // emb[1,0] -= 0.1 * (5+7+6+8) = 0.1 * 26 = 2.6
        // tok1 never used → unchanged
        let expected = vec![
            10.0 - 1.0,
            20.0, //
            30.0 - 2.6,
            40.0,
        ];
        assert_close(emb.c.data.as_slice(), &expected, 1e-12);
    }
}
