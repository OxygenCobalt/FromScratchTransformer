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
        let mut new_shape = self.c.shape.clone();
        new_shape[1] = a_in.shape[0];
        let mut new = Tensor::init(Fill {
            shape: new_shape,
            with: 0.0,
        })
        .unwrap();
        let mut new_point = vec![0; new.shape.len()];
        let mut new_idx = 0;
        while new_point[1] < a_in.shape[0] {
            let idx = a_in.data[new_point[1]] as usize;
            let mut old_idx = idx * self.c.stride[1];
            'iterate: loop {
                new.data[new_idx] = self.c.data[old_idx];
                for i in 0..new.shape.len() {
                    if i == 1 {
                        continue;
                    }
                    if new_point[i] == new.shape[i] - 1 {
                        if i == 1 {
                            break 'iterate;
                        }
                        new_idx -= new.stride[i] * new_point[i];
                        old_idx += self.c.stride[i] * new_point[i];
                        new_point[i] = 0;
                    } else {
                        new_idx += new.stride[i];
                        old_idx += self.c.stride[i];
                        new_point[i] += 1;
                        continue 'iterate;
                    }
                }
                break;
            }
            new_idx += new.stride[1];
            new_point[1] += 1;
        }
        new
    }

    pub fn backward(&mut self, c: f64, a_in: Tensor, grad: Tensor) -> Tensor {
        let mut grad_point = vec![0; grad.shape.len()];
        let mut grad_idx = 0;
        while grad_point[1] < a_in.shape[0] {
            let idx = a_in.data[grad_point[1]] as usize;
            let mut t_grad_idx = idx * self.c.stride[1];
            'iterate: loop {
                self.c.data[t_grad_idx] -= c * grad.data[grad_idx];
                for i in 0..grad.shape.len() {
                    if i == 1 {
                        continue;
                    }
                    if grad_point[i] == grad.shape[i] - 1 {
                        grad_idx -= grad.stride[i] * grad_point[i];
                        t_grad_idx -= self.c.stride[i] * grad_point[i];
                        grad_point[i] = 0;
                    } else {
                        grad_idx += grad.stride[i];
                        t_grad_idx += self.c.stride[i];
                        grad_point[i] += 1;
                        continue 'iterate;
                    }
                }
                break;
            }
            grad_idx += grad.stride[1];
            grad_point[1] += 1;
        }
        a_in
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
        let input = tensor_with_data(vec![3], &[2.0, 0.0, 3.0]);

        let out = emb.forward(input);

        assert_eq!(out.shape, vec![2, 3]);
        let expected = vec![
            2.0, 0.0, 3.0, //
            12.0, 10.0, 13.0,
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
        let input = tensor_with_data(vec![4], &[1.0, 1.0, 0.0, 1.0]);

        let out = emb.forward(input);

        assert_eq!(out.shape, vec![3, 4]);
        let expected = vec![
            2.0, 2.0, 1.0, 2.0, //
            5.0, 5.0, 4.0, 5.0, //
            8.0, 8.0, 7.0, 8.0,
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
        let input = tensor_with_data(vec![3], &[2.0, 0.0, 3.0]);
        let grad = tensor_with_data(
            vec![2, 3],
            &[
                0.5, -0.25, 1.0, //
                -2.0, 0.75, 0.5,
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
        let input = tensor_with_data(vec![3], &[1.0, 1.0, 1.0]);
        let grad = tensor_with_data(
            vec![2, 3],
            &[
                0.1, 0.2, -0.4, //
                1.0, -3.0, 2.0,
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
        let input = tensor_with_data(vec![2], &[0.0, 2.0]);
        let grad = tensor_with_data(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let before = emb.c.data.clone();

        emb.backward(0.0, input, grad);

        assert_close(emb.c.data.as_slice(), before.as_slice(), 1e-12);
    }
}
