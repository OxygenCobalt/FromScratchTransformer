use std::io::{self, Write};

use crate::tensor::cpu2::{FillUninit, Tensor};

pub struct Activation {
    function: ActivationFn,
    dropout: f64,
    mask: Option<Tensor>,
    input_shape: Vec<usize>,
}

impl Activation {
    pub fn new(input_shape: Vec<usize>, function: ActivationFn, dropout: f64) -> Self {
        Self {
            function,
            dropout,
            mask: None,
            input_shape,
        }
    }

    pub fn prepare(&mut self, train: bool, batch: usize) {
        if !train || self.dropout <= 0.0 {
            self.mask = None;
            return;
        }
        let mut shape = self.input_shape.clone();
        shape.push(batch);
        let mut mask = Tensor::init(unsafe { FillUninit::new(shape) }).unwrap();
        let classes = mask.shape[0];
        let cols = mask.data.len() / classes;
        let flat_mask = mask.r_mut(&[classes, cols]).unwrap();
        let mask_data = flat_mask.tensor.data.as_mut_slice();
        let mut idx = 0;
        for _ in 0..classes {
            let x = if rand::random_range(0.0..1.0) < self.dropout {
                0.0
            } else {
                1.0
            };
            for _ in 0..cols {
                mask_data[idx] = x;
                idx += flat_mask.stride[1];
            }
            idx -= flat_mask.stride[1] * cols;
            idx += flat_mask.stride[0];
        }
        self.mask = Some(mask);
    }

    pub fn forward(&self, mut a_in: Tensor) -> Tensor {
        match self.function {
            ActivationFn::Sigmoid => self.apply_fn(&mut a_in, |x| 1.0 / (1.0 + (-x).exp())),
            ActivationFn::ReLU => self.apply_fn(&mut a_in, |x| if x > 0.0 { x } else { 0.0 }),
            ActivationFn::SiLU => self.apply_fn(&mut a_in, |x| x / (1.0 + (-x).exp())),
            ActivationFn::Softmax => {
                // what we want to do is be able to take the softmax
                // while preserving many-dimensional structures, so
                // we flatten down to [classes, cols] where cols is
                // just the flattened remaining dimensions that we
                // all factor into the norm.
                let classes = a_in.shape[0];
                let cols = a_in.data.len() / classes;
                let a_in_flat = a_in.r_mut(&[classes, cols]).unwrap();
                let mut idx = 0;
                let a_in_flat_data = a_in_flat.tensor.data.as_mut_slice();
                for _ in 0..cols {
                    let mut max = f64::NEG_INFINITY;
                    let mut max_idx = idx;
                    for _ in 0..classes {
                        let x = a_in_flat_data[max_idx];
                        if x > max {
                            max = x;
                        }
                        max_idx += a_in_flat.stride[0];
                    }
                    let mut norm = 0.0;
                    let mut pass1_idx = idx;
                    for _ in 0..classes {
                        a_in_flat_data[pass1_idx] = (a_in_flat_data[pass1_idx] - max).exp();
                        norm += a_in_flat_data[pass1_idx];
                        pass1_idx += a_in_flat.stride[0];
                    }
                    let mut pass2_idx = idx;
                    for _ in 0..classes {
                        a_in_flat_data[pass2_idx] /= norm;
                        if let Some(mask) = self.mask.as_ref() {
                            a_in_flat_data[pass2_idx] *= mask.data[pass2_idx];
                        }
                        pass2_idx += a_in_flat.stride[0];
                    }
                    idx += a_in_flat.stride[1];
                }
            }
            ActivationFn::Tanh => self.apply_fn(&mut a_in, |x| x.tanh()),
        }
        a_in
    }

    #[inline]
    fn apply_fn(&self, a_in: &mut Tensor, block: impl Fn(f64) -> f64) {
        if let Some(mask) = self.mask.as_ref() {
            a_in.data
                .as_mut_slice()
                .iter_mut()
                .zip(mask.data.as_slice())
                .for_each(|(v, m)| *v = block(*v) * *m);
            return;
        }
        a_in.data
            .as_mut_slice()
            .iter_mut()
            .for_each(|v| *v = block(*v));
    }

    pub fn backward(&mut self, c: f64, a_in: Tensor, mut grad: Tensor) -> Tensor {
        if let Some(mask) = self.mask.as_ref() {
            match self.function {
                ActivationFn::Sigmoid => a_in
                    .data
                    .iter()
                    .zip(grad.data.iter_mut())
                    .zip(mask.data.as_slice())
                    .for_each(|((v, g), m)| {
                        if *m == 0.0 {
                            *g = 0.0;
                        }
                        let sig = 1.0 / (1.0 + (-*v).exp());
                        *g = sig * (1.0 - sig) * (*g);
                    }),
                ActivationFn::ReLU => a_in
                    .data
                    .iter()
                    .zip(grad.data.iter_mut())
                    .zip(mask.data.as_slice())
                    .for_each(|((v, g), m)| {
                        if *m == 0.0 {
                            *g = 0.0;
                        }
                        *g = if *v > 0.0 { *g } else { 0.0 };
                    }),
                ActivationFn::SiLU => a_in
                    .data
                    .iter()
                    .zip(grad.data.iter_mut())
                    .zip(mask.data.as_slice())
                    .for_each(|((v, g), m)| {
                        if *m == 0.0 {
                            *g = 0.0;
                        }
                        let sig = 1.0 / (1.0 + (-*v).exp());
                        *g = (sig + *v * sig * (1.0 - sig)) * (*g);
                    }),
                ActivationFn::Softmax => {
                    todo!("Softmax backward pass is not implemented yet");
                }
                ActivationFn::Tanh => a_in
                    .data
                    .iter()
                    .zip(grad.data.iter_mut())
                    .zip(mask.data.as_slice())
                    .for_each(|((v, g), m)| {
                        if *m == 0.0 {
                            *g = 0.0;
                        }
                        let t = v.tanh();
                        *g = (1.0 - t * t) * (*g);
                    }),
            }
        } else {
            match self.function {
                ActivationFn::Sigmoid => {
                    a_in.data
                        .iter()
                        .zip(grad.data.iter_mut())
                        .for_each(|(v, g)| {
                            let sig = 1.0 / (1.0 + (-*v).exp());
                            *g = sig * (1.0 - sig) * (*g);
                        })
                }
                ActivationFn::ReLU => {
                    a_in.data
                        .iter()
                        .zip(grad.data.iter_mut())
                        .for_each(|(v, g)| {
                            *g = if *v > 0.0 { *g } else { 0.0 };
                        })
                }
                ActivationFn::SiLU => {
                    a_in.data
                        .iter()
                        .zip(grad.data.iter_mut())
                        .for_each(|(v, g)| {
                            let sig = 1.0 / (1.0 + (-*v).exp());
                            *g = (sig + *v * sig * (1.0 - sig)) * (*g);
                        })
                }
                ActivationFn::Softmax => {
                    todo!("Softmax backward pass is not implemented yet");
                }
                ActivationFn::Tanh => {
                    a_in.data
                        .iter()
                        .zip(grad.data.iter_mut())
                        .for_each(|(v, g)| {
                            let t = v.tanh();
                            *g = (1.0 - t * t) * (*g);
                        })
                }
            }
        }
        grad
    }

    pub fn read(read: &mut impl io::Read) -> io::Result<Self> {
        let mut shape_len_bytes = [0u8; 8];
        read.read_exact(&mut shape_len_bytes)?;
        let shape_len = usize::from_le_bytes(shape_len_bytes);
        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_bytes = [0u8; 8];
            read.read_exact(&mut dim_bytes)?;
            shape.push(usize::from_le_bytes(dim_bytes));
        }
        let mut id = [0u8; 8];
        read.read_exact(&mut id)?;
        let function = ActivationFn::from_id(&id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "invalid activation function signature",
            )
        })?;
        let mut dropout_bytes = [0u8; 8];
        read.read_exact(&mut dropout_bytes)?;
        let dropout = f64::from_le_bytes(dropout_bytes);
        Ok(Self::new(shape, function, dropout))
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(&self.input_shape.len().to_le_bytes())?;
        for dim in &self.input_shape {
            write.write_all(&dim.to_le_bytes())?;
        }
        write.write_all(self.function.id())?;
        write.write_all(&self.dropout.to_le_bytes())?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum ActivationFn {
    Sigmoid,
    ReLU,
    SiLU,
    Softmax,
    Tanh,
}

impl ActivationFn {
    pub fn id(&self) -> &'static [u8; 8] {
        match self {
            Self::Sigmoid => b"ActvSigm",
            Self::ReLU => b"ActvReLU",
            Self::SiLU => b"ActvSiLU",
            Self::Softmax => b"ActvSfmx",
            Self::Tanh => b"ActvTanh",
        }
    }

    pub fn from_id(id: &[u8; 8]) -> Option<Self> {
        match id {
            b"ActvSigm" => Some(Self::Sigmoid),
            b"ActvReLU" => Some(Self::ReLU),
            b"ActvSiLU" => Some(Self::SiLU),
            b"ActvSfmx" => Some(Self::Softmax),
            b"ActvTanh" => Some(Self::Tanh),
            _ => None,
        }
    }
}
