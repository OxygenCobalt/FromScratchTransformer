use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::dataset::{EagerExample, Example, Train};
use crate::tensor::cpu2::{Tensor, Tt};

use super::{activation::Activation, loss2::Loss, axons::{Axon, ff::FeedForward}};

pub struct NeuralNetwork<'a> {
    axons: Vec<Axon<'a>>
}

impl <'a> NeuralNetwork<'a> {
    pub fn test<'i>(&self, input: &Tensor<'i>) -> Tensor<'i> {
        let mut current = input.clone();
        for axon in &self.axons {
            current = axon.forward(current)
        }
        current
    }
}

impl <'a> NeuralNetwork<'a> {
    pub fn train<'t>(
        setup: &impl Setup,
        reporting: &impl Reporting,
        train: &Train<impl Example<crate::tensor::cpu::CPUTensor>>,
        hyperparams: &Hyperparams,
        loss: Loss,
    ) -> io::Result<Self> {
        println!(
            "{}: epochs = {} / batch size = {} / learning rate = {}",
            "train".cyan(),
            hyperparams.epochs,
            hyperparams.batch_size,
            hyperparams.learning_rate
        );
        let mut init = setup.setup()?;
        let start = init.at_epoch.unwrap_or(0);
        if start > hyperparams.epochs {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid setup, epoch exceeds training parameters",
            ));
        }
        println!(
            "{}: starting w/nn @ {}",
            "train".cyan(),
            init.at_epoch
                .map(|e| format!["epoch {}", e + 1])
                .unwrap_or_else(|| "init".to_string())
        );
        if let None = init.at_epoch {
            reporting.report(&init.nn, None)?;
        }
        for epoch in init.at_epoch.map(|e| e + 1).unwrap_or(0)..hyperparams.epochs {
            let batches = train.batch(hyperparams.batch_size);
            let sgd_bar = ProgressBar::new(batches.len() as u64)
                .with_style(ProgressStyle::with_template("{prefix}: {bar:40} {pos:>4}/{len:4} [{eta_precise}] / avg batch loss = {msg}")
                                .unwrap()
                                .progress_chars("=> "))
                .with_prefix(format!["train@epoch {}", epoch + 1].blue().to_string());
            let mut total_loss = 0.0;
            for (i, batch) in batches.into_iter().enumerate() {
                let mut concat_example = EagerExample {
                    input: vec![],
                    output: vec![],
                };
                for example in batch {
                    concat_example.input.push(example.input().to_cpu2());
                    concat_example.output.push(example.output().to_cpu2());
                }
                let example = EagerExample {
                    input: Tensor::init(Tt(concat_example.input)).unwrap(),
                    output: Tensor::init(Tt(concat_example.output)).unwrap(),
                };
                let mut current = example.input;
                let mut activations = vec![];
                for axon in &init.nn.axons {
                    activations.push(current.clone());
                    current = axon.forward(current);
                }
                activations.push(current.clone());
                let losses = loss.run(&current, &example.output);
                std::mem::drop(current);
                total_loss += losses.loss.data.iter().sum::<f64>() / *losses.loss.shape.first().unwrap_or(&1) as f64;
                let c = hyperparams.learning_rate / hyperparams.batch_size as f64;
                let mut grad = losses.prime;
                for (i, axon) in init.nn.axons.iter_mut().enumerate().rev() {
                    let a_in = &activations[i];
                    let a_out = &activations[i + 1];
                    grad = axon.backward(c, &a_in, &a_out, grad);
                }
                sgd_bar.inc(1);
                sgd_bar.set_message(format!["{:.3}", total_loss / (i + 1) as f64]);
            }
            sgd_bar.finish();
            reporting.report(&init.nn, Some(epoch))?;
        }
        Ok(init.nn)
    }
    
    pub fn read(read: &mut impl Read) -> io::Result<Self> {
        let mut signature = [0u8; 8];
        read.read_exact(&mut signature)?;
        if &signature != b"NeuralNt" {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "invalid neuralnet signature",
            ));
        }
        let mut nb = [0u8; 8];
        read.read_exact(&mut nb)?;
        let axon_count = usize::from_le_bytes(nb);
        let mut axons = Vec::with_capacity(axon_count);
        for _ in 0..axon_count {
            axons.push(Axon::read(read)?)
        }
        Ok(Self { axons })
    }

    pub fn write(&self, write: &mut impl Write) -> io::Result<()> {
        write.write_all(b"NeuralNt")?;
        write.write_all(&self.axons.len().to_le_bytes())?;
        for axon in &self.axons {
            axon.write(write)?;
        }
        Ok(())
    }
}

pub trait Setup {
    fn setup<'a>(&self) -> io::Result<Init<'a>>;
}

pub trait Reporting {
    fn report(&self, nn: &NeuralNetwork, epoch: Option<u64>) -> io::Result<()>;
}

pub struct Init<'a> {
    nn: NeuralNetwork<'a>,
    at_epoch: Option<u64>,
}

pub struct Layers(Vec<Layer>);

impl Layers {
    pub fn new(layers: Vec<Layer>) -> Option<Self> {
        if layers.len() < 2 {
            return None;
        }
        Some(Self(layers))
    }
}

impl Setup for Layers {
    fn setup<'a>(&self) -> io::Result<Init<'a>> {
        let mut axons = vec![];
        for i in 0..self.0.len() {
            axons.push(self.0[i].axon(if i > 0 { Some(&self.0[i - 1]) } else { None }));
        }
        Ok(Init {
            nn: NeuralNetwork { axons },
            at_epoch: None,
        })
    }
}

pub struct Hyperparams {
    pub epochs: u64,
    pub batch_size: usize,
    pub learning_rate: f64,
}

pub enum Layer {
    Dense {
        input_shape: Option<Vec<usize>>,
        neurons: usize,
        activation: Activation,
    }
}

impl Layer {
    fn axon<'a>(&self, last: Option<&Layer>) -> Axon<'a> {
        match self {
            Self::Dense {
                input_shape,
                neurons,
                activation,
            } => Axon::Dense(FeedForward::new(
                    last.map(|l| l.activation_shape())
                        .or(input_shape.clone())
                        .unwrap(),
                    *neurons,
                    *activation,
                ))
        }
    }

    fn activation_shape(&self) -> Vec<usize> {
        match self {
            Self::Dense { neurons, .. } => vec![*neurons]
        }
    }
}


pub struct Checkpoint<'a, S: Setup, R: Reporting> {
    setup: &'a S,
    reporting: &'a R,
    path: &'a Path
}

impl<'a, S: Setup, R: Reporting> Checkpoint<'a, S, R> {
    pub fn new(setup: &'a S, reporting: &'a R, path: &'a Path) -> Self {
        Self {
            setup,
            reporting,
            path
        }
    }

    fn checkpoint_path(&self, epoch: Option<u64>) -> PathBuf {
        self.path.join(Path::new(&format![
            "{}.nn",
            epoch
                .map(|e| (e + 1).to_string())
                .unwrap_or_else(|| "init".to_string())
        ]))
    }
}

impl<'a, S: Setup, R: Reporting> Setup for Checkpoint<'a, S, R> {
    fn setup<'b>(&self) -> io::Result<Init<'b>> {
        fn open<'c>(path: &Path) -> io::Result<NeuralNetwork<'c>> {
            let mut file = File::open(path)?;
            let nn = NeuralNetwork::<'c>::read(&mut file)?;
            Ok(nn)
        }
        let amount = self.path.read_dir()?.count();
        for epoch in (0..amount)
            .map(|i| Some(i as u64))
            .rev()
            .chain(std::iter::once(None))
        {
            let path = self.checkpoint_path(epoch);
            match open(&path) {
                Ok(nn) => {
                    println!(
                        "{}: located checkpointed nn at epoch {}",
                        "checkpoint".red(),
                        epoch
                            .map(|e| (e + 1).to_string())
                            .unwrap_or_else(|| "init".to_string())
                    );
                    return Ok(Init {
                        nn,
                        at_epoch: epoch,
                    });
                }
                Err(e) => {
                    println!(
                        "{}: no checkpointed nn located at {}: {}",
                        "checkpoint".red(),
                        path.display(),
                        e
                    );
                }
            }
        }
        self.setup.setup()
    }
}

impl<'a, S: Setup, R: Reporting> Reporting for Checkpoint<'a, S, R> {
    fn report(&self, nn: &NeuralNetwork, epoch: Option<u64>) -> io::Result<()> {
        self.reporting.report(nn, epoch)?;
        let path = self.checkpoint_path(epoch);
        println!(
            "{}: writing nn to {}",
            "checkpoint".red(),
            path.display().to_string()
        );
        let mut file = File::create(path)?;
        nn.write(&mut file)?;
        Ok(())
    }
}