Note: Very InDev, Currently have done shallow nets/convnets on MNIST and currently working on shallow language models (bengio)

Compile with `MALLOC_CONF="thp:always,metadata_thp:always" RUSTFLAGS="-C target-cpu=native" cargo build --release`

Run with `./target/release/nn`