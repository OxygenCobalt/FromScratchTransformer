Compile with `MALLOC_CONF="thp:always,metadata_thp:always" RUSTFLAGS="-C target-cpu=native" cargo build --release`

Run with `./target/release/nn`