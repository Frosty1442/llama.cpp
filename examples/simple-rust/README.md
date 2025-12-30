# llama-simple (Pure Rust)

A pure Rust implementation of LLM inference, equivalent to llama.cpp's simple example. This uses the [candle](https://github.com/huggingface/candle) ML framework - no C/C++ dependencies required.

## Features

- **Pure Rust**: No FFI, no C/C++ toolchain needed
- **GGUF Support**: Loads quantized models in GGUF format
- **Greedy & Temperature Sampling**: Configurable generation parameters
- **Cross-platform**: Works on Linux, macOS, Windows

## Building

```bash
cd examples/simple-rust
cargo build --release
```

### With GPU acceleration

```bash
# CUDA (NVIDIA)
cargo build --release --features cuda

# Metal (Apple Silicon)
cargo build --release --features metal

# Accelerate (macOS CPU optimization)
cargo build --release --features accelerate
```

## Usage

```bash
./target/release/llama-simple -m /path/to/model.gguf [options] [prompt]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <PATH>` | Path to the GGUF model file | (required) |
| `-t, --tokenizer <PATH>` | Path to tokenizer.json | (auto-detect) |
| `-n, --n-predict <N>` | Number of tokens to generate | 32 |
| `--temperature <T>` | Sampling temperature (0 = greedy) | 0.0 |
| `--top-p <P>` | Top-p (nucleus) sampling | 1.0 |
| `--seed <S>` | Random seed for sampling | 299792458 |

### Examples

```bash
# Basic greedy generation
./target/release/llama-simple -m models/llama-7b-q4.gguf

# Custom prompt with more tokens
./target/release/llama-simple -m models/llama-7b-q4.gguf -n 100 "Once upon a time"

# Temperature sampling for more creative output
./target/release/llama-simple -m models/llama-7b-q4.gguf --temperature 0.8 --top-p 0.9

# With explicit tokenizer
./target/release/llama-simple -m model.gguf -t tokenizer.json "Hello world"
```

## Project Structure

```
simple-rust/
├── Cargo.toml      # Pure Rust dependencies (candle, tokenizers, clap)
├── src/
│   └── main.rs     # Complete inference implementation
└── README.md
```

## How It Works

1. **GGUF Parsing**: Uses `candle_core::quantized::gguf_file` to parse the GGUF container format
2. **Model Loading**: Builds a quantized Llama model using `candle_transformers::models::quantized_llama`
3. **Tokenization**: Uses the `tokenizers` crate (same as Hugging Face Transformers)
4. **Inference**: Pure Rust tensor operations via candle
5. **Sampling**: `LogitsProcessor` for greedy or temperature-based token selection

## Tokenizer

The tokenizer can be provided in several ways (checked in order):

1. `--tokenizer` command line argument
2. Embedded in GGUF metadata (`tokenizer.huggingface.json`)
3. `tokenizer.json` file in the same directory as the model

## Comparison with C++ Version

| Aspect | C++ (llama.cpp) | Rust (this) |
|--------|-----------------|-------------|
| Dependencies | C/C++ toolchain, cmake | Rust toolchain only |
| GPU Support | CUDA, Metal, Vulkan, etc. | CUDA, Metal (via candle) |
| Binary Size | ~2-5 MB | ~10-15 MB |
| Performance | Highly optimized | Good, improving |
| GGUF Support | Full | Most common quantizations |

## License

MIT
