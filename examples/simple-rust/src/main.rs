//! Simple LLM inference example in pure Rust
//!
//! This is a pure Rust implementation equivalent to llama.cpp's simple example.
//! It uses the candle framework for tensor operations and model inference.
//!
//! Usage:
//!     llama-simple -m model.gguf [-n n_predict] [prompt]

use anyhow::{bail, Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as llama;
use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Simple LLM inference example
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the GGUF model file
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Path to tokenizer.json (if not embedded in GGUF)
    #[arg(short = 't', long)]
    tokenizer: Option<PathBuf>,

    /// Number of tokens to predict
    #[arg(short = 'n', long, default_value_t = 32)]
    n_predict: usize,

    /// Temperature for sampling (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Top-p sampling threshold
    #[arg(long, default_value_t = 1.0)]
    top_p: f64,

    /// Random seed
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Prompt to generate text from
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,
}

/// Token sampler that supports greedy and temperature-based sampling
struct Sampler {
    logits_processor: LogitsProcessor,
}

impl Sampler {
    fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        Self {
            logits_processor: LogitsProcessor::new(seed, temperature, top_p),
        }
    }

    fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let token = self.logits_processor.sample(&logits)?;
        Ok(token)
    }
}

/// Load tokenizer from file or try to extract from GGUF metadata
fn load_tokenizer(args: &Args, gguf: &gguf_file::Content) -> Result<Tokenizer> {
    // First try explicit tokenizer path
    if let Some(ref tokenizer_path) = args.tokenizer {
        return Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e));
    }

    // Try to get tokenizer from GGUF metadata
    if let Some(tokenizer_json) = gguf.metadata.get("tokenizer.huggingface.json") {
        if let gguf_file::Value::String(json) = tokenizer_json {
            return Tokenizer::from_bytes(json.as_bytes())
                .map_err(|e| anyhow::anyhow!("Failed to parse embedded tokenizer: {}", e));
        }
    }

    // Try common tokenizer locations relative to model
    let model_dir = args.model.parent().unwrap_or(std::path::Path::new("."));
    let tokenizer_path = model_dir.join("tokenizer.json");
    if tokenizer_path.exists() {
        return Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e));
    }

    bail!(
        "Could not find tokenizer. Please provide --tokenizer path or place tokenizer.json next to the model."
    )
}

/// Check if a token is an end-of-generation token
fn is_eog_token(token: u32, eos_token_id: u32, eot_tokens: &[u32]) -> bool {
    token == eos_token_id || eot_tokens.contains(&token)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Combine prompt arguments into a single string
    let prompt = if args.prompt.is_empty() {
        "Hello my name is".to_string()
    } else {
        args.prompt.join(" ")
    };

    eprintln!("Prompt: {}", prompt);
    eprintln!("Model: {}", args.model.display());
    eprintln!("n_predict: {}", args.n_predict);
    eprintln!("temperature: {}", args.temperature);
    eprintln!();

    // Select device (CPU for now, can be extended for CUDA/Metal)
    let device = Device::Cpu;

    // Load the GGUF model file
    eprintln!("Loading model...");
    let t_load_start = Instant::now();

    let mut file = std::fs::File::open(&args.model)
        .with_context(|| format!("Failed to open model file: {}", args.model.display()))?;

    let gguf = gguf_file::Content::read(&mut file)
        .with_context(|| "Failed to parse GGUF file")?;

    // Load tokenizer
    let tokenizer = load_tokenizer(&args, &gguf)?;

    // Build the model from GGUF
    let model = llama::ModelWeights::from_gguf(gguf, &mut file, &device)
        .with_context(|| "Failed to load model weights")?;

    let t_load = t_load_start.elapsed();
    eprintln!("Model loaded in {:.2}s", t_load.as_secs_f64());

    // Get special token IDs
    let eos_token_id = tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<|end|>"))
        .unwrap_or(2); // Default EOS token ID

    // Common end-of-turn tokens
    let eot_tokens: Vec<u32> = ["<|eot_id|>", "<|end_of_turn|>", "<|im_end|>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();

    // Tokenize the prompt
    let encoding = tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let n_prompt = prompt_tokens.len();

    eprintln!("Prompt tokens: {}", n_prompt);

    // Initialize the sampler
    let temperature = if args.temperature <= 0.0 {
        None // Greedy sampling
    } else {
        Some(args.temperature)
    };
    let top_p = if args.top_p >= 1.0 {
        None
    } else {
        Some(args.top_p)
    };
    let mut sampler = Sampler::new(args.seed, temperature, top_p);

    // Print the prompt
    print!("{}", prompt);
    io::stdout().flush()?;

    // Main generation loop
    let t_gen_start = Instant::now();
    let mut n_decode = 0usize;
    let mut all_tokens = prompt_tokens.clone();
    let mut pos = 0usize;

    // Process prompt tokens
    let prompt_tensor = Tensor::new(prompt_tokens.as_slice(), &device)?
        .unsqueeze(0)?;
    let logits = model.forward(&prompt_tensor, pos)?;
    pos += n_prompt;

    // Sample first token
    let mut next_token = sampler.sample(&logits.i((.., n_prompt - 1.., ..))?)?;

    // Check for immediate EOG
    if is_eog_token(next_token, eos_token_id, &eot_tokens) {
        println!();
        eprintln!("\n(end of generation on first token)");
        return Ok(());
    }

    // Decode and print first generated token
    if let Some(text) = tokenizer.decode(&[next_token], false).ok() {
        print!("{}", text);
        io::stdout().flush()?;
    }
    all_tokens.push(next_token);
    n_decode += 1;

    // Continue generating tokens
    while n_decode < args.n_predict {
        // Forward pass for single token
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, pos)?;
        pos += 1;

        // Sample next token
        next_token = sampler.sample(&logits)?;

        // Check for end of generation
        if is_eog_token(next_token, eos_token_id, &eot_tokens) {
            break;
        }

        // Decode and print token
        if let Some(text) = tokenizer.decode(&[next_token], false).ok() {
            print!("{}", text);
            io::stdout().flush()?;
        }

        all_tokens.push(next_token);
        n_decode += 1;
    }

    println!();

    // Print timing statistics
    let t_gen = t_gen_start.elapsed();
    let elapsed_secs = t_gen.as_secs_f64();

    eprintln!();
    eprintln!(
        "main: decoded {} tokens in {:.2} s, speed: {:.2} t/s",
        n_decode,
        elapsed_secs,
        if elapsed_secs > 0.0 {
            n_decode as f64 / elapsed_secs
        } else {
            0.0
        }
    );

    // Print detailed timing
    eprintln!();
    eprintln!("llama_perf_context_print:        load time = {:8.2} ms", t_load.as_secs_f64() * 1000.0);
    eprintln!(
        "llama_perf_context_print: prompt eval time = {:8.2} ms / {:5} tokens ({:8.2} ms per token, {:8.2} tokens per second)",
        0.0, // We don't separate prompt eval time in this simple version
        n_prompt,
        0.0,
        0.0
    );
    eprintln!(
        "llama_perf_context_print:        eval time = {:8.2} ms / {:5} runs   ({:8.2} ms per token, {:8.2} tokens per second)",
        t_gen.as_secs_f64() * 1000.0,
        n_decode,
        if n_decode > 0 { t_gen.as_secs_f64() * 1000.0 / n_decode as f64 } else { 0.0 },
        if elapsed_secs > 0.0 { n_decode as f64 / elapsed_secs } else { 0.0 }
    );
    eprintln!(
        "llama_perf_context_print:       total time = {:8.2} ms / {:5} tokens",
        (t_load.as_secs_f64() + t_gen.as_secs_f64()) * 1000.0,
        n_prompt + n_decode
    );

    eprintln!();

    Ok(())
}
