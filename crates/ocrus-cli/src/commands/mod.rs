pub mod bench;
pub mod dataset;
pub mod recognize;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ocrus")]
#[command(version, about = "Lightning-fast Japanese OCR")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Recognize text from an image
    Recognize(RecognizeArgs),
    /// Run benchmarks
    Bench(BenchArgs),
    /// Generate training dataset
    Dataset(DatasetArgs),
}

#[derive(Parser)]
pub struct DatasetArgs {
    #[command(subcommand)]
    pub command: DatasetCommands,
}

#[derive(Subcommand)]
pub enum DatasetCommands {
    /// Generate training data from fonts x characters x augmentations
    Generate(DatasetGenerateArgs),
    /// Generate training data from char_accuracy test failures
    FromFailures(DatasetFailuresArgs),
}

#[derive(Parser)]
pub struct DatasetGenerateArgs {
    /// Output directory
    #[arg(short, long)]
    pub output: PathBuf,
    /// Character categories (comma-separated)
    #[arg(short, long, default_value = "hiragana,katakana,jis_level1,jis_level2")]
    pub categories: String,
    /// Characters per image
    #[arg(long, default_value = "15")]
    pub chars_per_image: usize,
    /// Samples per character
    #[arg(long, default_value = "5")]
    pub samples_per_char: usize,
    /// Validation split ratio
    #[arg(long, default_value = "0.1")]
    pub val_ratio: f32,
    /// Character data directory
    #[arg(long)]
    pub char_data_dir: Option<PathBuf>,
}

#[derive(Parser)]
pub struct DatasetFailuresArgs {
    /// Path to failures.json
    #[arg(short, long)]
    pub failures: PathBuf,
    /// Output directory
    #[arg(short, long)]
    pub output: PathBuf,
    /// Samples per failed character
    #[arg(long, default_value = "10")]
    pub samples_per_char: usize,
}

#[derive(Parser)]
pub struct RecognizeArgs {
    /// Input image path
    pub input: PathBuf,

    /// Output format
    #[arg(short, long, default_value = "text")]
    pub format: OutputFormat,

    /// Model directory
    #[arg(long, env = "OCRUS_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,

    /// Number of threads
    #[arg(short = 't', long)]
    pub threads: Option<usize>,

    /// Processing mode
    #[arg(long, default_value = "auto")]
    pub mode: CliMode,

    /// Character set
    #[arg(long, default_value = "full")]
    pub charset: CliCharset,

    /// Custom dictionary path
    #[arg(long)]
    pub dict: Option<PathBuf>,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum CliMode {
    Auto,
    Fastest,
    Accurate,
}

#[derive(Clone, Debug, clap::ValueEnum)]
pub enum CliCharset {
    Full,
    Jis,
}

#[derive(Parser)]
pub struct BenchArgs {
    /// Input image path
    pub input: PathBuf,

    /// Number of iterations
    #[arg(short = 'n', long, default_value = "10")]
    pub iterations: u32,

    /// Model directory
    #[arg(long, env = "OCRUS_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,
}
