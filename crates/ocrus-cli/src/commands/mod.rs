pub mod bench;
pub mod dataset;
pub mod recognize;
pub mod tui;

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
    /// Interactive TUI for OCR operations
    Tui,
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
    /// Native image heights (comma-separated); training resizes images as needed
    #[arg(long, value_delimiter = ',', default_value = "48", value_parser = clap::value_parser!(u32).range(1..))]
    pub render_heights: Vec<u32>,
    /// Samples per character
    #[arg(long, default_value = "5")]
    pub samples_per_char: usize,
    /// Validation split ratio
    #[arg(long, default_value = "0.1")]
    pub val_ratio: f32,
    /// Character data directory
    #[arg(long)]
    pub char_data_dir: Option<PathBuf>,
    /// Font styles to include (comma-separated: mincho,gothic,script,monospace,other)
    #[arg(long)]
    pub font_styles: Option<String>,
    /// Additional font directories (searched recursively; repeat to add more)
    #[arg(long)]
    pub font_dirs: Vec<PathBuf>,
    /// Use only --font-dirs, without searching system fonts
    #[arg(long, requires = "font_dirs")]
    pub no_system_fonts: bool,
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
    /// Native font-rendering heights (comma-separated; incompatible with --test-images)
    #[arg(long, value_delimiter = ',', default_value = "48", value_parser = clap::value_parser!(u32).range(1..), conflicts_with = "test_images")]
    pub render_heights: Vec<u32>,
    /// Additional font directories to search
    #[arg(long)]
    pub font_dirs: Vec<PathBuf>,
    /// Use only --font-dirs, without searching system fonts
    #[arg(long, requires = "font_dirs")]
    pub no_system_fonts: bool,
    /// Ignore font_name in failures (generate with all available fonts)
    #[arg(long)]
    pub all_fonts: bool,
    /// Use pre-rendered test images instead of font rendering
    #[arg(long)]
    pub test_images: Option<PathBuf>,
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

    /// Enable ruby (furigana) separation
    #[arg(long)]
    pub ruby: bool,

    /// Cascade classifier model path
    #[arg(long)]
    pub cascade: Option<PathBuf>,
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

    /// ベンチマークの反復回数（1 以上）
    #[arg(short = 'n', long, default_value = "10", value_parser = clap::value_parser!(u32).range(1..))]
    pub iterations: u32,

    /// Model directory
    #[arg(long, env = "OCRUS_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_render_heights_parse_validate_and_conflict_with_test_images() {
        let cli = Cli::try_parse_from([
            "ocrus",
            "dataset",
            "generate",
            "--output",
            "data",
            "--render-heights",
            "12,16,48",
        ])
        .unwrap();
        let Commands::Dataset(DatasetArgs {
            command: DatasetCommands::Generate(args),
        }) = cli.command
        else {
            panic!("expected dataset generation");
        };
        assert_eq!(args.render_heights, [12, 16, 48]);
        assert!(
            Cli::try_parse_from([
                "ocrus",
                "dataset",
                "generate",
                "--output",
                "data",
                "--render-heights",
                "0",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "ocrus",
                "dataset",
                "from-failures",
                "--output",
                "data",
                "--failures",
                "f.json",
                "--test-images",
                "images",
                "--render-heights",
                "12",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "ocrus",
                "dataset",
                "from-failures",
                "--output",
                "data",
                "--failures",
                "f.json",
                "--test-images",
                "images",
            ])
            .is_ok()
        );
    }

    #[test]
    fn dataset_accepts_multiple_font_roots_and_requires_explicit_roots_when_isolated() {
        let cli = Cli::try_parse_from([
            "ocrus",
            "dataset",
            "generate",
            "--output",
            "data",
            "--font-dirs",
            "fonts/a",
            "--font-dirs",
            "fonts/b",
            "--no-system-fonts",
        ])
        .unwrap();
        let Commands::Dataset(DatasetArgs {
            command: DatasetCommands::Generate(args),
        }) = cli.command
        else {
            panic!("expected dataset generation");
        };
        assert_eq!(
            args.font_dirs,
            [PathBuf::from("fonts/a"), PathBuf::from("fonts/b")]
        );
        assert!(args.no_system_fonts);
        assert!(
            Cli::try_parse_from([
                "ocrus",
                "dataset",
                "generate",
                "--output",
                "data",
                "--no-system-fonts",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "ocrus",
                "dataset",
                "from-failures",
                "--failures",
                "failures.json",
                "--output",
                "data",
                "--no-system-fonts",
            ])
            .is_err()
        );
    }

    #[test]
    fn benchmark_rejects_zero_iterations() {
        assert!(Cli::try_parse_from(["ocrus", "bench", "image.png", "-n", "0"]).is_err());
        assert!(Cli::try_parse_from(["ocrus", "bench", "image.png", "-n", "1"]).is_ok());
        assert!(Cli::try_parse_from(["ocrus", "bench", "image.png"]).is_ok());
    }
}
