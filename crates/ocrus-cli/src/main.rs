mod commands;
mod output;

use anyhow::Result;
use clap::Parser;

use commands::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Recognize(args) => commands::recognize::run(args),
        Commands::Bench(args) => commands::bench::run(args),
        Commands::Dataset(args) => match args.command {
            commands::DatasetCommands::Generate(ref gen_args) => {
                commands::dataset::run_generate(gen_args)
            }
            commands::DatasetCommands::FromFailures(ref fail_args) => {
                commands::dataset::run_from_failures(fail_args)
            }
        },
    }
}
