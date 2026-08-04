mod audio;
mod burn_rvc;
mod config;
mod pipeline;
mod retrieval;
mod rvc;
mod simd;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::config::RvcConfig;
use crate::pipeline::RvcPipeline;

#[derive(Parser, Debug)]
#[command(name = "vocal", about = "Rust RVC-style voice conversion pipeline")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Infer {
        #[arg(long)]
        input_wav: PathBuf,
        #[arg(long)]
        output_wav: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        burnpack: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        pitch_shift_semitones: i32,
        #[arg(long, default_value_t = 0.0)]
        index_mix: f32,
        #[arg(long, default_value_t = 0.0)]
        rms_mix_rate: f32,
        #[arg(long, default_value_t = 48000)]
        target_sr: u32,
        #[arg(long, default_value_t = false)]
        force_female_extreme: bool,
    },
    NewConfig {
        #[arg(long)]
        output: PathBuf,
    },
    InspectBurnpack {
        #[arg(long)]
        model: PathBuf,
    },
    DumpBurnpack {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 200)]
        limit: usize,
    },
    ConvertPth {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        top_level_key: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::NewConfig { output } => {
            let cfg = RvcConfig::default();
            let text = serde_json::to_string_pretty(&cfg)?;
            std::fs::write(&output, text)
                .with_context(|| format!("failed writing {}", output.display()))?;
            println!("wrote {}", output.display());
        }
        Commands::InspectBurnpack { model } => {
            let model = burn_rvc::BurnRvcModel::from_burnpack(&model)?;
            println!("{}", model.summary());
            if let Some(name) = model.first_tensor_name() {
                println!("first_tensor={name}");
            }
        }
        Commands::DumpBurnpack { model, limit } => {
            let model = burn_rvc::BurnRvcModel::from_burnpack(&model)?;
            for line in model.dump_tensors(limit) {
                println!("{line}");
            }
        }
        Commands::ConvertPth {
            input,
            output,
            top_level_key,
        } => {
            let count =
                burn_rvc::convert_pth_to_burnpack(&input, &output, top_level_key.as_deref())?;
            println!(
                "converted {} tensors: {} -> {}",
                count,
                input.display(),
                output.display()
            );
        }
        Commands::Infer {
            input_wav,
            output_wav,
            config,
            burnpack,
            pitch_shift_semitones,
            index_mix,
            rms_mix_rate,
            target_sr,
            force_female_extreme,
        } => {
            if !(0.0..=1.0).contains(&index_mix) {
                bail!("--index-mix must be in [0,1]");
            }
            if !(0.0..=1.0).contains(&rms_mix_rate) {
                bail!("--rms-mix-rate must be in [0,1]");
            }

            let mut cfg = if let Some(path) = config {
                let text = std::fs::read_to_string(&path)
                    .with_context(|| format!("failed reading {}", path.display()))?;
                serde_json::from_str::<RvcConfig>(&text).context("invalid config JSON")?
            } else {
                RvcConfig::default()
            };

            cfg.pitch_shift_semitones = pitch_shift_semitones;
            cfg.index_mix = index_mix;
            cfg.rms_mix_rate = rms_mix_rate;
            cfg.target_sr = target_sr;
            cfg.force_female_extreme = force_female_extreme;
            if let Some(path) = burnpack {
                cfg.burnpack_path = Some(path);
            }

            let pipe = RvcPipeline::new(cfg)?;
            let (x, sr) = audio::read_wav_mono(&input_wav)?;
            let y = pipe.infer(&x, sr)?;
            audio::write_wav_mono(&output_wav, &y, pipe.target_sr())?;
            println!("wrote {}", output_wav.display());
        }
    }
    Ok(())
}
