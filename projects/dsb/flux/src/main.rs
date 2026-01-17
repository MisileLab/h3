use clap::{Parser, Subcommand};
use flux_lang::{FluxCompiler, eval};
use std::fs;
use std::path::PathBuf;
use anyhow::Result;

#[derive(Parser)]
#[command(name = "fluxc")]
#[command(about = "Flux language compiler", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Type-check a Flux source file
    Check {
        /// Path to the Flux source file
        file: PathBuf,
    },
    /// Compile a Flux source file
    Build {
        /// Path to the Flux source file
        file: PathBuf,

        /// Target platform (x86_64 or wasm32)
        #[arg(short, long, default_value = "x86_64")]
        target: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Compile and run a Flux source file
    Run {
        /// Path to the Flux source file
        file: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Check { file } => check_file(&file),
        Commands::Build { file, target, output } => build_file(&file, &target, output),
        Commands::Run { file } => run_file(&file),
    }
}

fn check_file(file: &PathBuf) -> Result<()> {
    let source = fs::read_to_string(file)?;
    let mut compiler = FluxCompiler::new();

    match compiler.check_source(&source) {
        Ok(_) => {
            println!("✓ Type checking passed");
            Ok(())
        }
        Err(e) => {
            eprintln!("✗ Type checking failed:");
            eprintln!("  {}", e);
            std::process::exit(1);
        }
    }
}

fn build_file(file: &PathBuf, target: &str, output: Option<PathBuf>) -> Result<()> {
    let source = fs::read_to_string(file)?;
    let mut compiler = FluxCompiler::new();

    println!("Compiling {} for target {}...", file.display(), target);

    match target {
        "x86_64" | "x86-64" => {
            match compiler.compile_source_x86(&source) {
                Ok(_backend) => {
                    let output_path = output.unwrap_or_else(|| {
                        let mut path = file.clone();
                        path.set_extension("out");
                        path
                    });

                    println!("✓ Compilation successful");
                    println!("  Output: {}", output_path.display());
                    println!("  (Note: x86-64 JIT compilation - binary not written to disk)");
                    Ok(())
                }
                Err(e) => {
                    eprintln!("✗ Compilation failed:");
                    eprintln!("  {}", e);
                    std::process::exit(1);
                }
            }
        }
        "wasm32" | "wasm" => {
            match compiler.compile_source_wasm(&source) {
                Ok(wasm_bytes) => {
                    let output_path = output.unwrap_or_else(|| {
                        let mut path = file.clone();
                        path.set_extension("wasm");
                        path
                    });

                    fs::write(&output_path, wasm_bytes)?;

                    println!("✓ Compilation successful");
                    println!("  Output: {}", output_path.display());
                    println!("  Size: {} bytes", fs::metadata(&output_path)?.len());
                    Ok(())
                }
                Err(e) => {
                    eprintln!("✗ Compilation failed:");
                    eprintln!("  {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("✗ Unknown target: {}", target);
            eprintln!("  Supported targets: x86_64, wasm32");
            std::process::exit(1);
        }
    }
}

fn run_file(file: &PathBuf) -> Result<()> {
    let source = fs::read_to_string(file)?;

    println!("Compiling and running {}...", file.display());

    match eval(&source) {
        Ok(result) => {
            println!("\n✓ Program executed successfully");
            println!("  Result: {}", result);
            Ok(())
        }
        Err(e) => {
            eprintln!("\n✗ Execution failed:");
            eprintln!("  {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_check_valid_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "inc: i32 -> i32").unwrap();
        writeln!(file, "inc x = x + 1").unwrap();
        file.flush().unwrap();

        let result = check_file(&file.path().to_path_buf());
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_simple_program() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "main: i32").unwrap();
        writeln!(file, "main = 42").unwrap();
        file.flush().unwrap();

        let result = run_file(&file.path().to_path_buf());
        assert!(result.is_ok());
    }
}
