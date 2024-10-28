mod transform;

use anyhow::Result;
use clap::arg;
use clap::ArgMatches;
use clap::Args;
use clap::Command;
use std::io::Read;
use xrcf::Passes;

use crate::transform::parse_and_transform;

/// An example Python compiler that can compile a small subset of Python
#[derive(Args, Debug)]
#[command(version, about)]
struct PythonArgs {
    /// The input file (defaults to "-", which is stdin)
    #[arg(default_value = "-")]
    input: String,
    /// Convert Python operations to MLIR
    #[arg(long, name = "convert-python-to-mlir")]
    convert_python_to_mlir: bool,
}

fn process(matches: &ArgMatches, input_text: &str) -> Result<()> {
    let passes = Passes::from_convert_args(&matches);

    parse_and_transform(&input_text, &passes)?;
    Ok(())
}

fn main() {
    let cli = Command::new("pythonc").args(xrcf::default_passes());
    let cli = PythonArgs::augment_args(cli);
    let matches = cli.get_matches();

    let input = matches.get_one::<String>("input").unwrap();

    let input_text = if input == "-" {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer).unwrap();
        buffer
    } else {
        std::fs::read_to_string(input).unwrap()
    };

    process(&matches, &input_text).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    fn run_app(args: Vec<&str>, input_text: &str) -> Result<()> {
        let cli = Command::new("pythonc").args(xrcf::default_passes());
        let cli = PythonArgs::augment_args(cli);
        let args_owned: Vec<String> = args.iter().map(|&s| s.to_string()).collect();
        let matches = cli.try_get_matches_from(args_owned)?;
        process(&matches, input_text)?;
        Ok(())
    }

    #[test]
    fn test_help() {
        let args = vec!["pythonc", "--help"];
        let result = run_app(args, "");
        let err = match result {
            Ok(_) => panic!("Expected an error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("Python compiler"));
        assert!(err.to_string().contains("--convert-func-to-llvm"));
    }

    #[test]
    fn test_invalid_args() {
        let result = run_app(vec!["pythonc", "--invalid-flag"], "");
        assert!(result.is_err());
    }

    #[test]
    fn test_pass_order() {
        let src = indoc! {
            r#"
            func.func @main() -> i32 {
                %0 = arith.constant 1 : i32
                return %0 : i32
            }
            "#
        };
        // The order of these passes is important.
        let args = vec!["--convert-func-to-llvm", "--convert-mlir-to-llvmir"];
        let result = run_app(args, src);
        println!("{:?}", result);
        assert!(result.is_ok());
    }
}
