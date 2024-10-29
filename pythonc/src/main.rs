mod python_to_mlir;
mod transform;

use anyhow::Result;
use clap::arg;
use clap::ArgMatches;
use clap::Args;
use clap::Command;
use std::io::Read;
use xrcf::convert::RewriteResult;
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

fn process(passes: &Passes, input_text: &str) -> Result<RewriteResult> {
    let result = parse_and_transform(&input_text, &passes)?;
    Ok(result)
}

fn main() {
    let cli = Command::new("pythonc").args(xrcf::default_passes());
    let cli = PythonArgs::augment_args(cli);
    let args = std::env::args_os();
    let passes = Passes::from_convert_args(args);
    let matches = cli.get_matches();

    let input = matches.get_one::<String>("input").unwrap();

    let input_text = if input == "-" {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer).unwrap();
        buffer
    } else {
        std::fs::read_to_string(input).unwrap()
    };

    process(&passes, &input_text).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;

    fn run_app(args: Vec<&str>, input_text: &str) -> Result<RewriteResult> {
        let cli = Command::new("pythonc").args(xrcf::default_passes());
        let cli = PythonArgs::augment_args(cli);
        let args_owned: Vec<String> = args.iter().map(|&s| s.to_string()).collect();
        let _matches = cli.try_get_matches_from(args_owned)?;
        println!("Args: {:?}", args);
        let passes = Passes::from_convert_vec(args);
        println!("Passes: {}", passes);
        let result = process(&passes, input_text)?;
        Ok(result)
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
        let src = indoc! {r#"
        func.func @main() -> i32 {
            %0 = arith.constant 1 : i32
            return %0 : i32
        }
        "#
        };
        // The order of these passes is important.
        let args = vec!["--convert-func-to-llvm", "--convert-mlir-to-llvmir"];
        println!("\nBefore {args:?}:\n{src}");
        let result = run_app(args.clone(), src);
        assert!(result.is_ok());
        let actual = match result.unwrap() {
            RewriteResult::Changed(op) => {
                let op = op.0.try_read().unwrap();
                op.to_string()
            }
            RewriteResult::Unchanged => panic!("Expected a change"),
        };
        println!("\nAfter {args:?}:\n{actual}");
        assert!(actual.contains("define i32 @main"));
    }
}
