mod arnold;
mod arnold_to_mlir;
mod transform;

use clap::arg;
use clap::ArgMatches;
use clap::Args;
use clap::Command;
use std::env::ArgsOs;
use std::io::Read;
use xrcf::convert::RewriteResult;
use xrcf::init_subscriber;
use xrcf::Passes;

use crate::transform::parse_and_transform;

/// A compiler for the ArnoldC language.
#[derive(Args, Debug)]
#[command(version, about)]
struct ArnoldcArgs {
    /// The input file (- is interpreted as stdin)
    #[arg(default_value = "-")]
    input: String,
    /// Convert ArnoldC operations to MLIR
    #[arg(long, name = "convert-arnold-to-mlir")]
    convert_arnold_to_mlir: bool,
    /// Compile the code
    #[arg(long, name = "compile")]
    compile: bool,
    /// Print debug information
    #[arg(long, name = "debug")]
    debug: bool,
}

fn cli() -> Command {
    let cli = Command::new("arnoldc").args(xrcf::default_passes());
    let cli = ArnoldcArgs::augment_args(cli);
    cli
}

fn compile_passes() -> Vec<&'static str> {
    vec![
        "--convert-arnold-to-mlir",
        "--convert-experimental-to-mlir",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-func-to-llvm",
        "--convert-mlir-to-llvmir",
    ]
}

fn passes_from_args(args: ArgsOs, matches: ArgMatches) -> Passes {
    if matches.get_flag("compile") {
        Passes::from_convert_vec(compile_passes())
    } else {
        Passes::from_convert_args(args)
    }
}

fn init_tracing(level: tracing::Level) {
    match init_subscriber(level) {
        Ok(_) => (),
        Err(_e) => (),
    }
}

fn main() {
    let cli = cli();
    let args = std::env::args_os();
    let matches = cli.get_matches();
    if matches.get_flag("debug") {
        init_tracing(tracing::Level::DEBUG);
    } else {
        init_tracing(tracing::Level::INFO);
    }
    let passes = passes_from_args(args, matches.clone());

    let input = matches.get_one::<String>("input").unwrap();

    let input_text = if input == "-" {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer).unwrap();
        buffer
    } else {
        std::fs::read_to_string(input).unwrap()
    };

    let result = parse_and_transform(&input_text, &passes).unwrap();
    let result = match result {
        RewriteResult::Changed(op) => op.op.try_read().unwrap().to_string(),
        RewriteResult::Unchanged => input_text.to_string(),
    };
    println!("{result}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use indoc::indoc;
    use xrcf::convert::RewriteResult;
    use xrcf::tester::Tester;

    fn run_app(args: Vec<&str>, input_text: &str) -> Result<RewriteResult> {
        let cli = cli();
        let args_owned: Vec<String> = args.iter().map(|&s| s.to_string()).collect();
        let _matches = cli.try_get_matches_from(args_owned)?;
        let passes = Passes::from_convert_vec(args);
        let result = parse_and_transform(input_text, &passes)?;
        Ok(result)
    }

    #[test]
    fn test_help() {
        let args = vec!["xr-example", "--help"];
        let result = run_app(args, "");
        let err = match result {
            Ok(_) => panic!("Expected an error"),
            Err(e) => e,
        };
        assert!(err
            .to_string()
            .contains("A compiler for the ArnoldC language"));
        assert!(err.to_string().contains("--convert-func-to-llvm"));
    }

    #[test]
    fn test_invalid_args() {
        let result = run_app(vec!["xr-example", "--invalid-flag"], "");
        assert!(result.is_err());
    }

    #[test]
    fn test_pass_order() {
        Tester::init_tracing();
        let src = indoc! {r#"
        func.func @main() -> i32 {
            %0 = arith.constant 1 : i32
            return %0 : i32
        }
        "#
        };
        // The order of these passes is important.
        let args = vec!["--convert-func-to-llvm", "--convert-mlir-to-llvmir"];
        tracing::info!("\nBefore {args:?}:\n{src}");
        let result = run_app(args.clone(), src);
        assert!(result.is_ok());
        let actual = match result.unwrap() {
            RewriteResult::Changed(op) => {
                let op = op.op.try_read().unwrap();
                op.to_string()
            }
            RewriteResult::Unchanged => panic!("Expected a change"),
        };
        tracing::info!("\nAfter {args:?}:\n{actual}");
        assert!(actual.contains("define i32 @main"));
    }
}
