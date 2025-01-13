#![allow(clippy::arc_with_non_send_sync)]
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
use xrcf::shared::SharedExt;
use xrcf::Passes;
use xrcf::TransformOptions;

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
    let cli = Command::new("arnoldc").args(xrcf::default_arguments());

    ArnoldcArgs::augment_args(cli)
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
    let options = TransformOptions::from_args(matches.clone(), passes.clone());

    let input = matches.get_one::<String>("input").unwrap();

    let input_text = if input == "-" {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer).unwrap();
        buffer
    } else {
        std::fs::read_to_string(input).unwrap()
    };

    let result = parse_and_transform(&input_text, &options).unwrap();
    let result = match result {
        RewriteResult::Changed(op) => op.op.rd().to_string(),
        RewriteResult::Unchanged => input_text.to_string(),
    };
    println!("{result}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use indoc::indoc;
    use std::panic::Location;

    use transform::ArnoldParserDispatch;
    use transform::ArnoldTransformDispatch;
    use xrcf::convert::RewriteResult;
    use xrcf::shared::Shared;
    use xrcf::tester::Tester;
    type ArnoldTester = Tester<ArnoldParserDispatch, ArnoldTransformDispatch>;

    fn run_app(
        out: Option<Shared<dyn std::io::Write + Send>>,
        args: Vec<&str>,
        input_text: &str,
    ) -> Result<RewriteResult> {
        let cli = cli();
        let args_owned: Vec<String> = args.iter().map(|&s| s.to_string()).collect();
        let _matches = cli.try_get_matches_from(args_owned)?;
        let passes = Passes::from_convert_vec(args.clone());
        let mut options = TransformOptions::from_passes(passes.clone());
        if args.contains(&"--print-ir-before-all") {
            options.set_print_ir_before_all(true);
        }
        if let Some(out) = out {
            options.set_writer(out);
        }
        let result = parse_and_transform(input_text, &options)?;
        Ok(result)
    }

    #[test]
    fn test_help() {
        let args = vec!["xr-example", "--help"];
        let result = run_app(None, args, "");
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
        let args = vec!["xr-example", "--invalid-flag"];
        let result = run_app(None, args, "");
        assert!(result.is_err());
    }

    #[test]
    fn test_pass_order() {
        ArnoldTester::init_tracing();
        let src = indoc! {r#"
        func.func @main() -> i32 {
            %0 = arith.constant 1 : i32
            return %0 : i32
        }
        "#
        };
        // The order of these passes is important.
        let args = vec![
            "--convert-func-to-llvm",
            "--convert-mlir-to-llvmir",
            "--print-ir-before-all",
        ];
        tracing::info!("\nBefore {args:?}:\n{src}");
        let out: Shared<Vec<u8>> = Shared::new(Vec::new().into());
        let result = run_app(Some(out.clone()), args.clone(), src);
        assert!(result.is_ok());
        let actual = match result.unwrap() {
            RewriteResult::Changed(op) => {
                let op = op.op.rd();
                op.to_string()
            }
            RewriteResult::Unchanged => panic!("Expected a change"),
        };
        tracing::info!("\nAfter {args:?}:\n{actual}");
        assert!(actual.contains("define i32 @main"));

        let printed = out.rd();
        let printed = String::from_utf8(printed.clone()).unwrap();
        let expected = indoc! {r#"
        // ----- // IR Dump before convert-func-to-llvm //----- //
        // ----- // IR Dump before convert-mlir-to-llvmir //----- //
        "#};
        ArnoldTester::check_lines_contain(&printed, expected, Location::caller());
    }
}
