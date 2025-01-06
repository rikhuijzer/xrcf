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
use transform::WeaParserDispatch;
use transform::WeaTransformDispatch;
use xrcf::convert::RewriteResult;
use xrcf::frontend::Parser;
use xrcf::init_subscriber;
use xrcf::shared::SharedExt;
use xrcf::transform;
use xrcf::Passes;
use xrcf::TransformOptions;

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

    let src = if input == "-" {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer).unwrap();
        buffer
    } else {
        std::fs::read_to_string(input).unwrap()
    };

    let module = Parser::<WeaParserDispatch>::parse(&src).unwrap();
    let result = transform::<WeaTransformDispatch>(module, &options).unwrap();
    let result = match result {
        RewriteResult::Changed(op) => op.op.rd().to_string(),
        RewriteResult::Unchanged => src.to_string(),
    };
    println!("{result}");
}
