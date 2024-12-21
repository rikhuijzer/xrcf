use clap::arg;
use clap::Args;
use clap::Command;
use std::io::Read;
use xrcf::convert::RewriteResult;
use xrcf::frontend::DefaultParserDispatch;
use xrcf::frontend::Parser;
use xrcf::init_subscriber;
use xrcf::shared::SharedExt;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::Passes;
use xrcf::TransformOptions;

/// An example XRCF compiler that contains all default passes
#[derive(Args, Debug)]
#[command(version, about)]
struct XRCFArgs {
    /// The input file (- is interpreted as stdin)
    #[arg(default_value = "-")]
    input: String,
    /// Print debug information
    #[arg(long, name = "debug")]
    debug: bool,
}

fn cli() -> Command {
    let cli = Command::new("xrcf").args(xrcf::default_arguments());
    XRCFArgs::augment_args(cli)
}

fn remove_comments(src: &str) -> String {
    src.lines().filter(|line| !line.starts_with("//")).collect()
}

fn parse_and_transform(src: &str, options: &TransformOptions) -> String {
    let src = remove_comments(src);
    let module = Parser::<DefaultParserDispatch>::parse(&src).unwrap();
    let result = transform::<DefaultTransformDispatch>(module, options).unwrap();
    let result = match result {
        RewriteResult::Changed(op) => op.op.rd().to_string(),
        RewriteResult::Unchanged => src.to_string(),
    };
    result
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
    let passes = Passes::from_convert_args(args);
    let matches = cli.get_matches();
    let options = TransformOptions::from_args(matches.clone(), passes.clone());
    if matches.get_flag("debug") {
        init_tracing(tracing::Level::DEBUG);
    } else {
        init_tracing(tracing::Level::INFO);
    }

    let input = matches.get_one::<String>("input").unwrap();

    let input_text = if input == "-" {
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer).unwrap();
        buffer
    } else {
        std::fs::read_to_string(input).unwrap()
    };

    let result = parse_and_transform(&input_text, &options);
    println!("{result}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    fn run_app(args: Vec<&str>, input_text: &str) -> Result<String> {
        let cli = cli();
        let args_owned: Vec<String> = args.iter().map(|&s| s.to_string()).collect();
        let _matches = cli.try_get_matches_from(args_owned)?;
        let passes = Passes::from_convert_vec(args);
        let options = TransformOptions::from_args(_matches.clone(), passes.clone());
        let result = parse_and_transform(input_text, &options);
        Ok(result)
    }

    #[test]
    fn test_help() {
        let args = vec!["xrcf", "--help"];
        let result = run_app(args, "");
        let err = match result {
            Ok(_) => panic!("Expected an error"),
            Err(e) => e,
        };
        let result = err.to_string();
        println!("{result}");
        assert!(result.contains("Usage: xrcf"));
        assert!(result.contains("--convert-func-to-llvm"));
    }
}
