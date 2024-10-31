use clap::arg;
use clap::Args;
use clap::Command;
use std::io::Read;
use xrcf::convert::RewriteResult;
use xrcf::parser::DefaultParserDispatch;
use xrcf::parser::Parser;
use xrcf::transform;
use xrcf::DefaultTransformDispatch;
use xrcf::Passes;

/// An example XRCF compiler that contains all default passes
#[derive(Args, Debug)]
#[command(version, about)]
struct XRCFArgs {
    /// The input file (- is interpreted as stdin)
    #[arg(default_value = "-")]
    input: String,
}

fn cli() -> Command {
    let cli = Command::new("xrcf").args(xrcf::default_passes());
    let cli = XRCFArgs::augment_args(cli);
    cli
}

fn parse_and_transform(src: &str, passes: &Passes) -> String {
    let module = Parser::<DefaultParserDispatch>::parse(src).unwrap();
    let result = transform::<DefaultTransformDispatch>(module, &passes).unwrap();
    let result = match result {
        RewriteResult::Changed(op) => op.0.try_read().unwrap().to_string(),
        RewriteResult::Unchanged => src.to_string(),
    };
    result
}

fn main() {
    let cli = cli();
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

    let result = parse_and_transform(&input_text, &passes);
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
        let result = parse_and_transform(input_text, &passes);
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
