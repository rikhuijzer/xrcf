mod transform;

use clap::arg;
use clap::Args;
use clap::Command;
use xrcf::Passes;

use crate::transform::parse_and_transform;

/// An example Python compiler that can compile a small subset of Python
#[derive(Args, Debug)]
#[command(version, about)]
struct PythonArgs {
    /// The name of the input file
    input: String,
    /// Convert Python operations to MLIR
    #[arg(long, name = "convert-python-to-mlir")]
    convert_python_to_mlir: bool,
}

fn main() {
    let cli = Command::new("pythonc").args(xrcf::default_passes());
    let cli = PythonArgs::augment_args(cli);

    let matches = cli.get_matches();
    let passes = Passes::from_convert_args(&matches);

    let input = matches.get_one::<String>("input").unwrap();
    let input = std::fs::read_to_string(input).unwrap();

    parse_and_transform(&input, &passes).unwrap();
}
