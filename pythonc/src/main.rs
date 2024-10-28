mod transform;

use clap::arg;
use clap::Args;
use clap::Command;

/// An example Python compiler that can compile a small subset of Python
#[derive(Args, Debug)]
#[command(version, about)]
struct PythonArgs {
    /// The name of the input file
    input: String,
    /// Convert Python operations to MLIR
    #[arg(long)]
    convert_python_to_mlir: bool,
}

fn main() {
    let cli = Command::new("pythonc")
        .args(xrcf::default_passes());
    let cli = PythonArgs::augment_args(cli);

    let matches = cli.get_matches();
    let input = matches.get_one::<String>("input").unwrap();
    let input = std::fs::read_to_string(input).unwrap();
    println!("input: {}", input);
    println!(
        "unstable: {:?}",
        matches.get_flag("convert-unstable-to-mlir")
    );
    println!("python: {:?}", matches.get_flag("convert-python-to-mlir"));

    println!("Hello, world 2!");
}
