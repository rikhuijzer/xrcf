mod transform;

use clap::arg;
use clap::Arg;
use clap::ArgAction;
use clap::Args;
use clap::Command;

#[derive(Args, Debug)]
struct PythonArgs {
    /// Convert Python operations to MLIR
    #[arg(long)]
    convert_python_to_mlir: bool,
}

fn main() {
    let cli = Command::new("pythonc")
        .about("An example Python compiler (that can only compile a small subset of Python)")
        .arg(
            Arg::new("convert-unstable-to-mlir")
                .long("convert-unstable-to-mlir")
                .help("Convert unstable operations to MLIR")
                .action(ArgAction::SetTrue),
        );
    let cli = PythonArgs::augment_args(cli);

    let matches = cli.get_matches();
    println!(
        "unstable: {:?}",
        matches.get_flag("convert-unstable-to-mlir")
    );
    println!("python: {:?}", matches.get_flag("convert-python-to-mlir"));

    println!("Hello, world 2!");
}
