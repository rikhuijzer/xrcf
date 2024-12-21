use crate::canonicalize::Canonicalize;
use crate::convert::ConvertCFToLLVM;
use crate::convert::ConvertExperimentalToMLIR;
use crate::convert::ConvertFuncToLLVM;
use crate::convert::ConvertMLIRToLLVMIR;
use crate::convert::ConvertSCFToCF;
use crate::convert::Pass;
use crate::convert::RewriteResult;
use crate::ir::Op;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use clap::Arg;
use clap::ArgAction;
use clap::ArgMatches;
use std::env::ArgsOs;
use std::fmt;
use std::fmt::Display;
use tracing::subscriber::SetGlobalDefaultError;
use tracing::Level;

/// A transformation pass (e.g., `--convert-func-to-llvm`).
#[derive(Clone)]
pub struct SinglePass {
    pass: String,
}

impl Display for SinglePass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pass)
    }
}

impl SinglePass {
    pub fn new(pass: &str) -> SinglePass {
        let pass = if pass.starts_with("--") {
            pass.split("--").last().unwrap()
        } else {
            pass
        };
        SinglePass {
            pass: pass.to_string(),
        }
    }
}

/// A collection of [SinglePass]es.
#[derive(Clone)]
pub struct Passes {
    passes: Vec<SinglePass>,
}

impl Display for Passes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.passes
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<String>>()
                .join(" ")
        )
    }
}

impl Passes {
    pub fn from_vec(passes: Vec<&str>) -> Passes {
        Passes {
            passes: passes.iter().map(|p| SinglePass::new(p)).collect(),
        }
    }
    pub fn from_convert_vec(args: Vec<&str>) -> Passes {
        let mut passes = vec![];
        for arg in args {
            if arg.starts_with("--convert-") || arg.starts_with("--canonicalize") {
                passes.push(arg);
            }
        }
        Passes::from_vec(passes)
    }
    /// Extract passes (starting with `--convert-` or `--canonicalize`) from the
    /// given args.
    pub fn from_convert_args(args: ArgsOs) -> Passes {
        let mut passes = vec![];
        for arg in args {
            let arg = arg.to_string_lossy();
            if arg.starts_with("--convert-") || arg.starts_with("--canonicalize") {
                passes.push(SinglePass::new(&arg));
            }
        }
        Passes { passes }
    }
    pub fn vec(&self) -> &Vec<SinglePass> {
        &self.passes
    }
}

pub struct TransformOptions {
    passes: Passes,
    /// Print the IR before each pass.
    ///
    /// `print-ir-after-all` is not implemented because I don't see when it
    /// would be useful. `print-ir-before-all` is useful to see the IR after
    /// parsing, but seeing the IR after the last pass is already the final
    /// output.
    print_ir_before_all: bool,
    writer: Shared<dyn std::io::Write + Send>,
}

impl TransformOptions {
    pub fn from_args(matches: ArgMatches, passes: Passes) -> TransformOptions {
        let print_ir_before_all = matches.get_flag("print-ir-before-all");
        TransformOptions {
            passes,
            print_ir_before_all,
            writer: Shared::new(std::io::stderr().into()),
        }
    }
    pub fn from_passes(passes: Passes) -> TransformOptions {
        TransformOptions {
            passes,
            print_ir_before_all: false,
            writer: Shared::new(std::io::stderr().into()),
        }
    }
    pub fn print_ir_before_all(&self) -> bool {
        self.print_ir_before_all
    }
    pub fn passes(&self) -> &Passes {
        &self.passes
    }
    pub fn set_print_ir_before_all(&mut self, print_ir_before_all: bool) {
        self.print_ir_before_all = print_ir_before_all;
    }
    pub fn set_writer(&mut self, writer: Shared<dyn std::io::Write + Send>) {
        self.writer = writer;
    }
}

/// Interface to add custom passes to the compiler.
pub trait TransformDispatch {
    fn dispatch(op: Shared<dyn Op>, pass: &SinglePass) -> Result<RewriteResult>;
}

/// Default implementation of [TransformDispatch].
///
/// This default implementation knows only passes that are implemented in xrcf.
pub struct DefaultTransformDispatch;

/// Initialize logging with the given level.
pub fn init_subscriber(level: Level) -> Result<(), SetGlobalDefaultError> {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(level)
        .with_test_writer()
        .without_time()
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
}

impl TransformDispatch for DefaultTransformDispatch {
    fn dispatch(op: Shared<dyn Op>, pass: &SinglePass) -> Result<RewriteResult> {
        let pass = pass.to_string();
        match pass.as_str() {
            Canonicalize::NAME => Canonicalize::convert(op.clone()),
            ConvertCFToLLVM::NAME => ConvertCFToLLVM::convert(op.clone()),
            ConvertExperimentalToMLIR::NAME => ConvertExperimentalToMLIR::convert(op.clone()),
            ConvertFuncToLLVM::NAME => ConvertFuncToLLVM::convert(op.clone()),
            ConvertMLIRToLLVMIR::NAME => ConvertMLIRToLLVMIR::convert(op.clone()),
            ConvertSCFToCF::NAME => ConvertSCFToCF::convert(op.clone()),
            _ => Err(anyhow::anyhow!("Unknown pass: {}", pass)),
        }
    }
}

/// Default arguments that are available in xrcf.
///
/// This includes options such as `--print-ir-before-all`, but also the default
/// passes such as `--convert-func-to-llvm`. `--debug` is not included to allow
/// downstream projects to handle the logging differently.
///
/// For an example on how to use this, see the usage in the `arnoldc/` directory.
pub fn default_arguments() -> Vec<Arg> {
    vec![
        Arg::new("convert-scf-to-cf")
            .long("convert-scf-to-cf")
            .help("Convert structured control flow (scf) operations to cf")
            .action(ArgAction::SetTrue),
        Arg::new("convert-cf-to-llvm")
            .long("convert-cf-to-llvm")
            .help("Convert control flow (cf) operations to LLVM")
            .action(ArgAction::SetTrue),
        Arg::new("convert-experimental-to-mlir")
            .long("convert-experimental-to-mlir")
            .help("Convert experimental operations to MLIR")
            .action(ArgAction::SetTrue),
        Arg::new("convert-func-to-llvm")
            .long("convert-func-to-llvm")
            .help("Convert function operations to LLVM")
            .action(ArgAction::SetTrue),
        Arg::new("convert-mlir-to-llvmir")
            .long("convert-mlir-to-llvmir")
            .help("Convert MLIR to LLVM IR")
            .action(ArgAction::SetTrue),
        Arg::new("print-ir-before-all")
            .long("print-ir-before-all")
            .help("Print the IR before each pass")
            .action(ArgAction::SetTrue),
    ]
}

/// Transform the given operation via given passen.
///
/// This is the main function that most users will interact with. The name
/// "transform" is used instead of "compile" because the infrastructure is not
/// limited to compiling. For example, it could also be used to build
/// decompilers (i.e., for security research where the assembly is decompiled to
/// a more readable form).
pub fn transform<T: TransformDispatch>(
    op: Shared<dyn Op>,
    options: &TransformOptions,
) -> Result<RewriteResult> {
    let mut result = RewriteResult::Unchanged;
    for pass in options.passes().vec() {
        if options.print_ir_before_all() {
            let op = op.rd();
            writeln!(
                &mut *options.writer.wr(),
                "// ----- // IR Dump before {pass} //----- //\n{op}\n\n",
            )?;
        }
        let new_result = T::dispatch(op.clone(), pass)?;
        if let RewriteResult::Changed(_) = new_result {
            result = new_result;
        }
    }

    Ok(result)
}
