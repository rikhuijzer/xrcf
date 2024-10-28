use crate::canonicalize::Canonicalize;
use crate::convert::ConvertFuncToLLVM;
use crate::convert::ConvertMLIRToLLVMIR;
use crate::convert::ConvertUnstableToMLIR;
use crate::convert::Pass;
use crate::convert::RewriteResult;
use crate::ir::Op;
use anyhow::Result;
use clap::Arg;
use clap::ArgAction;
use clap::ArgMatches;
use std::sync::Arc;
use std::sync::RwLock;
use tracing::subscriber::SetGlobalDefaultError;
use tracing::Level;
use tracing_subscriber;

pub struct Passes {
    passes: Vec<String>,
}

impl Passes {
    /// Extract passes (starting with `--convert-`) from the given args.
    pub fn from_convert_args(matches: &ArgMatches) -> Passes {
        let mut passes = vec![];
        for id in matches.ids() {
            println!("id: {}", id);
            if id.to_string().starts_with("convert-") {
                passes.push(id.to_string());
            }
        }
        Passes { passes }
    }
    pub fn from_vec(passes: Vec<String>) -> Passes {
        Passes { passes }
    }
    pub fn vec(&self) -> &Vec<String> {
        &self.passes
    }
}

/// Interface to add custom passes to the compiler.
///
/// Downstream crates can implement this trait to add custom passes to their
/// compiler.  This is similar to `ParserDispatch`.
pub trait TransformDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, passes: &Passes) -> Result<RewriteResult>;
}

/// Default implementation of [TransformDispatch].
///
/// This implementation knows only passes that are implemented in xrcf.
pub struct DefaultTransformDispatch;

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
    fn dispatch(op: Arc<RwLock<dyn Op>>, passes: &Passes) -> Result<RewriteResult> {
        let mut result = RewriteResult::Unchanged;
        for pass in passes.vec() {
            let pass = pass.trim_start_matches("--");
            let new_result = match pass {
                Canonicalize::NAME => Canonicalize::convert(op.clone())?,
                ConvertFuncToLLVM::NAME => ConvertFuncToLLVM::convert(op.clone())?,
                ConvertMLIRToLLVMIR::NAME => ConvertMLIRToLLVMIR::convert(op.clone())?,
                ConvertUnstableToMLIR::NAME => ConvertUnstableToMLIR::convert(op.clone())?,
                _ => return Err(anyhow::anyhow!("Unknown pass: {}", pass)),
            };
            if let RewriteResult::Changed(_) = new_result {
                result = new_result;
            }
        }
        Ok(result)
    }
}

pub fn default_passes() -> Vec<Arg> {
    vec![
        Arg::new("convert-unstable-to-mlir")
            .long("convert-unstable-to-mlir")
            .help("Convert unstable operations to MLIR")
            .action(ArgAction::SetTrue),
        Arg::new("convert-func-to-llvm")
            .long("convert-func-to-llvm")
            .help("Convert function operations to LLVM IR")
            .action(ArgAction::SetTrue),
        Arg::new("convert-mlir-to-llvmir")
            .long("convert-mlir-to-llvmir")
            .help("Convert MLIR to LLVM IR")
            .action(ArgAction::SetTrue),
    ]
}

/// Transform the given operation via given passen.
///
/// This is the main function that most users will interact with. The name
/// `transform` is used instead of `compile` because the infrastructure is not
/// limited to compiling. For example, it could also be used to build
/// decompilers (i.e., for security research where the assembly is decompiled to
/// a more readable form.). In MLIR, this function is called `opt`, but clearly
/// that is too restrictive too since a set of passes don't have to optimize the
/// code.
pub fn transform<T: TransformDispatch>(
    op: Arc<RwLock<dyn Op>>,
    passes: &Passes,
) -> Result<RewriteResult> {
    T::dispatch(op, &passes)
}
