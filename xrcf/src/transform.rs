use crate::canonicalize::Canonicalize;
use crate::convert::ConvertFuncToLLVM;
use crate::convert::ConvertMLIRToLLVMIR;
use crate::convert::RewriteResult;
use crate::ir::Op;
use crate::convert::Pass;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use tracing::subscriber::SetGlobalDefaultError;
use tracing::Level;
use tracing_subscriber;

/// Interface to add custom passes to the compiler.
///
/// Downstream crates can implement this trait to add custom passes to their
/// compiler.  This is similar to `ParserDispatch`.
pub trait TransformDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, passes: Vec<&str>) -> Result<RewriteResult>;
}

/// Default implementation of [TransformDispatch].
///
/// This implementation knows only passes that are implemented in xrcf.
pub struct DefaultTransformDispatch;

pub struct TransformOptions {
    canonicalize: bool,
    convert_func_to_llvm: bool,
    mlir_to_llvmir: bool,
}

impl TransformOptions {
    pub fn from_str(flags: &str) -> Result<TransformOptions> {
        let mut options = TransformOptions::default();
        let flags = flags.split(' ').collect::<Vec<&str>>();
        if flags.len() == 0 {
            return Ok(options);
        } else if 1 < flags.len() {
            panic!("Multiple flags not yet supported");
        }
        for flag in flags {
            match flag {
                "--canonicalize" => options.canonicalize = true,
                "--convert-func-to-llvm" => options.convert_func_to_llvm = true,
                "--convert-mlir-to-llvmir" => options.mlir_to_llvmir = true,
                _ => return Err(anyhow::anyhow!("Invalid flag: {}", flag)),
            }
        }
        Ok(options)
    }
    pub fn set_canonicalize(&mut self, canonicalize: bool) {
        self.canonicalize = canonicalize;
    }
    pub fn set_convert_func_to_llvm(&mut self, convert_func_to_llvm: bool) {
        self.convert_func_to_llvm = convert_func_to_llvm;
    }
    pub fn set_mlir_to_llvmir(&mut self, mlir_to_llvmir: bool) {
        self.mlir_to_llvmir = mlir_to_llvmir;
    }
}

impl Default for TransformOptions {
    fn default() -> Self {
        TransformOptions {
            canonicalize: false,
            convert_func_to_llvm: false,
            mlir_to_llvmir: false,
        }
    }
}

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
    fn dispatch(op: Arc<RwLock<dyn Op>>, passes: Vec<&str>) -> Result<RewriteResult> {
        for pass in passes {
            let pass = pass.trim_start_matches("--");
            return match pass {
                Canonicalize::NAME => Canonicalize::convert(op),
                ConvertFuncToLLVM::NAME => ConvertFuncToLLVM::convert(op),
                ConvertMLIRToLLVMIR::NAME => ConvertMLIRToLLVMIR::convert(op),
                _ => return Err(anyhow::anyhow!("Unknown pass: {}", pass)),
            };
        }
        Ok(RewriteResult::Unchanged)
    }
}

fn parse_passes(arguments: &str) -> Result<Vec<&str>> {
    let mut passes = vec![];
    let arguments = arguments.split(' ').collect::<Vec<&str>>();
    for argument in arguments {
        passes.push(argument);
    }
    Ok(passes)
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
    arguments: &str,
) -> Result<RewriteResult> {
    let passes = parse_passes(arguments)?;
    T::dispatch(op, passes)
}
