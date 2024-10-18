use crate::canonicalize::Canonicalize;
use crate::convert::ConvertFuncToLLVM;
use crate::convert::ConvertMLIRToLLVMIR;
use crate::convert::RewriteResult;
use crate::ir::Op;
use crate::Pass;
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
pub trait CompilerDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &str) -> Result<RewriteResult>;
}

/// Default implementation of `CompilerDispatch`.
///
/// This implementation knows only passes that are implemented in xrcf.
pub struct DefaultCompilerDispatch;

pub fn init_subscriber(level: Level) -> Result<(), SetGlobalDefaultError> {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(level)
        .with_test_writer()
        .without_time()
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
}

impl CompilerDispatch for DefaultCompilerDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &str) -> Result<RewriteResult> {
        match pass {
            Canonicalize::NAME => Canonicalize::convert(op),
            ConvertFuncToLLVM::NAME => ConvertFuncToLLVM::convert(op),
            ConvertMLIRToLLVMIR::NAME => ConvertMLIRToLLVMIR::convert(op),
            _ => return Err(anyhow::anyhow!("Unknown pass: {}", pass)),
        }
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

pub fn compile<T: CompilerDispatch>(
    op: Arc<RwLock<dyn Op>>,
    arguments: &str,
) -> Result<RewriteResult> {
    let passes = parse_passes(arguments)?;
    for pass in passes {
        let pass = pass.trim_start_matches("--");
        return T::dispatch(op, pass);
    }
    Ok(RewriteResult::Unchanged)
}
