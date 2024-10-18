mod toy_to_mlir;

use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
pub use xrcf::compile;
use xrcf::convert::RewriteResult;
use xrcf::ir::Op;
use xrcf::CompilerDispatch;
use xrcf::DefaultCompilerDispatch;

pub struct ToyCompilerDispatch;

impl CompilerDispatch for ToyCompilerDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &str) -> Result<RewriteResult> {
        let result = DefaultCompilerDispatch::dispatch(op, pass)?;
        if let RewriteResult::Changed(_) = result {
            return Ok(result);
        }
        match pass {
            _ => return Err(anyhow::anyhow!("Unknown pass: {}", pass)),
        }
    }
}
