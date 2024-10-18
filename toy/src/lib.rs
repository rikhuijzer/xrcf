use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
pub use xrcf::compile;
use xrcf::convert::RewriteResult;
use xrcf::ir::Op;
use xrcf::CompilerDispatch;

struct ToyCompilerDispatch;

impl CompilerDispatch for ToyCompilerDispatch {
    fn dispatch(op: Arc<RwLock<dyn Op>>, pass: &str) -> Result<RewriteResult> {
        Ok(RewriteResult::Unchanged)
    }
}
