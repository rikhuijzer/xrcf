use crate::ir::Op;
use crate::rewrite::Rewrite;
use crate::rewrite::RewriteResult;
use anyhow::Result;

pub struct ConvertFuncToLLVM {}

impl Rewrite for ConvertFuncToLLVM {
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, op: &dyn Op) -> Result<RewriteResult> {
        Ok(RewriteResult::Unchanged)
    }
}
