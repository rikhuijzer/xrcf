use crate::ir::Op;
use crate::rewrite::apply_rewrites;
use crate::rewrite::Rewrite;
use crate::rewrite::RewriteResult;
use crate::Pass;
use anyhow::Result;

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn is_match(&self, _op: &dyn Op) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, _op: &dyn Op) -> Result<RewriteResult> {
        Ok(RewriteResult::Unchanged)
    }
}

struct ConstantOpLowering;

impl Rewrite for ConstantOpLowering {
    fn is_match(&self, _op: &dyn Op) -> Result<bool> {
        Ok(true)
    }
    fn rewrite(&self, _op: &dyn Op) -> Result<RewriteResult> {
        Ok(RewriteResult::Unchanged)
    }
}

pub struct ConvertFuncToLLVM;

impl Pass for ConvertFuncToLLVM {
    fn name() -> &'static str {
        "convert-func-to-llvm"
    }
    fn convert(op: &dyn Op) -> Result<()> {
        let rewrites: Vec<&dyn Rewrite> = vec![&ConstantOpLowering];
        apply_rewrites(op, &rewrites)?;
        Ok(())
    }
}
