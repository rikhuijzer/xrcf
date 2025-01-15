use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect::cf;
use crate::dialect::llvm;
use crate::ir::Op;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;

struct BranchLowering;

impl Rewrite for BranchLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::BranchLowering"
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<cf::BranchOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let new_op = llvm::BranchOp::from_operation_arc(op.operation().clone());
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct CondBranchLowering;

impl Rewrite for CondBranchLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::CondBranchLowering"
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<cf::CondBranchOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let new_op = llvm::CondBranchOp::from_operation_arc(op.operation().clone());
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertCFToLLVM;

impl Pass for ConvertCFToLLVM {
    const NAME: &'static str = "convert-cf-to-llvm";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&BranchLowering, &CondBranchLowering];
        apply_rewrites(op, &rewrites)
    }
}
