use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect::cf;
use crate::dialect::llvm;
use crate::ir::Op;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct BranchLowering;

impl Rewrite for BranchLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::BranchLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<cf::BranchOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<cf::BranchOp>().unwrap();
        let new_op = llvm::BranchOp::from_operation_arc(op.operation().clone());
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct CondBranchLowering;

impl Rewrite for CondBranchLowering {
    fn name(&self) -> &'static str {
        "func_to_llvm::CondBranchLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<cf::CondBranchOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<cf::CondBranchOp>().unwrap();
        let new_op = llvm::CondBranchOp::from_operation_arc(op.operation().clone());
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertCFToLLVM;

impl Pass for ConvertCFToLLVM {
    const NAME: &'static str = "convert-cf-to-llvm";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&BranchLowering, &CondBranchLowering];
        apply_rewrites(op, &rewrites)
    }
}
