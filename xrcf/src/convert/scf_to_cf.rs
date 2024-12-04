use crate::convert::apply_rewrites;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::dialect::llvm;
use crate::dialect::llvm::PointerType;
use crate::ir::APInt;
use crate::ir::Block;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOp;
use crate::ir::GuardedOperation;
use crate::ir::IntegerAttr;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::StringAttr;
use crate::ir::Value;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct IfLowering;

impl Rewrite for IfLowering {
    fn name(&self) -> &'static str {
        "scf_to_cf::IfLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        println!("foo");
        Ok(op.as_any().is::<dialect::scf::IfOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        todo!()
    }
}

pub struct ConvertSCFToCF;

impl Pass for ConvertSCFToCF {
    const NAME: &'static str = "convert-scf-to-cf";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&IfLowering];
        apply_rewrites(op, &rewrites)
    }
}
