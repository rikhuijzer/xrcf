use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
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
use crate::ir::GuardedRegion;
use crate::ir::IntegerAttr;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::StringAttr;
use crate::ir::Value;
use crate::ir::Values;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct IfLowering;

fn new_block_from_region(
    region: Arc<RwLock<Region>>,
    parent: Arc<RwLock<Block>>,
) -> Result<Arc<RwLock<Block>>> {
    let parent = parent.parent().expect("Expected parent");
    let label = parent.unique_block_name();
    let arguments = Values::default();
    let ops = region.ops();
    let ops = Arc::new(RwLock::new(ops));
    let block = Block::new(Some(label), arguments, ops, Some(parent));
    let block = Arc::new(RwLock::new(block));
    Ok(block)
}

impl Rewrite for IfLowering {
    fn name(&self) -> &'static str {
        "scf_to_cf::IfLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::scf::IfOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let parent = op.operation().parent().expect("Expected parent");
        let op = op.as_any().downcast_ref::<dialect::scf::IfOp>().unwrap();

        let then = op.then().expect("Expected `then` region");
        let els = op.els().expect("Expected `else` region");

        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        operation.set_operand(0, op.operation().operand(0).clone().unwrap());
        let new = dialect::cf::CondBranchOp::from_operation(operation);
        let new = Arc::new(RwLock::new(new));
        op.replace(new.clone());

        Ok(RewriteResult::Changed(ChangedOp::new(new)))
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
