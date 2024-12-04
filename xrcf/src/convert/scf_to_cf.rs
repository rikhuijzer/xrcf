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
use crate::ir::BlockDest;
use crate::ir::BlockLabel;
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
use core::net;
use std::sync::Arc;
use std::sync::RwLock;

struct IfLowering;

fn add_block_from_ops(
    ops: Vec<Arc<RwLock<dyn Op>>>,
    parent_region: Arc<RwLock<Region>>,
) -> Result<Arc<RwLock<OpOperand>>> {
    let unset_block = parent_region.add_empty_block();
    let block = unset_block.set_parent(Some(parent_region.clone()));
    block.set_ops(Arc::new(RwLock::new(ops)));
    let label = format!("^{}", parent_region.unique_block_name());
    block.set_label(Some(label.clone()));
    let label = BlockLabel::new(label);
    let label = Value::BlockLabel(label);
    let label = Arc::new(RwLock::new(label));
    let operand = OpOperand::new(label);
    let operand = Arc::new(RwLock::new(operand));
    Ok(operand)
}

fn add_merge_and_exit_blocks(parent_region: Arc<RwLock<Region>>) -> Result<()> {
    let unset_block = parent_region.add_empty_block();
    let block = unset_block.set_parent(Some(parent_region.clone()));
    let merge_label = format!("^{}", parent_region.unique_block_name());
    block.set_label(Some(merge_label.clone()));

    let unset_block = parent_region.add_empty_block();
    let return_block = unset_block.set_parent(Some(parent_region.clone()));
    let return_label = format!("^{}", parent_region.unique_block_name());
    return_block.set_label(Some(return_label.clone()));

    let mut operation = Operation::default();
    operation.set_parent(Some(block.clone()));
    let mut merge_op = dialect::cf::BranchOp::from_operation(operation);
    merge_op.set_dest(Some(Arc::new(RwLock::new(BlockDest::new(&return_label)))));
    let merge_op = Arc::new(RwLock::new(merge_op));
    block.set_ops(Arc::new(RwLock::new(vec![merge_op.clone()])));
    Ok(())
}

/// Add blocks for the `then` and `els` regions of `scf.if`.
///
/// For example, this rewrites:
/// ```mlir
///   %result = scf.if %0 -> (i32) {
///     %c1_i32 = arith.constant 3 : i32
///     scf.yield %c1_i32 : i32
///   } else {
///     %c2_i32 = arith.constant 4 : i32
///     scf.yield %c2_i32 : i32
///   }
///   return %result : i32
/// ```
/// to
/// ```mlir
///   %result = cf.cond_br %0, ^bb1, ^bb2
/// ^bb1:
///   %1 = arith.constant 3 : i32
///   cf.br ^bb3(%1 : i32)
/// ^bb2:
///   %2 = arith.constant 4 : i32
///   cf.br ^bb3(%2 : i32)
/// ^bb3(%0 : i32):
///   cf.br ^bb4
/// ^bb4:
///   return %0 : i32
/// ```
fn add_blocks(
    then: Arc<RwLock<Region>>,
    els: Arc<RwLock<Region>>,
    parent_region: Arc<RwLock<Region>>,
) -> Result<(Arc<RwLock<OpOperand>>, Arc<RwLock<OpOperand>>)> {
    let then_label = add_block_from_ops(then.ops(), parent_region.clone())?;
    let else_label = add_block_from_ops(els.ops(), parent_region.clone())?;
    add_merge_and_exit_blocks(parent_region.clone())?;
    Ok((then_label, else_label))
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
        let parent_region = parent.parent().expect("Expected parent region");
        let op = op.as_any().downcast_ref::<dialect::scf::IfOp>().unwrap();

        let then = op.then().expect("Expected `then` region");
        let els = op.els().expect("Expected `else` region");
        let (then_label, else_label) =
            add_blocks(then.clone(), els.clone(), parent_region.clone())?;

        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        operation.set_operand(0, op.operation().operand(0).clone().unwrap());
        operation.set_operand(1, then_label.clone());
        operation.set_operand(2, else_label.clone());
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
