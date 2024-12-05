use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::ir::Block;
use crate::ir::BlockArgument;
use crate::ir::BlockDest;
use crate::ir::BlockLabel;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOp;
use crate::ir::GuardedOperation;
use crate::ir::GuardedRegion;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Value;
use crate::ir::Values;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct IfLowering;

fn lower_yield_op(op: &dialect::scf::YieldOp, merge_label: String) -> Result<Arc<RwLock<dyn Op>>> {
    let mut new_op = dialect::cf::BranchOp::from_operation_arc(op.operation().clone());
    new_op.set_dest(Some(Arc::new(RwLock::new(BlockDest::new(&merge_label)))));
    let new_op = Arc::new(RwLock::new(new_op));
    Ok(new_op)
}

fn add_block_from_region(
    label: String,
    merge_label: String,
    region: Arc<RwLock<Region>>,
    parent_region: Arc<RwLock<Region>>,
) -> Result<Arc<RwLock<OpOperand>>> {
    let mut ops = region.ops();
    let ops_clone = ops.clone();
    let last_op = ops_clone.last().unwrap();
    let last_op = last_op.try_read().unwrap();
    let yield_op = last_op
        .as_any()
        .downcast_ref::<dialect::scf::YieldOp>();
    if let Some(yield_op) = yield_op {
        let new_op = lower_yield_op(&yield_op, merge_label)?;
        ops.pop();
        ops.push(new_op.clone());
    }

    let unset_block = parent_region.add_empty_block();
    let block = unset_block.set_parent(Some(parent_region.clone()));
    block.set_ops(Arc::new(RwLock::new(ops)));
    block.set_label(Some(label.clone()));
    let label = BlockLabel::new(label);
    let label = Value::BlockLabel(label);
    let label = Arc::new(RwLock::new(label));
    let operand = OpOperand::new(label);
    let operand = Arc::new(RwLock::new(operand));
    Ok(operand)
}

/// Move all successors of `scf.if` to the return block.
///
/// This rewrites:
/// ```mlir
///   cf.cond_br %0, ^bb1, ^bb2
///   return %result : i32
/// ^bb1:
///   llvm.br ^bb3
/// ^bb2:
///   llvm.br ^bb3
/// ^bb3:
/// ```
/// to
/// ```mlir
///   cf.cond_br %0, ^bb1, ^bb2
/// ^bb1:
///   llvm.br ^bb3
/// ^bb2:
///   llvm.br ^bb3
/// ^bb3:
///   return %result : i32
/// ```
fn move_successors_to_exit_block(
    op: &dialect::scf::IfOp,
    exit_block: Arc<RwLock<Block>>,
) -> Result<()> {
    let if_op_parent = op.operation().parent().expect("Expected parent");
    let if_op_index = if_op_parent
        .index_of(&op.operation().try_read().unwrap())
        .expect("Expected index");
    let ops = if_op_parent.ops();
    let mut ops = ops.try_write().unwrap();
    let return_ops = ops[if_op_index + 1..].to_vec();
    exit_block.set_ops(Arc::new(RwLock::new(return_ops)));
    ops.drain(if_op_index + 1..);
    Ok(())
}

fn add_merge_block(
    parent_region: Arc<RwLock<Region>>,
    merge_label: String,
    results: Values,
    exit_label: String,
) -> Result<()> {
    let unset_block = parent_region.add_empty_block();
    let block = unset_block.set_parent(Some(parent_region.clone()));
    block.set_label(Some(merge_label.clone()));
    let merge_block_operands = as_block_arguments(results, block.clone())?;
    block.set_arguments(merge_block_operands);

    let mut operation = Operation::default();
    operation.set_parent(Some(block.clone()));
    let mut merge_op = dialect::cf::BranchOp::from_operation(operation);
    merge_op.set_dest(Some(Arc::new(RwLock::new(BlockDest::new(&exit_label)))));
    let merge_op = Arc::new(RwLock::new(merge_op));
    block.set_ops(Arc::new(RwLock::new(vec![merge_op.clone()])));
    Ok(())
}

fn add_exit_block(
    op: &dialect::scf::IfOp,
    parent_region: Arc<RwLock<Region>>,
    exit_label: String,
) -> Result<()> {
    let unset_block = parent_region.add_empty_block();
    let exit_block = unset_block.set_parent(Some(parent_region.clone()));
    exit_block.set_label(Some(exit_label.clone()));

    move_successors_to_exit_block(op, exit_block)?;
    Ok(())
}

/// Convert [OpResult]s to [BlockArgument]s.
///
/// Necessary to translate `%result = scf.if` to `^merge:(%result)`.
fn as_block_arguments(results: Values, parent: Arc<RwLock<Block>>) -> Result<Values> {
    let results = results.vec();
    let results = results.try_read().unwrap();
    let mut out = vec![];
    for result in results.iter() {
        let result = result.try_read().unwrap();
        let name = result.name();
        let typ = result.typ().unwrap();
        let mut arg = BlockArgument::new(name, typ);
        arg.set_parent(Some(parent.clone()));
        let arg = Value::BlockArgument(arg);
        let arg = Arc::new(RwLock::new(arg));
        out.push(arg);
    }
    Ok(Values::from_vec(out))
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
    op: &dialect::scf::IfOp,
    parent_region: Arc<RwLock<Region>>,
) -> Result<(Arc<RwLock<OpOperand>>, Arc<RwLock<OpOperand>>)> {
    let then_label = format!("^{}", parent_region.unique_block_name());
    let then_label_index = then_label
        .trim_start_matches("^bb")
        .parse::<usize>()
        .unwrap();
    let else_label = format!("^bb{}", then_label_index + 1);
    let merge_label = format!("^bb{}", then_label_index + 2);
    let exit_label = format!("^bb{}", then_label_index + 3);

    let then = op.then().expect("Expected `then` region");
    let els = op.els().expect("Expected `else` region");

    let then_label =
        add_block_from_region(then_label, merge_label.clone(), then, parent_region.clone())?;
    let else_label =
        add_block_from_region(else_label, merge_label.clone(), els, parent_region.clone())?;

    let results = op.operation().results();
    add_merge_block(
        parent_region.clone(),
        merge_label,
        results,
        exit_label.clone(),
    )?;
    add_exit_block(op, parent_region.clone(), exit_label)?;
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

        let (then_label, else_label) = add_blocks(&op, parent_region.clone())?;

        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        operation.set_operand(0, op.operation().operand(0).clone().unwrap());
        operation.set_operand(1, then_label.clone());
        operation.set_operand(2, else_label.clone());
        let new = dialect::cf::CondBranchOp::from_operation(operation);
        let new: Arc<RwLock<dyn Op>> = Arc::new(RwLock::new(new));
        op.replace(new.clone());
        // `replace` moves the results of the old op to the new op, but
        // `cf.cond_br` should not have results.
        new.operation().set_results(Values::default());

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
