use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::ir::Block;
use crate::ir::BlockArgument;
use crate::ir::BlockArgumentName;
use crate::ir::BlockName;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Users;
use crate::ir::Value;
use crate::ir::Values;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;

/// Lower `scf.if` to `cf.cond_br`.
///
/// For example, this rewrites:
/// ```mlir
///   %result = scf.if %0 -> (i32) {
///     %1 = arith.constant 3 : i32
///     scf.yield %c1_i32 : i32
///   } else {
///     %2 = arith.constant 4 : i32
///     scf.yield %2 : i32
///   }
/// ```
/// to
/// ```mlir
///   cf.cond_br %0, ^bb1, ^bb2
/// ^bb1:
///   %1 = arith.constant 3 : i32
///   cf.br ^bb3(%1 : i32)
/// ^bb2:
///   %2 = arith.constant 4 : i32
///   cf.br ^bb3(%2 : i32)
/// ^bb3(%result : i32):
///   cf.br ^bb4
/// ^bb4:
///   return %result : i32
/// ```
///
/// This lowering is similar to the following rewrite method in MLIR:
/// ```cpp
/// LogicalResult IfLowering::matchAndRewrite
/// ```
struct IfLowering;

fn lower_yield_op(op: &dialect::scf::YieldOp, after: Shared<Block>) -> Result<Shared<dyn Op>> {
    let operation = op.operation();
    let var = operation.rd().operand(0).unwrap();
    let operand = Shared::new(OpOperand::from_block(after).into());
    operation.wr().set_operand(0, operand);
    operation.wr().set_operand(1, var);
    let new_op = dialect::cf::BranchOp::from_operation_arc(operation.clone());
    let new_op = Shared::new(new_op.into());
    Ok(new_op)
}

fn branch_op(after: Shared<Block>) -> Shared<dyn Op> {
    let operation = Operation::default();
    let mut new_op = dialect::cf::BranchOp::from_operation(operation);
    let operand = Shared::new(OpOperand::from_block(after).into());
    new_op.set_dest(operand);
    Shared::new(new_op.into())
}

/// Add a `cf.br` to the end of `block` with destination `after`.
fn add_branch_to_after(block: Shared<Block>, after: Shared<Block>) {
    let ops = block.rd().ops();
    let mut ops = ops.wr();
    let ops_clone = ops.clone();
    let last_op = ops_clone.last().unwrap();
    let last_op = last_op.rd();
    let yield_op = last_op.as_any().downcast_ref::<dialect::scf::YieldOp>();
    if let Some(yield_op) = yield_op {
        let new_op = lower_yield_op(yield_op, after.clone()).unwrap();
        ops.pop();
        ops.push(new_op.clone());
    } else {
        let new_op = branch_op(after.clone());
        new_op.rd().set_parent(block.clone());
        ops.push(new_op.clone());
    };
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
fn move_successors_to_exit_block(op: &dialect::scf::IfOp, exit_block: Shared<Block>) -> Result<()> {
    let if_op_parent = op.operation().rd().parent().expect("Expected parent");
    let if_op_index = if_op_parent
        .rd()
        .index_of(&op.operation().rd())
        .expect("Expected index");
    let ops = if_op_parent.rd().ops();
    let mut ops = ops.wr();
    let return_ops = ops[if_op_index + 1..].to_vec();
    for op in return_ops.iter() {
        let op = op.rd();
        op.set_parent(exit_block.clone());
    }
    exit_block.wr().set_ops(Shared::new(return_ops.into()));
    ops.drain(if_op_index + 1..);
    Ok(())
}

fn add_merge_block(
    parent_region: Shared<Region>,
    results: Values,
    exit: Shared<Block>,
) -> Result<(Shared<Block>, Values)> {
    let unset_block = parent_region.rd().add_empty_block_before(exit.clone());
    let merge = unset_block.set_parent(Some(parent_region.clone()));
    let merge_block_arguments = as_block_arguments(results, merge.clone())?;
    merge.wr().set_arguments(merge_block_arguments.clone());
    merge.wr().set_label(BlockName::Unset);

    let mut operation = Operation::default();
    operation.set_parent(Some(merge.clone()));
    let mut merge_op = dialect::cf::BranchOp::from_operation(operation);

    let operand = Shared::new(OpOperand::from_block(exit).into());
    merge_op.set_dest(operand);

    let merge_op: Shared<dyn Op> = Shared::new(merge_op.into());
    merge
        .wr()
        .set_ops(Shared::new(vec![merge_op.clone()].into()));
    Ok((merge, merge_block_arguments))
}

fn add_exit_block(op: &dialect::scf::IfOp, parent_region: Shared<Region>) -> Result<Shared<Block>> {
    let unset_block = parent_region.rd().add_empty_block();
    let exit = unset_block.set_parent(Some(parent_region.clone()));
    exit.wr().set_label(BlockName::Unset);
    move_successors_to_exit_block(op, exit.clone())?;
    Ok(exit)
}

/// Convert [OpResult]s to [BlockArgument]s.
///
/// Necessary to translate `%result = scf.if` to `^merge:(%result)`.
fn as_block_arguments(results: Values, parent: Shared<Block>) -> Result<Values> {
    let mut out = vec![];
    for result in results.into_iter() {
        let result_rd = result.rd();
        let name = result_rd.name();
        let typ = result_rd.typ().unwrap();
        let name = BlockArgumentName::Name(name.unwrap());
        let mut arg = BlockArgument::new(name, typ);
        arg.set_parent(Some(parent.clone()));
        let arg = Value::BlockArgument(arg);
        let arg = Shared::new(arg.into());
        out.push(arg);
    }
    Ok(Values::from_vec(out))
}

fn results_users(results: Values) -> Vec<Users> {
    let mut out = vec![];
    for result in results.into_iter() {
        out.push(result.rd().users());
    }
    out
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
    parent_region: Shared<Region>,
) -> Result<(Shared<Block>, Shared<Block>)> {
    let results = op.operation().rd().results();
    let results_users = results_users(results.clone());
    let exit = add_exit_block(op, parent_region.clone())?;
    let has_results = !results.is_empty();

    let then_region = op.then().expect("Expected `then` region");
    let then = then_region.rd().blocks().into_iter().next().unwrap();
    then.wr().set_label(BlockName::Unset);
    exit.rd().inline_region_before(then_region.clone());

    let else_region = op.els().expect("Expected `else` region");
    let els = else_region.rd().blocks().into_iter().next().unwrap();
    els.wr().set_label(BlockName::Unset);
    exit.rd().inline_region_before(else_region.clone());

    let after = if has_results {
        let (merge, merge_block_arguments) =
            add_merge_block(parent_region.clone(), results.clone(), exit.clone())?;
        let merge_block_arguments = merge_block_arguments.vec();
        let merge_block_arguments = merge_block_arguments.rd();

        assert!(results_users.len() == merge_block_arguments.len());
        for i in 0..results_users.len() {
            let users = &results_users[i];
            let users = match users {
                Users::OpOperands(users) => users,
                Users::HasNoOpResults => &vec![],
            };
            println!("users.len: {}", users.len());
            let arg = merge_block_arguments[i].clone();
            for user in users.iter() {
                let mut user = user.wr();
                user.set_value(arg.clone());
            }
        }
        merge
    } else {
        exit.clone()
    };

    add_branch_to_after(then.clone(), after.clone());
    add_branch_to_after(els.clone(), after.clone());

    Ok((then, els))
}

impl Rewrite for IfLowering {
    fn name(&self) -> &'static str {
        "scf_to_cf::IfLowering"
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let parent = op.operation().rd().parent().expect("Expected parent");
        let parent_region = parent.rd().parent().expect("Expected parent region");
        let op = match op.as_any().downcast_ref::<dialect::scf::IfOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };

        let (then, els) = add_blocks(op, parent_region.clone())?;

        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        operation.set_operand(0, op.operation().rd().operand(0).clone().unwrap());
        let then_operand = Shared::new(OpOperand::from_block(then).into());
        operation.set_operand(1, then_operand);
        let els_operand = Shared::new(OpOperand::from_block(els).into());
        operation.set_operand(2, els_operand);
        let new = dialect::cf::CondBranchOp::from_operation(operation);
        let new = Shared::new(new.into());
        op.replace(new.clone());
        // `replace` moves the results of the old op to the new op, but
        // `cf.cond_br` should not have results.
        new.rd().operation().wr().set_results(Values::default());

        Ok(RewriteResult::Changed(ChangedOp::new(new)))
    }
}

pub struct ConvertSCFToCF;

impl Pass for ConvertSCFToCF {
    const NAME: &'static str = "convert-scf-to-cf";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&IfLowering];
        apply_rewrites(op, &rewrites)
    }
}
