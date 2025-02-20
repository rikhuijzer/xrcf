use crate::canonicalize::DeadCodeElimination;
use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::ir;
use crate::ir::Block;
use crate::ir::BlockArgument;
use crate::ir::BlockArgumentName;
use crate::ir::Constant;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpResult;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::TypeConvert;
use crate::ir::Types;
use crate::ir::Users;
use crate::ir::Value;
use crate::ir::Values;
use crate::shared::Shared;
use crate::shared::SharedExt;
use crate::targ3t;
use anyhow::Result;
use std::str::FromStr;

struct AddLowering;

/// Return an [OpOperand] containing a [Constant].
///
/// Otherwise, return `None`.
fn constant_op_operand(operand: Shared<OpOperand>) -> Option<Shared<OpOperand>> {
    let op = operand.rd().defining_op();
    if let Some(op) = op {
        if op.rd().is_const() {
            let op = op.rd();
            let op = op.as_any().downcast_ref::<dialect::llvm::ConstantOp>();
            if let Some(op) = op {
                let value = op.value();
                let new_value = Constant::new(value.clone());
                let new_operand = Value::Constant(new_value);
                let new_operand = OpOperand::new(Shared::new(new_operand.into()));
                return Some(Shared::new(new_operand.into()));
            }
        }
    }
    None
}

/// Replace the operands that point to a constant operation by a [Constant].
fn replace_constant_operands(op: &dyn Op) {
    let operation = op.operation();
    let operands = operation.rd().operands().vec();
    let mut operands = operands.wr();
    for i in 0..operands.len() {
        let operand = &operands[i];
        let new_operand = constant_op_operand(operand.clone());
        if let Some(new_operand) = new_operand {
            operands[i] = new_operand;
        }
    }
}

impl Rewrite for AddLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::AddLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::AddOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation();
        let new_op = targ3t::llvmir::AddOp::from_operation_arc(operation.clone());
        replace_constant_operands(&new_op);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct AllocaLowering;

impl Rewrite for AllocaLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::AllocaLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::AllocaOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::AllocaOp::from_operation_arc(operation.clone());
        operation
            .rd()
            .results()
            .convert_types::<ConvertMLIRToLLVMIR>()?;
        new_op.set_element_type(op.element_type().unwrap());
        replace_constant_operands(&new_op);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

/// Replace `llvm.br` by `br`.
///
/// This is only executed once the `phi` node has been inserted by
/// [MergeLowering].
struct BranchLowering;

impl Rewrite for BranchLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::BranchLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::BranchOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let valid_values = op.operation().rd().operands().into_iter().all(|operand| {
            let operand = operand.rd();
            let value = &operand.value;
            let value = value.rd();
            matches!(&*value, Value::BlockLabel(_) | Value::BlockPtr(_))
        });
        if !valid_values {
            // [MergeLowering] has not yet removed the operands.
            return Ok(RewriteResult::Unchanged);
        }

        let operation = op.operation();
        let new_op = targ3t::llvmir::BranchOp::from_operation_arc(operation.clone());
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct CallLowering;

impl Rewrite for CallLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::CallLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::CallOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation();

        let mut new_op = targ3t::llvmir::CallOp::from_operation_arc(operation.clone());
        new_op.set_identifier(op.identifier().unwrap());
        let varargs = match op.varargs() {
            Some(varargs) => {
                let varargs = ConvertMLIRToLLVMIR::convert_type(&varargs)?;
                Some(varargs)
            }
            None => None,
        };
        new_op.set_varargs(varargs);
        replace_constant_operands(&new_op);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct CondBranchLowering;

impl Rewrite for CondBranchLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::CondBranchLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::CondBranchOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation();
        let new_op = targ3t::llvmir::BranchOp::from_operation_arc(operation.clone());
        replace_constant_operands(&new_op);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct FuncLowering;

fn lower_block_argument_types(operation: &mut Operation) {
    let new_arguments = {
        let arguments = operation.arguments();
        let arguments = arguments.vec();
        let arguments = arguments.rd();
        let mut new_arguments = vec![];
        for argument in arguments.iter() {
            if let Value::Variadic = &*argument.rd() {
                new_arguments.push(argument.clone());
            } else {
                let typ = argument.rd().typ().unwrap();
                let typ = typ.rd();
                if typ.as_any().is::<dialect::llvm::PointerType>() {
                    let typ = targ3t::llvmir::PointerType::from_str("ptr").unwrap();
                    let typ = Shared::new(typ.into());
                    let name = BlockArgumentName::Unset;
                    let arg = Value::BlockArgument(BlockArgument::new(name, typ));
                    new_arguments.push(Shared::new(arg.into()));
                } else {
                    new_arguments.push(argument.clone());
                }
            };
        }
        new_arguments
    };
    if !new_arguments.is_empty() {
        let new_arguments = Values::from_vec(new_arguments);
        operation.set_arguments(new_arguments);
    }
}

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::FuncLowering"
    }
    fn parallelizable(&self) -> bool {
        false
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::FuncOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::FuncOp::from_operation_arc(operation.clone());

        {
            let mut operation = operation.wr();
            lower_block_argument_types(&mut operation);
        }

        new_op.set_identifier(op.identifier().unwrap());
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

/// Replace a merge point by a `phi` instruction.
///
/// Example:
/// ```mlir
/// then:
///   %0 = llvm.mlir.constant(3 : i32) : i32
///   llvm.br ^merge(%0 : i32)
/// else:
///   %1 = llvm.mlir.constant(4 : i32) : i32
///   llvm.br ^merge(%1 : i32)
/// merge(%result : i32):
///   br label %exit
/// ```
/// becomes
/// ```mlir
/// then:
///   br label %merge
/// else:
///   br label %merge
/// merge:
///   %2 = phi i32 [ 4, %else ], [ 3, %then ]
/// ```
struct MergeLowering;

/// Determine which values are passed to the given block.
///
/// For example, in
/// ```mlir
/// ^then:
///   %c3_i32 = llvm.mlir.constant(3 : i32) : i32
///   llvm.br ^merge(%c3_i32 : i32)
/// ^else:
///   %c4_i32 = llvm.mlir.constant(4 : i32) : i32
///   llvm.br ^merge(%c4_i32 : i32)
/// ^merge(%result : i32):
///   llvm.br ^exit
/// ```
/// the return value of `determine_argument_pairs` will be `[(%c3_i32, ^then),
/// (%c4_i32, ^else)]`.
fn determine_argument_pairs(block: &Shared<Block>) -> Vec<(Shared<OpOperand>, Shared<Block>)> {
    let callers = block.rd().callers();
    if callers.is_none() {
        return vec![];
    }
    let callers = callers.unwrap();
    let mut argument_pairs = vec![];
    for caller in callers.iter() {
        let caller = caller.rd();
        let caller_operand = caller.operation().rd().operand(1).unwrap();
        let caller_block = caller.operation().rd().parent().unwrap();
        argument_pairs.push((caller_operand, caller_block.clone()));
    }
    argument_pairs
}

/// Replace the operands of the argument pairs by constants if possible.
fn replace_constant_argument_pairs(pairs: &mut Vec<(Shared<OpOperand>, Shared<Block>)>) {
    for pair in pairs {
        let (op_operand, block) = pair.clone();
        let new = constant_op_operand(op_operand);
        if let Some(new) = new {
            *pair = (new, block);
        }
    }
}

fn verify_argument_pairs(pairs: &[(Shared<OpOperand>, Shared<Block>)]) {
    if pairs.len() != 2 {
        panic!("Expected two callers");
    }
    let mut typ: Option<Shared<dyn Type>> = None;
    for (op_operand, _) in pairs.iter() {
        let value_typ = op_operand.rd().value.rd().typ().unwrap();
        if let Some(typ) = &typ {
            let typ = typ.rd().to_string();
            let value_typ = value_typ.rd().to_string();
            if typ != value_typ {
                panic!("Expected same type, but got {typ} and {value_typ}");
            }
        } else {
            typ = Some(value_typ.clone());
        }
    }
}

/// Set the result of phi given the block `argument`.
///
/// This is part of rewriting `%result` as a block argument:
/// ```mlir
/// ^merge(%result : i32):
/// ```
/// to `%result` as a operation result:
/// ```mlir
/// merge:
///   %result = phi i32 [ 3, %then ], [ 4, %else ]
/// ```
fn set_phi_result(phi: Shared<dyn Op>, argument: &Shared<Value>) {
    let argument = argument.rd();

    let users = argument.users();
    let users = match users {
        Users::OpOperands(users) => users,
        Users::HasNoOpResults => vec![],
    };

    if let Value::BlockArgument(arg) = &*argument {
        let typ = Some(arg.typ());
        let defining_op = Some(phi.clone());
        let name = match &arg.name {
            BlockArgumentName::Name(name) => name.to_string(),
            BlockArgumentName::Anonymous => panic!("Expected a named block argument"),
            BlockArgumentName::Unset => panic!("Block argument has no name"),
        };
        let res = OpResult::new(Some(name), typ, defining_op);
        let new = Value::OpResult(res);
        let new = Shared::new(new.into());
        phi.rd()
            .operation()
            .wr()
            .set_results(Values::from_vec(vec![new.clone()]));

        for user in users.iter() {
            let mut user = user.wr();
            user.value = new.clone();
        }
    } else {
        panic!("Expected a block argument");
    }
}

/// Replace the only argument of the block by a `phi` instruction.
fn insert_phi(block: Shared<Block>) {
    let arguments = block.rd().arguments.vec();
    let mut arguments = arguments.wr();
    assert!(
        arguments.len() == 1,
        "Not implemented for multiple arguments"
    );
    let mut operation = Operation::default();
    operation.set_parent(Some(block.clone()));

    let operation = Shared::new(operation.into());
    let mut phi = targ3t::llvmir::PhiOp::new(operation);
    let mut argument_pairs = determine_argument_pairs(&block);
    replace_constant_argument_pairs(&mut argument_pairs);
    verify_argument_pairs(&argument_pairs);
    phi.set_argument_pairs(Some(argument_pairs));
    let argument = arguments.first().unwrap();
    let phi = Shared::new(phi.into());
    set_phi_result(phi.clone(), argument);
    arguments.clear();
    block.wr().insert_op(phi, 0);
}

/// Remove the operands of the callers that call given block.
///
/// For example, in
/// ```mlir
/// ^then:
///   %1 = llvm.mlir.constant(3 : i32) : i32
///   llvm.br ^merge(%1 : i32)
/// ^merge(%result : i32):
///   llvm.br ^exit
/// ```
/// on line 3, `%1 : i32` in `llvm.br ^merge(%1 : i32)` will be removed.
fn remove_caller_operands(block: Shared<Block>) {
    let callers = block.rd().callers();
    if callers.is_none() {
        return;
    }
    let callers = callers.unwrap();
    for caller in callers.iter() {
        let operands = caller.rd().operation().rd().operands();
        let operands = operands.vec();
        let mut operands = operands.wr();
        operands.pop();
    }
}

fn one_child_block_has_argument(op: &dyn Op) -> Result<bool> {
    if !op.is_func() {
        return Ok(false);
    }
    let operation = op.operation();
    if operation.rd().region().is_none() {
        return Ok(false);
    }
    for block in operation.rd().blocks().into_iter() {
        let has_argument = !block.rd().arguments.vec().rd().is_empty();
        if has_argument {
            return Ok(true);
        }
    }
    Ok(false)
}

impl Rewrite for MergeLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::MergeLowering"
    }
    fn parallelizable(&self) -> bool {
        false
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        if !one_child_block_has_argument(&*op.rd())? {
            return Ok(RewriteResult::Unchanged);
        }
        let blocks = op.rd().operation().rd().region().unwrap().rd().blocks();
        for block in blocks.into_iter() {
            let has_argument = !block.rd().arguments.vec().rd().is_empty();
            if has_argument {
                insert_phi(block.clone());
                remove_caller_operands(block.clone());
            }
        }
        Ok(RewriteResult::Unchanged)
    }
}

struct ModuleLowering;

impl Rewrite for ModuleLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::ModuleLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        if !op.rd().as_any().is::<ir::ModuleOp>() {
            return Ok(RewriteResult::Unchanged);
        }
        let operation = op.rd().operation().clone();
        let new_op = targ3t::llvmir::ModuleOp::from_operation_arc(operation);
        let new_op = Shared::new(new_op.into());
        op.rd().replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ReturnLowering;

impl Rewrite for ReturnLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::ReturnLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::ReturnOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation();
        let new_op = targ3t::llvmir::ReturnOp::from_operation_arc(operation.clone());
        replace_constant_operands(&new_op);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct StoreLowering;

impl Rewrite for StoreLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::ReturnLowering"
    }
    fn parallelizable(&self) -> bool {
        true
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = match op.as_any().downcast_ref::<dialect::llvm::StoreOp>() {
            Some(op) => op,
            None => return Ok(RewriteResult::Unchanged),
        };
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::StoreOp::from_operation_arc(operation.clone());
        {
            let op_operand = op.value();
            let value = &op_operand.rd().value;
            let value_typ = value.rd().typ().unwrap();
            let value_typ = value_typ.rd();
            let value_typ = value_typ
                .as_any()
                .downcast_ref::<dialect::llvm::ArrayType>()
                .unwrap();
            new_op.set_len(value_typ.num_elements() as usize);
        }
        replace_constant_operands(&new_op);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertMLIRToLLVMIR;

impl TypeConvert for ConvertMLIRToLLVMIR {
    fn convert_str(src: &str) -> Result<Shared<dyn Type>> {
        let typ = if src == "..." {
            dialect::llvm::VariadicType::new()
        } else {
            panic!("Not implemented for {}", src);
        };
        Ok(Shared::new(typ.into()))
    }
    fn convert_type(from: &Shared<dyn Type>) -> Result<Shared<dyn Type>> {
        let from_rd = from.rd();
        if from_rd.as_any().is::<IntegerType>() {
            return Ok(from.clone());
        }
        if from_rd.as_any().is::<dialect::llvm::PointerType>() {
            let typ = targ3t::llvmir::PointerType::new();
            return Ok(Shared::new(typ.into()));
        }
        if let Some(typ) = from_rd
            .as_any()
            .downcast_ref::<dialect::llvm::FunctionType>()
        {
            let arguments = typ.arguments().clone();
            let converted = arguments
                .types
                .iter()
                .map(Self::convert_type)
                .collect::<Result<Vec<_>>>()?;
            let arguments = Types::from_vec(converted);
            let typ = targ3t::llvmir::FunctionType::new(typ.return_types().clone(), arguments);
            return Ok(Shared::new(typ.into()));
        }
        let typ = Self::convert_str(&from_rd.to_string())?;
        Ok(typ)
    }
}

impl Pass for ConvertMLIRToLLVMIR {
    const NAME: &'static str = "convert-mlir-to-llvmir";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &AddLowering,
            &AllocaLowering,
            &BranchLowering,
            &CallLowering,
            &CondBranchLowering,
            &DeadCodeElimination,
            &FuncLowering,
            &MergeLowering,
            &ModuleLowering,
            &ReturnLowering,
            &StoreLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
