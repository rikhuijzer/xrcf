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
use crate::ir::Constant;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOp;
use crate::ir::GuardedOpOperand;
use crate::ir::GuardedOperation;
use crate::ir::GuardedRegion;
use crate::ir::GuardedValue;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpOperands;
use crate::ir::OpResult;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::TypeConvert;
use crate::ir::Types;
use crate::ir::Value;
use crate::ir::Values;
use crate::targ3t;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct AddLowering;

/// Return an [OpOperand] containing a [Constant].
///
/// Otherwise, return `None`.
fn constant_op_operand(operand: Arc<RwLock<OpOperand>>) -> Option<Arc<RwLock<OpOperand>>> {
    let op = operand.defining_op();
    if let Some(op) = op {
        if op.is_const() {
            let op = op.try_read().unwrap();
            let op = op.as_any().downcast_ref::<dialect::llvm::ConstantOp>();
            if let Some(op) = op {
                let value = op.value();
                let new_value = Constant::new(value.clone());
                let new_operand = Value::Constant(new_value);
                let new_operand = OpOperand::new(Arc::new(RwLock::new(new_operand)));
                return Some(Arc::new(RwLock::new(new_operand)));
            }
        }
    }
    None
}

/// Replace the operands that point to a constant operation by a [Constant].
fn replace_constant_operands(op: &dyn Op) {
    let operation = op.operation();
    let operands = operation.operands().vec();
    let mut operands = operands.try_write().unwrap();
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
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvm::AddOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<dialect::llvm::AddOp>().unwrap();
        let operation = op.operation();
        let new_op = targ3t::llvmir::AddOp::from_operation_arc(operation.clone());
        replace_constant_operands(&new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct AllocaLowering;

impl Rewrite for AllocaLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::AllocaLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvm::AllocaOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvm::AllocaOp>()
            .unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::AllocaOp::from_operation_arc(operation.clone());
        {
            let operation = operation.try_read().unwrap();
            operation.results().convert_types::<ConvertMLIRToLLVMIR>()?;
        }
        new_op.set_element_type(op.element_type().unwrap());
        replace_constant_operands(&new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

/// Lower blocks by removing `^` from the label and removing arguments.
///
/// ```mlir
/// func.func @some_name() {
///   ...
/// ^merge(%result : i32):
///   br label %exit
/// }
/// ```
/// becomes
/// ```mlir
/// func.func @some_name() {
///   ...
/// merge:
///   br label %exit
/// }
struct BlockLowering;

impl Rewrite for BlockLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::BlockLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        if !op.is_func() {
            return Ok(false);
        }
        let region = op.operation().region();
        if let Some(region) = region {
            let blocks = region.blocks();
            let blocks = blocks.try_read().unwrap();
            for block in blocks.iter() {
                let label = block.label();
                if let Some(label) = label {
                    if label.starts_with("^") {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let blocks = op.operation().blocks();
        for block in blocks.iter() {
            let mut block = block.try_write().unwrap();
            let label = block.label();
            if let Some(label) = label {
                if label.starts_with("^") {
                    let new_label = label[1..].to_string();
                    block.set_label(Some(new_label));
                }
            }
        }
        Ok(RewriteResult::Changed(ChangedOp::new(op)))
    }
}

/// Replace `llvm.br` by `br`.
struct BranchLowering;

impl Rewrite for BranchLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::BranchLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        if op.as_any().is::<dialect::llvm::BranchOp>() {
            let operands = op.operation().operands().vec();
            let operands = operands.try_read().unwrap();
            // Check whether [MergeLowering] has already removed the operands.
            if operands.is_empty() {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvm::BranchOp>()
            .unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::BranchOp::from_operation_arc(operation.clone());
        let dest = op.dest().unwrap();
        let mut dest = dest.try_write().unwrap();
        let name = dest.name();
        if name.starts_with("^") {
            let name = name[1..].to_string();
            dest.set_name(&format!("%{}", name));
        }
        new_op.set_dest(op.dest().unwrap());
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct CallLowering;

impl Rewrite for CallLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::CallLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvm::CallOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<dialect::llvm::CallOp>().unwrap();
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
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct CondBranchLowering;

impl Rewrite for CondBranchLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::CondBranchLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvm::CondBranchOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvm::CondBranchOp>()
            .unwrap();
        let operation = op.operation();
        let new_op = targ3t::llvmir::BranchOp::from_operation_arc(operation.clone());
        replace_constant_operands(&new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct FuncLowering;

fn lower_block_argument_types(operation: &mut Operation) {
    let new_arguments = {
        let arguments = operation.arguments();
        let arguments = arguments.vec();
        let arguments = arguments.try_read().unwrap();
        let mut new_arguments = vec![];
        for argument in arguments.iter() {
            let argument_rd = argument.try_read().unwrap();
            if let Value::Variadic = &*argument_rd {
                new_arguments.push(argument.clone());
            } else {
                let typ = argument.typ().unwrap();
                let typ = typ.try_read().unwrap();
                if typ.as_any().is::<dialect::llvm::PointerType>() {
                    let typ = targ3t::llvmir::PointerType::from_str("ptr");
                    let typ = Arc::new(RwLock::new(typ));
                    let arg = Value::BlockArgument(BlockArgument::new(None, typ));
                    new_arguments.push(Arc::new(RwLock::new(arg)));
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
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvm::FuncOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<dialect::llvm::FuncOp>().unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::FuncOp::from_operation_arc(operation.clone());

        {
            let mut operation = operation.try_write().unwrap();
            lower_block_argument_types(&mut operation);
        }

        new_op.set_identifier(op.identifier().unwrap());
        let new_op = Arc::new(RwLock::new(new_op));
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
fn determine_argument_pairs(
    block: &Arc<RwLock<Block>>,
) -> Vec<(Arc<RwLock<OpOperand>>, Arc<RwLock<Block>>)> {
    let callers = block.callers();
    if callers.is_none() {
        return vec![];
    }
    let callers = callers.unwrap();
    let mut argument_pairs = vec![];
    for caller in callers.iter() {
        let caller = caller.try_read().unwrap();
        let caller_operand = caller.operation().operand(0).unwrap();
        let caller_block = caller.operation().parent().unwrap();
        argument_pairs.push((caller_operand, caller_block.clone()));
    }
    argument_pairs
}

/// Replace the operands of the argument pairs by constants if possible.
fn replace_constant_argument_pairs(pairs: &mut Vec<(Arc<RwLock<OpOperand>>, Arc<RwLock<Block>>)>) {
    for i in 0..pairs.len() {
        let (op_operand, block) = pairs[i].clone();
        let new = constant_op_operand(op_operand);
        if let Some(new) = new {
            pairs[i] = (new, block);
        }
    }
}

fn verify_argument_pairs(pairs: &Vec<(Arc<RwLock<OpOperand>>, Arc<RwLock<Block>>)>) {
    if pairs.len() != 2 {
        panic!("Expected two callers");
    }
    let mut typ: Option<Arc<RwLock<dyn Type>>> = None;
    for (op_operand, _) in pairs.iter() {
        let op_operand = op_operand.try_read().unwrap();
        let value = op_operand.value();
        let value_typ = value.typ().unwrap();
        if let Some(typ) = &typ {
            let typ = typ.try_read().unwrap();
            let value_typ = value_typ.try_read().unwrap();
            let value_typ = value_typ.to_string();
            let typ = typ.to_string();
            if typ != value_typ {
                panic!("Expected same type, but got {typ} and {value_typ}");
            }
        } else {
            typ = Some(value_typ.clone());
        }
    }
}

/// Set the result of phi given the block argument.
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
fn set_phi_result(phi: Arc<RwLock<dyn Op>>, argument: &Arc<RwLock<Value>>) {
    todo!("find the users here and after the change update them");

    let operation = phi.operation();
    let argument = argument.try_read().unwrap();
    if let Value::BlockArgument(arg) = &*argument {
        let typ = Some(arg.typ());
        let defining_op = Some(phi.clone());
        let res = OpResult::new(arg.name(), typ, defining_op);
        let new = Value::OpResult(res);
        operation.set_results(Values::from_vec(vec![Arc::new(RwLock::new(new))]));
    }
}

/// Replace the only argument of the block by a `phi` instruction.
fn insert_phi(block: Arc<RwLock<Block>>) {
    let block_read = block.try_read().unwrap();
    let arguments = block_read.arguments().vec();
    let mut arguments = arguments.try_write().unwrap();
    assert!(
        arguments.len() == 1,
        "Not implemented for multiple arguments"
    );
    let mut operation = Operation::default();
    operation.set_parent(Some(block.clone()));

    let operation = Arc::new(RwLock::new(operation));
    let mut phi = targ3t::llvmir::PhiOp::new(operation);
    let mut argument_pairs = determine_argument_pairs(&block);
    replace_constant_argument_pairs(&mut argument_pairs);
    verify_argument_pairs(&argument_pairs);
    phi.set_argument_pairs(Some(argument_pairs));
    let argument = arguments.get(0).unwrap();
    let phi = Arc::new(RwLock::new(phi));
    set_phi_result(phi.clone(), argument);
    arguments.clear();
    block_read.insert_op(phi, 0);
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
fn remove_caller_operands(block: Arc<RwLock<Block>>) {
    let callers = block.callers();
    if callers.is_none() {
        return;
    }
    let callers = callers.unwrap();
    for caller in callers.iter() {
        let caller = caller.operation();
        caller.set_operands(OpOperands::default());
    }
}

impl Rewrite for MergeLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::MergeLowering"
    }
    /// Return true if the operation is a function where one of the children
    /// blocks has an argument.
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        if !op.is_func() {
            return Ok(false);
        }
        let operation = op.operation();
        if operation.region().is_none() {
            return Ok(false);
        }
        let blocks = operation.blocks();
        for block in blocks.iter() {
            let block = block.try_read().unwrap();
            let has_argument = !block.arguments().vec().try_read().unwrap().is_empty();
            if has_argument {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let blocks = op.operation().region().unwrap().blocks();
        let blocks = blocks.try_read().unwrap();
        for block in blocks.iter() {
            let block_read = block.try_read().unwrap();
            let has_argument = !block_read.arguments().vec().try_read().unwrap().is_empty();
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
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<ir::ModuleOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let operation = op.operation().clone();
        let new_op = targ3t::llvmir::ModuleOp::from_operation_arc(operation);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ReturnLowering;

impl Rewrite for ReturnLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::ReturnLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvm::ReturnOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvm::ReturnOp>()
            .unwrap();
        let operation = op.operation();
        let new_op = targ3t::llvmir::ReturnOp::from_operation_arc(operation.clone());
        replace_constant_operands(&new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct StoreLowering;

impl Rewrite for StoreLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::ReturnLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvm::StoreOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvm::StoreOp>()
            .unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::StoreOp::from_operation_arc(operation.clone());
        {
            let op_operand = op.value();
            let value = op_operand.value();
            let value_typ = value.typ().unwrap();
            let value_typ = value_typ.try_read().unwrap();
            let value_typ = value_typ
                .as_any()
                .downcast_ref::<dialect::llvm::ArrayType>()
                .unwrap();
            new_op.set_len(value_typ.num_elements() as usize);
        }
        replace_constant_operands(&new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertMLIRToLLVMIR;

impl TypeConvert for ConvertMLIRToLLVMIR {
    fn convert_str(src: &str) -> Result<Arc<RwLock<dyn Type>>> {
        let typ = if src == "..." {
            dialect::llvm::VariadicType::new()
        } else {
            panic!("Not implemented for {}", src);
        };
        Ok(Arc::new(RwLock::new(typ)))
    }
    fn convert_type(from: &Arc<RwLock<dyn Type>>) -> Result<Arc<RwLock<dyn Type>>> {
        let from_rd = from.try_read().unwrap();
        if from_rd.as_any().is::<IntegerType>() {
            return Ok(from.clone());
        }
        if from_rd.as_any().is::<dialect::llvm::PointerType>() {
            let typ = targ3t::llvmir::PointerType::new();
            return Ok(Arc::new(RwLock::new(typ)));
        }
        if let Some(typ) = from_rd
            .as_any()
            .downcast_ref::<dialect::llvm::FunctionType>()
        {
            let arguments = typ.arguments().clone();
            let converted = arguments
                .vec()
                .iter()
                .map(|argument| Self::convert_type(argument))
                .collect::<Result<Vec<_>>>()?;
            let arguments = Types::from_vec(converted);
            let typ = targ3t::llvmir::FunctionType::new(typ.return_types().clone(), arguments);
            return Ok(Arc::new(RwLock::new(typ)));
        }
        let typ = Self::convert_str(&from_rd.to_string())?;
        Ok(typ)
    }
}

impl Pass for ConvertMLIRToLLVMIR {
    const NAME: &'static str = "convert-mlir-to-llvmir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &AddLowering,
            &AllocaLowering,
            &BlockLowering,
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
