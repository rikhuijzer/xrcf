use crate::canonicalize::CanonicalizeOp;
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
use crate::ir::BlockArgument;
use crate::ir::GuardedOp;
use crate::ir::GuardedOpOperand;
use crate::ir::GuardedOperation;
use crate::ir::GuardedValue;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::TypeConvert;
use crate::ir::Value;
use crate::ir::Values;
use crate::targ3t;
use crate::targ3t::llvmir::OneConst;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

fn remove_operand_to_constant(new_op: &dyn OneConst) {
    let operation = new_op.operation();
    let operands = operation.operands();
    let operands = operands.vec();
    let mut operands = operands.try_write().unwrap();
    operands.remove(0);
}

fn set_constant_value(new_op: &mut dyn OneConst, value: Arc<RwLock<Value>>) {
    let value = value.try_read().unwrap();
    match &*value {
        Value::BlockArgument(_) => todo!(),
        Value::FuncResult(_) => todo!(),
        Value::OpResult(op_res) => {
            let op = op_res.defining_op();
            let op = op.expect("Defining op not set for OpResult");
            let op = op.try_read().unwrap();
            let op = op.as_any().downcast_ref::<dialect::llvm::ConstantOp>();
            if let Some(op) = op {
                let value = op.value();
                new_op.set_const_value(value.clone());
                remove_operand_to_constant(new_op);
            }
        }
        Value::Variadic(_) => todo!(),
    }
}

struct AddLowering;

/// Find the operand that points to a constant operation.
///
/// This is used to find a value that can be unlinked during the lowering process.
fn find_constant_operand(op: &dyn Op) -> Option<Arc<RwLock<Value>>> {
    let operation = op.operation();
    let operands = operation.operands().vec();
    let operands = operands.try_read().unwrap();
    for operand in operands.iter() {
        let op = operand.defining_op();
        if let Some(op) = op {
            if op.is_const() {
                let value = operand.value();
                return Some(value.clone());
            }
        }
    }
    None
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
        let mut new_op = targ3t::llvmir::AddOp::from_operation_arc(operation.clone());
        let value = find_constant_operand(op).unwrap();
        set_constant_value(&mut new_op, value);
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
        let value = find_constant_operand(op).unwrap();
        set_constant_value(&mut new_op, value);
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
            let typ = argument.typ();
            let typ = typ.try_read().unwrap();
            if typ.as_any().is::<dialect::llvm::PointerType>() {
                let typ = targ3t::llvmir::PointerType::from_str("ptr");
                let typ = Arc::new(RwLock::new(typ));
                let arg = Value::BlockArgument(BlockArgument::new(None, typ));
                new_arguments.push(Arc::new(RwLock::new(arg)));
            } else {
                new_arguments.push(argument.clone());
            }
        }
        new_arguments
    };
    if 0 < new_arguments.len() {
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
        let mut new_op = targ3t::llvmir::ReturnOp::from_operation_arc(operation.clone());
        //
        let value = op.operand();
        if let Some(value) = value {
            set_constant_value(&mut new_op, value.value().clone());
        }
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
        let op_operand = op.value();
        let value = op_operand.value();
        set_constant_value(&mut new_op, value.clone());
        let value_typ = value.typ();
        let value_typ = value_typ.try_read().unwrap();
        let value_typ = value_typ
            .as_any()
            .downcast_ref::<dialect::llvm::ArrayType>()
            .unwrap();
        new_op.set_len(value_typ.num_elements() as usize);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertMLIRToLLVMIR;

impl TypeConvert for ConvertMLIRToLLVMIR {
    fn convert_str(_src: &str) -> Result<Arc<RwLock<dyn Type>>> {
        todo!()
    }
    fn convert(from: &Arc<RwLock<dyn Type>>) -> Result<Arc<RwLock<dyn Type>>> {
        let from_read = from.try_read().unwrap();
        if from_read.as_any().is::<IntegerType>() {
            return Ok(from.clone());
        }
        if from_read.as_any().is::<dialect::llvm::PointerType>() {
            let typ = targ3t::llvmir::PointerType::new();
            return Ok(Arc::new(RwLock::new(typ)));
        }
        let typ = Self::convert_str(&from_read.to_string())?;
        Ok(typ)
    }
}

impl Pass for ConvertMLIRToLLVMIR {
    const NAME: &'static str = "convert-mlir-to-llvmir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &AddLowering,
            &AllocaLowering,
            &CallLowering,
            &CanonicalizeOp,
            &DeadCodeElimination,
            &FuncLowering,
            &ModuleLowering,
            &ReturnLowering,
            &StoreLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
