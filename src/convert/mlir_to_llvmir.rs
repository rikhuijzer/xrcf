use crate::canonicalize::CanonicalizeOp;
use crate::canonicalize::DeadCodeElimination;
use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::dialect::func::Func;
use crate::ir;
use crate::ir::IntegerAttr;
use crate::ir::Op;
use crate::ir::Value;
use crate::targ3t;
use crate::targ3t::llvmir::OneConst;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

fn remove_operand_to_constant(new_op: &dyn OneConst) {
    let operation = new_op.operation();
    let operation = operation.try_read().unwrap();
    let operands = operation.operands();
    let mut operands = operands.try_write().unwrap();
    operands.remove(0);
}

fn set_constant_value(new_op: &mut dyn OneConst, value: &Value) {
    match &*value {
        Value::BlockArgument(_) => todo!(),
        Value::OpResult(op_res) => {
            let op = op_res.defining_op();
            let op = op.try_read().unwrap();
            let op = op.as_any().downcast_ref::<dialect::llvmir::ConstantOp>();
            if let Some(op) = op {
                let value = op.value();
                let value = value.as_any().downcast_ref::<IntegerAttr>().unwrap();
                new_op.set_const_value(value.i64().to_string());
                remove_operand_to_constant(new_op);
            }
        }
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
        let op = op.try_read().unwrap();
        let operation = op.operation().clone();
        let new_op = targ3t::llvmir::ModuleOp::from_operation(operation)?;
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvmir::FuncOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvmir::FuncOp>()
            .unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::FuncOp::from_operation(operation.clone())?;
        new_op.set_identifier(op.identifier().to_string());
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct AddLowering;

/// Find the operand that points to a constant operation.
///
/// This is used to find a value that can be unlinked during the lowering process.
fn find_constant_operand(op: &dyn Op) -> Option<Arc<RwLock<Value>>> {
    let operation = op.operation();
    let operation = operation.try_read().unwrap();
    let operands = operation.operands();
    let operands = operands.try_read().unwrap();
    for operand in operands.iter() {
        let operand = operand.try_read().unwrap();
        let op = operand.defining_op();
        if let Some(op) = op {
            let op = op.try_read().unwrap();
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
        Ok(op.as_any().is::<dialect::llvmir::AddOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvmir::AddOp>()
            .unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::AddOp::from_operation(operation.clone())?;
        let value = find_constant_operand(op).unwrap();
        set_constant_value(&mut new_op, &value.try_read().unwrap());
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ReturnLowering;

impl ReturnLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::ReturnLowering"
    }
}

impl Rewrite for ReturnLowering {
    fn name(&self) -> &'static str {
        "mlir_to_llvmir::ReturnLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::llvmir::ReturnOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvmir::ReturnOp>()
            .unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::ReturnOp::from_operation(operation.clone())?;
        let value = op.value();
        set_constant_value(&mut new_op, &value.try_read().unwrap());
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertMLIRToLLVMIR;

impl Pass for ConvertMLIRToLLVMIR {
    fn name() -> &'static str {
        "mlir_to_llvmir::ConvertMLIRToLLVMIR"
    }
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &CanonicalizeOp,
            &DeadCodeElimination,
            &FuncLowering,
            &ModuleLowering,
            &AddLowering,
            &ReturnLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
