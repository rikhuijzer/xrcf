use crate::canonicalize::CanonicalizeOp;
use crate::canonicalize::DeadCodeElimination;
use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::ir;
use crate::ir::IntegerAttr;
use crate::ir::Op;
use crate::ir::Value;
use crate::targ3t;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

struct ModuleLowering;

impl Rewrite for ModuleLowering {
    fn is_match(&self, op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(op
            .try_read()
            .unwrap()
            .as_any()
            .downcast_ref::<ir::ModuleOp>()
            .is_some())
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
    fn is_match(&self, op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(op
            .try_read()
            .unwrap()
            .as_any()
            .downcast_ref::<dialect::llvmir::FuncOp>()
            .is_some())
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

struct ReturnLowering;

impl ReturnLowering {
    fn remove_operand_to_constant(new_op: &targ3t::llvmir::ReturnOp) {
        let operation = new_op.operation();
        let operation = operation.try_read().unwrap();
        let operands = operation.operands();
        let mut operands = operands.try_write().unwrap();
        operands.remove(0);
    }
    fn try_set_constant_value(
        op: &dialect::llvmir::ReturnOp,
        new_op: &mut targ3t::llvmir::ReturnOp,
    ) {
        let value = op.value();
        let value = value.try_read().unwrap();
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
                    ReturnLowering::remove_operand_to_constant(&new_op);
                }
            }
        }
    }
}

impl Rewrite for ReturnLowering {
    fn is_match(&self, op: Arc<RwLock<dyn Op>>) -> Result<bool> {
        Ok(op
            .try_read()
            .unwrap()
            .as_any()
            .downcast_ref::<dialect::llvmir::ReturnOp>()
            .is_some())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op
            .as_any()
            .downcast_ref::<dialect::llvmir::ReturnOp>()
            .unwrap();
        let operation = op.operation();
        let mut new_op = targ3t::llvmir::ReturnOp::from_operation(operation.clone())?;
        ReturnLowering::try_set_constant_value(op, &mut new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertMLIRToLLVMIR;

impl Pass for ConvertMLIRToLLVMIR {
    fn name() -> &'static str {
        "convert-mlir-to-llvmir"
    }
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &CanonicalizeOp,
            &DeadCodeElimination,
            &FuncLowering,
            &ModuleLowering,
            &ReturnLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
