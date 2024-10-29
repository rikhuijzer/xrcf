use crate::python;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::apply_rewrites;
use xrcf::convert::ChangedOp;
use xrcf::convert::Pass;
use xrcf::convert::Rewrite;
use xrcf::convert::RewriteResult;
use xrcf::dialect::func;
use xrcf::dialect::func::Func;
use xrcf::dialect::unstable;
use xrcf::ir::Op;

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "python_to_mlir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == python::FuncOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<python::FuncOp>().unwrap();
        let identifier = op.identifier().unwrap();
        let operation = op.operation();
        let mut new_op = func::FuncOp::from_operation(operation.clone());
        new_op.set_identifier(identifier);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct PrintLowering;

impl Rewrite for PrintLowering {
    fn name(&self) -> &'static str {
        "python_to_mlir::PrintLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == python::PrintOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<python::PrintOp>().unwrap();
        let text = op.text().unwrap();
        let operation = op.operation();
        let mut new_op = unstable::PrintfOp::from_operation(operation.clone());
        new_op.set_text(text);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertPythonToMLIR;

impl Pass for ConvertPythonToMLIR {
    const NAME: &'static str = "convert-python-to-mlir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&FuncLowering, &PrintLowering];
        apply_rewrites(op, &rewrites)
    }
}
