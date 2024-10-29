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
        let func_op = op.try_read().unwrap();
        let func_op = func_op.as_any().downcast_ref::<python::FuncOp>().unwrap();
        let identifier = func_op.identifier().unwrap();
        let func_op = func_op.operation();
        let mut func_op = func::FuncOp::from_operation(func_op.clone());
        func_op.set_identifier(identifier);
        let func_op = Arc::new(RwLock::new(func_op));
        Ok(RewriteResult::Changed(ChangedOp::new(func_op)))
    }
}

pub struct ConvertPythonToMLIR;

impl Pass for ConvertPythonToMLIR {
    const NAME: &'static str = "convert-python-to-mlir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&FuncLowering];
        apply_rewrites(op, &rewrites)
    }
}
