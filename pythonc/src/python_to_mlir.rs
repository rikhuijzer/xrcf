use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::apply_rewrites;
use xrcf::convert::Pass;
use xrcf::convert::Rewrite;
use xrcf::convert::RewriteResult;
use xrcf::dialect::func;
use xrcf::ir::Op;

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "python_to_mlir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::FuncOp>())
    }
    fn rewrite(&self, _op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        todo!()
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
