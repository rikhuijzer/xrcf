use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::Rewrite;
use xrcf::convert::RewriteResult;
use xrcf::dialect::func;
use xrcf::ir::Op;
use xrcf::Pass;

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "toy_to_mlir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<func::ReturnOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        Ok(RewriteResult::Unchanged)
    }
}

#[allow(dead_code)]
pub struct ConvertToyToMLIR;

impl Pass for ConvertToyToMLIR {
    const NAME: &'static str = "convert-toy-to-mlir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&FuncLowering];
        xrcf::convert::apply_rewrites(op, &rewrites)
    }
}
