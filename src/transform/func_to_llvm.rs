use crate::ir::Op;
use crate::transform::Transform;
use anyhow::Result;

pub struct ConvertFuncToLLVM {}

impl Transform for ConvertFuncToLLVM {
    fn transform(&self, op: &dyn Op) -> Result<()> {
        Ok(())
    }
}
