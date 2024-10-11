use crate::canonicalize::CanonicalizeOp;
use crate::canonicalize::DeadCodeElimination;
use crate::ir::Op;
use crate::rewrite::apply_rewrites;
use crate::rewrite::ConvertFuncToLLVM;
use crate::rewrite::Rewrite;
use crate::Pass;
use anyhow::Result;

pub struct OptOptions {
    canonicalize: bool,
    convert_func_to_llvm: bool,
}

impl OptOptions {
    pub fn set_canonicalize(&mut self, canonicalize: bool) {
        self.canonicalize = canonicalize;
    }
    pub fn set_convert_func_to_llvm(&mut self, convert_func_to_llvm: bool) {
        self.convert_func_to_llvm = convert_func_to_llvm;
    }
}

impl Default for OptOptions {
    fn default() -> Self {
        OptOptions {
            canonicalize: false,
            convert_func_to_llvm: false,
        }
    }
}

pub fn opt(op: &mut dyn Op, options: OptOptions) -> Result<()> {
    if options.canonicalize {
        let rewrites: Vec<&dyn Rewrite> = vec![&CanonicalizeOp, &DeadCodeElimination];
        return Ok(apply_rewrites(op, &rewrites)?);
    }
    if options.convert_func_to_llvm {
        assert!(ConvertFuncToLLVM::name() == "convert-func-to-llvm");
        ConvertFuncToLLVM::convert(op)?;
    }
    Ok(())
}
