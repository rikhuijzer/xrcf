use crate::canonicalize::CanonicalizeOp;
use crate::canonicalize::DeadCodeElimination;
use crate::ir::Op;
use crate::rewrite::apply_rewrites;
use crate::rewrite::ConvertFuncToLLVM;
use crate::rewrite::Rewrite;
use anyhow::Result;
use core::fmt::Error;

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

pub fn opt(op: &mut dyn Op, options: OptOptions) -> Result<(), Error> {
    if options.canonicalize {
        let rewrites: Vec<&dyn Rewrite> = vec![&CanonicalizeOp, &DeadCodeElimination];
        apply_rewrites(op, &rewrites).unwrap();
    }
    if options.convert_func_to_llvm {
        let conversion = ConvertFuncToLLVM {};
        if conversion.is_match(op).unwrap() {
            conversion.rewrite(op).unwrap();
        }
    }
    Ok(())
}
