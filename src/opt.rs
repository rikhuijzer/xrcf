use crate::canonicalize;
use crate::ir::Op;
use crate::transform::ConvertFuncToLLVM;
use crate::transform::Transform;
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
        canonicalize(op);
    }
    if options.convert_func_to_llvm {
        let conversion = ConvertFuncToLLVM {};
        conversion.transform(op).unwrap();
    }
    Ok(())
}
