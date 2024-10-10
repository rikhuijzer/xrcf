use crate::canonicalize;
use crate::ir::Op;
use crate::parser::BuiltinParse;
use crate::parser::Parser;
use core::fmt::Error;
use core::fmt::Write;

pub trait Transform {
    fn transform(&self, from: &dyn Op, to: &mut dyn Write) -> Result<(), Error>;
}

struct MLIRToLLVMIRTranslation {}

impl Transform for MLIRToLLVMIRTranslation {
    fn transform(&self, _from: &dyn Op, _to: &mut dyn Write) -> Result<(), Error> {
        todo!()
    }
}

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
        todo!() // convert_func_to_llvm(op);
    }
    Ok(())
}
