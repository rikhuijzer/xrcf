use crate::parser::BuiltinParse;
use crate::parser::Parser;
use core::fmt::Error;
use core::fmt::Write;
use crate::ir::Op;

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
}

pub fn opt(src: &str, options: OptOptions) -> String {
    let module = Parser::<BuiltinParse>::parse(src).unwrap();
    if options.canonicalize {
        println!("canonicalize");
    }
    format!("{}", module)
}
