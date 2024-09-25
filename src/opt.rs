use crate::parser::BuiltinParse;
use crate::parser::Parser;
use core::fmt::Error;
use core::fmt::Write;

pub trait Transform {
    /// Transform the source code to the target code.
    // TODO: This should take a memory object not a string.
    fn transform(&self, src: &str, out: &mut dyn Write) -> Result<(), Error>;
}

struct MLIRToLLVMIRTranslation {}

impl Transform for MLIRToLLVMIRTranslation {
    fn transform(&self, _src: &str, _out: &mut dyn Write) -> Result<(), Error> {
        todo!()
    }
}

pub struct Options {
    canonicalize: bool,
}

pub fn opt(src: &str, options: Options) -> String {
    let module = Parser::<BuiltinParse>::parse(src).unwrap();
    if options.canonicalize {
        println!("canonicalize");
    }
    format!("{}", module)
}
