use core::fmt::Write;
use core::fmt::Error;

pub trait Transform {
    /// Transform the source code to the target code.
    // TODO: This should take a memory object not a string.
    fn transform(&self, src: &str, out: &mut dyn Write) -> Result<(), Error>;
}

struct MLIRToLLVMIRTranslation {}

impl Transform for MLIRToLLVMIRTranslation {
    fn transform(&self, src: &str, out: &mut dyn Write) -> Result<(), Error> {
        todo!()
    }
}

pub fn compile(src: &str) -> String {
    let step = MLIRToLLVMIRTranslation {};
    let mut out = String::new();
    step.transform(src, &mut out).unwrap();
    out
}