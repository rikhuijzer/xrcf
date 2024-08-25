use core::fmt::Write;
use core::fmt::Error;

pub trait Translate {
    /// Translate the source code to the target code.
    // TODO: This should take a memory object not a string.
    fn translate(&self, src: &str, out: &mut dyn Write) -> Result<(), Error>;
}

struct MLIRToLLVMIRTranslation {}

impl Translate for MLIRToLLVMIRTranslation {
    fn translate(&self, src: &str, out: &mut dyn Write) -> Result<(), Error> {
        todo!()
    }
}

pub fn translate(src: &str) -> String {
    let translation = MLIRToLLVMIRTranslation {};
    let mut out = String::new();
    translation.translate(src, &mut out).unwrap();
    out
}