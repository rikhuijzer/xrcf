use crate::canonicalize::CanonicalizeOp;
use crate::canonicalize::DeadCodeElimination;
use crate::convert::apply_rewrites;
use crate::convert::ConvertFuncToLLVM;
use crate::convert::ConvertMLIRToLLVMIR;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::ir::Op;
use crate::Pass;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

pub struct OptOptions {
    canonicalize: bool,
    convert_func_to_llvm: bool,
    mlir_to_llvmir: bool,
}

impl OptOptions {
    pub fn set_canonicalize(&mut self, canonicalize: bool) {
        self.canonicalize = canonicalize;
    }
    pub fn set_convert_func_to_llvm(&mut self, convert_func_to_llvm: bool) {
        self.convert_func_to_llvm = convert_func_to_llvm;
    }
    pub fn set_mlir_to_llvmir(&mut self, mlir_to_llvmir: bool) {
        self.mlir_to_llvmir = mlir_to_llvmir;
    }
}

impl Default for OptOptions {
    fn default() -> Self {
        OptOptions {
            canonicalize: false,
            convert_func_to_llvm: false,
            mlir_to_llvmir: false,
        }
    }
}

pub fn opt(op: Arc<RwLock<dyn Op>>, options: OptOptions) -> Result<RewriteResult> {
    if options.canonicalize {
        let rewrites: Vec<&dyn Rewrite> = vec![&CanonicalizeOp, &DeadCodeElimination];
        return Ok(apply_rewrites(op, &rewrites)?);
    }
    if options.convert_func_to_llvm {
        assert!(ConvertFuncToLLVM::name() == "convert-func-to-llvm");
        return ConvertFuncToLLVM::convert(op);
    }
    if options.mlir_to_llvmir {
        assert!(ConvertMLIRToLLVMIR::name() == "convert-mlir-to-llvmir");
        return ConvertMLIRToLLVMIR::convert(op);
    }
    Ok(RewriteResult::Unchanged)
}
