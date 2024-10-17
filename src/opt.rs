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
use tracing::info;
use tracing::subscriber::SetGlobalDefaultError;
use tracing::Level;
use tracing_subscriber;

pub struct OptOptions {
    canonicalize: bool,
    convert_func_to_llvm: bool,
    mlir_to_llvmir: bool,
}

impl OptOptions {
    pub fn from_str(flags: &str) -> Result<OptOptions> {
        let mut options = OptOptions::default();
        let flags = flags.split(' ').collect::<Vec<&str>>();
        if flags.len() == 0 {
            return Ok(options);
        } else if 1 < flags.len() {
            panic!("Multiple flags not yet supported");
        }
        for flag in flags {
            match flag {
                "--canonicalize" => options.canonicalize = true,
                "--convert-func-to-llvm" => options.convert_func_to_llvm = true,
                "--convert-mlir-to-llvmir" => options.mlir_to_llvmir = true,
                _ => return Err(anyhow::anyhow!("Invalid flag: {}", flag)),
            }
        }
        Ok(options)
    }
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

pub fn init_subscriber(level: Level) -> Result<(), SetGlobalDefaultError> {
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(level)
        .with_test_writer()
        .without_time()
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
}

pub fn opt(op: Arc<RwLock<dyn Op>>, options: OptOptions) -> Result<RewriteResult> {
    info!("Starting optimization pass");
    if options.canonicalize {
        let rewrites: Vec<&dyn Rewrite> = vec![&CanonicalizeOp, &DeadCodeElimination];
        return Ok(apply_rewrites(op, &rewrites)?);
    }
    if options.convert_func_to_llvm {
        assert!(ConvertFuncToLLVM::name() == "func_to_llvm::ConvertFuncToLLVM");
        return ConvertFuncToLLVM::convert(op);
    }
    if options.mlir_to_llvmir {
        assert!(ConvertMLIRToLLVMIR::name() == "mlir_to_llvmir::ConvertMLIRToLLVMIR");
        return ConvertMLIRToLLVMIR::convert(op);
    }
    Ok(RewriteResult::Unchanged)
}
