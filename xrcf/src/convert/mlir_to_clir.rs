use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::dialect::func::Func;
use crate::ir::canonicalize_identifier;
use crate::ir::Op;
use crate::ir::Operation;
use crate::shared::Shared;
use crate::shared::SharedExt;
use crate::targ3t;
use anyhow::Result;
use cranelift_codegen::ir::types::I32;
use cranelift_codegen::ir::AbiParam;
use cranelift_codegen::ir::Function;
use cranelift_codegen::ir::Signature;
use cranelift_codegen::ir::UserFuncName;
use cranelift_codegen::isa::CallConv;

/// Lower a MLIR `func` to a CLIR `func`.
///
/// This differs a bit from how other lowerings occur in xrcf. Unlike other
/// lowerings, this one personally grabs the operations from below the function
/// and manages the conversion for them too.  This is necessary because the
/// cranelift `Function` does not satisfy all the same behaviors as a
/// [xrcf::ir::Operation], such as for example that each `Operation` contains a
/// region.
///
/// In the end, it doesn't matter so much here. We can just convert the function
/// as a whole since we will not be doing any further lowering on the object.
struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "mlir_to_clir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::func::FuncOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<dialect::func::FuncOp>().unwrap();
        let old_operation = op.operation();
        // A sort of placeholder operation so that code that later assumptions
        // about the `Op` (such as each `Op` has a parent) are not violated.
        let mut new_operation = Operation::default();
        new_operation.set_parent(old_operation.rd().parent());
        let operation = Shared::new(new_operation.into());
        let mut new_op = targ3t::clif::FuncOp::from_operation_arc(operation);

        let return_type = op.return_type().expect("expected one return type");
        let return_type = return_type.rd();
        assert!(return_type.to_string() == "i32");
        let mut sig = Signature::new(CallConv::SystemV);
        sig.returns.push(AbiParam::new(I32));
        // let mut fn_builder_ctx = FunctionBuilderContext::new();
        let identifier = canonicalize_identifier(&op.identifier().unwrap());
        let name = UserFuncName::testcase(identifier);
        let func = Function::with_name_signature(name, sig);
        // let mut builder = FunctionBuilder::new(&mut func, &mut fn_builder_ctx);
        new_op.set_func(func);

        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ModuleLowering;

impl Rewrite for ModuleLowering {
    fn name(&self) -> &'static str {
        "mlir_to_clir::ModuleLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<crate::ir::ModuleOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let new_op = targ3t::clif::ModuleOp::new(op.rd().operation().clone());
        let new_op = Shared::new(new_op.into());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertMLIRToCLIR;

impl Pass for ConvertMLIRToCLIR {
    const NAME: &'static str = "convert-mlir-to-clif";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&FuncLowering, &ModuleLowering];
        apply_rewrites(op, &rewrites)
    }
}
