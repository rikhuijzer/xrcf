use crate::python;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::convert::apply_rewrites;
use xrcf::convert::ChangedOp;
use xrcf::convert::Pass;
use xrcf::convert::Rewrite;
use xrcf::convert::RewriteResult;
use xrcf::dialect::arith;
use xrcf::dialect::func;
use xrcf::dialect::func::Call;
use xrcf::dialect::func::Func;
use xrcf::dialect::unstable;
use xrcf::ir::IntegerAttr;
use xrcf::ir::Op;
use xrcf::ir::Operation;

struct CallLowering;

impl Rewrite for CallLowering {
    fn name(&self) -> &'static str {
        "python_to_mlir::CallLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<python::CallOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<python::CallOp>().unwrap();
        let identifier = op.identifier().unwrap();
        let operation = op.operation();
        let mut new_op = func::CallOp::from_operation(operation.clone());
        let identifier = format!("@{}", identifier);
        new_op.set_identifier(identifier);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct FuncLowering;

impl FuncLowering {
    /// Python code does not require a return statement, but MLIR does.
    fn ensure_return(&self, op: &mut func::FuncOp) {
        let ops = op.ops();
        let last = ops.last().unwrap();
        let last = last.try_read().unwrap();
        if last.as_any().is::<func::ReturnOp>() {
            return;
        }
        let operation = Operation::default();
        let operation = Arc::new(RwLock::new(operation));
        let ret = func::ReturnOp::from_operation(operation.clone());
        let ret = Arc::new(RwLock::new(ret));
        last.insert_after(ret);
    }
}

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "python_to_mlir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == python::FuncOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<python::FuncOp>().unwrap();
        let identifier = op.identifier().unwrap();
        let identifier = format!("@{}", identifier);
        let operation = op.operation();
        let mut new_op = func::FuncOp::from_operation(operation.clone());
        new_op.set_identifier(identifier);
        self.ensure_return(&mut new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ModuleLowering;

impl ModuleLowering {
    fn has_main(op: &dyn Op) -> bool {
        let ops = op.ops();
        let last = ops.last().unwrap();
        let last = last.try_read().unwrap();
        if last.as_any().is::<func::FuncOp>() {
            let op = last.as_any().downcast_ref::<func::FuncOp>().unwrap();
            return op.identifier().unwrap() == "@main";
        }
        false
    }
    fn return_zero(func: Arc<RwLock<func::FuncOp>>) {
        let func = func.try_read().unwrap();
        let ops = func.ops();
        if ops.is_empty() {
            panic!("Expected ops to be non-empty");
        }
        let last = ops.last().unwrap();
        let last = last.try_read().unwrap();

        let mut constant = Operation::default();
        constant.set_name(arith::ConstantOp::operation_name());
        let integer = IntegerAttr::from_i32(0);
        let constant = Arc::new(RwLock::new(constant));
        let constant = arith::ConstantOp::from_operation(constant.clone());
        constant.set_value(Arc::new(integer));
        let constant = Arc::new(RwLock::new(constant));
        last.insert_before(constant.clone());
    }
    fn ensure_main(module: Arc<RwLock<dyn Op>>) -> Result<()> {
        let module = module.try_read().unwrap();
        let ops = module.ops();
        let last = ops.last().unwrap();
        let last_read = last.try_read().unwrap();

        let mut main = Operation::default();
        let parent = last_read.operation().try_read().unwrap().parent();
        main.set_parent(parent);

        let main = Arc::new(RwLock::new(main));
        let mut main = func::FuncOp::from_operation(main.clone());
        main.set_identifier("@main".to_string());

        let main = Arc::new(RwLock::new(main));
        func::FuncOp::insert_op(main.clone(), last.clone());
        last_read.insert_after(main.clone());

        last_read.remove();
        Self::return_zero(main);
        Ok(())
    }
}

impl Rewrite for ModuleLowering {
    fn name(&self) -> &'static str {
        "python_to_mlir::ModuleLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        let is_module = op.name() == xrcf::ir::ModuleOp::operation_name();
        if is_module {
            let needs_main = !Self::has_main(op);
            Ok(needs_main)
        } else {
            Ok(false)
        }
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        println!("ModuleLowering");
        Self::ensure_main(op.clone())?;
        Ok(RewriteResult::Changed(ChangedOp::new(op)))
    }
}

struct PrintLowering;

impl Rewrite for PrintLowering {
    fn name(&self) -> &'static str {
        "python_to_mlir::PrintLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == python::PrintOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<python::PrintOp>().unwrap();
        let text = op.text().unwrap();
        let operation = op.operation();
        let mut new_op = unstable::PrintfOp::from_operation(operation.clone());
        new_op.set_text(text);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertPythonToMLIR;

impl Pass for ConvertPythonToMLIR {
    const NAME: &'static str = "convert-python-to-mlir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &CallLowering,
            &FuncLowering,
            &ModuleLowering,
            &PrintLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
