use crate::arnold;
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
use xrcf::dialect::experimental;
use xrcf::ir::APInt;
use xrcf::ir::Block;
use xrcf::ir::GuardedOp;
use xrcf::ir::GuardedOperation;
use xrcf::ir::IntegerAttr;
use xrcf::ir::IntegerType;
use xrcf::ir::Op;
use xrcf::ir::OpOperand;
use xrcf::ir::Operation;

struct CallLowering;

impl Rewrite for CallLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::CallLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arnold::CallOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<arnold::CallOp>().unwrap();
        let identifier = op.identifier().unwrap();
        let operation = op.operation();
        let mut new_op = func::CallOp::from_operation_arc(operation.clone());
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
        let mut operation = Operation::default();
        operation.set_parent(last.operation().parent().clone());
        let ret = func::ReturnOp::from_operation(operation);
        let ret = Arc::new(RwLock::new(ret));
        last.insert_after(ret);
    }
}

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == arnold::FuncOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<arnold::FuncOp>().unwrap();
        let identifier = op.identifier().unwrap();
        let identifier = format!("@{}", identifier);
        let operation = op.operation();
        let mut new_op = func::FuncOp::from_operation_arc(operation.clone());
        new_op.set_identifier(identifier);
        self.ensure_return(&mut new_op);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ModuleLowering;

impl ModuleLowering {
    fn constant_op(parent: &Arc<RwLock<Block>>) -> Arc<RwLock<dyn Op>> {
        let mut constant = Operation::default();
        constant.set_parent(Some(parent.clone()));
        constant.set_name(arith::ConstantOp::operation_name());
        let typ = IntegerType::new(32);
        let value = APInt::new(32, 0, true);
        let integer = IntegerAttr::new(typ, value);
        let name = parent.try_read().unwrap().unique_value_name();
        let result_type = Arc::new(RwLock::new(typ));
        let result = constant.add_new_op_result(&name, result_type.clone());
        let constant = arith::ConstantOp::from_operation(constant);
        constant.set_value(Arc::new(integer));
        let constant = Arc::new(RwLock::new(constant));
        result.set_defining_op(Some(constant.clone()));
        constant
    }
    fn return_op(
        parent: &Arc<RwLock<Block>>,
        constant: Arc<RwLock<dyn Op>>,
    ) -> Arc<RwLock<dyn Op>> {
        let typ = IntegerType::new(32);
        let result_type = Arc::new(RwLock::new(typ));
        let mut ret = Operation::default();
        ret.set_parent(Some(parent.clone()));
        ret.set_name(func::ReturnOp::operation_name());
        ret.set_anonymous_result(result_type).unwrap();
        let value = constant.result(0);
        let operand = OpOperand::new(value);
        let operand = Arc::new(RwLock::new(operand));
        ret.set_operand(0, operand);
        let ret = func::ReturnOp::from_operation(ret);
        let ret = Arc::new(RwLock::new(ret));
        ret
    }
    fn return_zero(func: Arc<RwLock<dyn Op>>) {
        let operation = func.operation();
        let typ = IntegerType::new(32);
        operation
            .set_anonymous_result(Arc::new(RwLock::new(typ)))
            .unwrap();

        let ops = func.ops();
        for op in &ops {
            let op = op.try_read().unwrap();
            if op.as_any().is::<func::ReturnOp>() {
                op.remove();
            }
        }
        if ops.is_empty() {
            panic!("Expected ops to be non-empty");
        }
        let first = ops.first().unwrap();
        let block = first.operation().parent();
        let block = block.unwrap();

        let constant = Self::constant_op(&block);
        first.insert_after(constant.clone());

        let ret = Self::return_op(&block, constant.clone());
        constant.insert_after(ret.clone());
    }
    fn returns_something(func: Arc<RwLock<dyn Op>>) -> bool {
        let func = func.try_read().unwrap();
        let func_op = func.as_any().downcast_ref::<func::FuncOp>().unwrap();
        let result = func_op.operation().results();
        result.vec().try_read().unwrap().len() == 1
    }
    fn ensure_main_returns_zero(module: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let ops = module.ops();
        let last = ops.last().unwrap();
        if !Self::returns_something(last.clone()) {
            Self::return_zero(last.clone());
            Ok(RewriteResult::Changed(ChangedOp::new(module)))
        } else {
            Ok(RewriteResult::Unchanged)
        }
    }
}

impl Rewrite for ModuleLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::ModuleLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == xrcf::ir::ModuleOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        Self::ensure_main_returns_zero(op.clone())
    }
}

struct PrintLowering;

impl Rewrite for PrintLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::PrintLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == arnold::PrintOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<arnold::PrintOp>().unwrap();
        let text = op.text().unwrap();
        let operation = op.operation();
        let mut new_op = experimental::PrintfOp::from_operation_arc(operation.clone());
        new_op.set_text(text);
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertArnoldToMLIR;

impl Pass for ConvertArnoldToMLIR {
    const NAME: &'static str = "convert-arnold-to-mlir";
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
