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
use xrcf::dialect::experimental;
use xrcf::dialect::func;
use xrcf::dialect::func::Call;
use xrcf::dialect::func::Func;
use xrcf::ir::APInt;
use xrcf::ir::Attribute;
use xrcf::ir::Block;
use xrcf::ir::GuardedOp;
use xrcf::ir::GuardedOpOperand;
use xrcf::ir::GuardedOperation;
use xrcf::ir::IntegerAttr;
use xrcf::ir::IntegerType;
use xrcf::ir::Op;
use xrcf::ir::OpOperand;
use xrcf::ir::Operation;
use xrcf::ir::RenameBareToPercent;
use xrcf::ir::StringAttr;
use xrcf::ir::Value;

const RENAMER: RenameBareToPercent = RenameBareToPercent;

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

struct DeclareIntLowering;

impl Rewrite for DeclareIntLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::DeclareIntLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arnold::DeclareIntOp>())
    }
    /// Rewrite `HEY CHRISTMAS TREE` (declare integer).
    ///
    /// Example:
    /// ```arnold
    /// HEY CHRISTMAS TREE x
    /// YOU SET US UP @NO PROBLEMO
    /// ```
    /// is rewritten to:
    /// ```mlir
    /// %x = arith.constant 1 : i16
    /// ```
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<arnold::DeclareIntOp>().unwrap();
        op.operation().rename_variables(&RENAMER)?;

        let successors = op.operation().successors();
        let set_initial_value = successors.first().unwrap();
        let set_initial_value = set_initial_value.try_read().unwrap();
        let set_initial_value = set_initial_value
            .as_any()
            .downcast_ref::<arnold::SetInitialValueOp>()
            .expect("Expected SetInitialValueOp after DeclareIntOp");

        let operation = Operation::default();
        let new_op = arith::ConstantOp::from_operation(operation);
        new_op.set_parent(op.operation().parent().clone().unwrap());
        new_op.set_value(set_initial_value.value());
        println!("new_op: {}", new_op);
        set_initial_value.remove();
        let new_op = Arc::new(RwLock::new(new_op));
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct FuncLowering;

impl Rewrite for FuncLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::FuncLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.name() == arnold::BeginMainOp::operation_name())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op = op.try_read().unwrap();
        let op = op.as_any().downcast_ref::<arnold::BeginMainOp>().unwrap();
        let identifier = "@main";
        let operation = op.operation();
        let mut new_op = func::FuncOp::from_operation_arc(operation.clone());
        new_op.set_identifier(identifier.to_string());
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
        if ops.is_empty() {
            panic!("Expected ops to be non-empty");
        }
        let last = ops.last().unwrap();
        let block = last.operation().parent();
        let block = block.expect("no parent for operation");

        let constant = Self::constant_op(&block);
        last.insert_after(constant.clone());

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
        let mut operation = Operation::default();
        operation.set_name(experimental::PrintfOp::operation_name());
        let operation = Arc::new(RwLock::new(operation));
        let mut new_op = experimental::PrintfOp::from_operation_arc(operation.clone());
        new_op.set_parent(op.operation().parent().clone().unwrap());

        let operand = op.text();
        let value = operand.value();
        match &*value.try_read().unwrap() {
            // printf("some text")
            Value::Constant(constant) => {
                let text = constant.value();
                let text = text.as_any().downcast_ref::<StringAttr>().unwrap();
                new_op.set_text(text.clone());
            }
            // printf("%d", variable)
            Value::OpResult(_) => {
                let text = StringAttr::from_str("%d");
                new_op.set_text(text);
                new_op.operation().set_operand(1, operand);
            }
            _ => panic!("expected constant or op result"),
        };

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
            &DeclareIntLowering,
            &FuncLowering,
            &ModuleLowering,
            &PrintLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
