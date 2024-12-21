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
use xrcf::dialect::scf;
use xrcf::ir::APInt;
use xrcf::ir::Attribute;
use xrcf::ir::Block;
use xrcf::ir::IntegerAttr;
use xrcf::ir::IntegerType;
use xrcf::ir::Op;
use xrcf::ir::OpOperand;
use xrcf::ir::Operation;
use xrcf::ir::RenameBareToPercent;
use xrcf::ir::StringAttr;
use xrcf::ir::Value;
use xrcf::shared::Shared;
use xrcf::shared::SharedExt;

const RENAMER: RenameBareToPercent = RenameBareToPercent;

struct CallLowering;

impl Rewrite for CallLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::CallLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arnold::CallOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<arnold::CallOp>().unwrap();
        let identifier = op.identifier().unwrap();
        let operation = op.operation();
        let mut new_op = func::CallOp::from_operation_arc(operation.clone());
        let identifier = format!("@{}", identifier);
        new_op.set_identifier(identifier);
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

/// Lower `HEY CHRISTMAS TREE` (declare integer).
///
/// Example:
/// ```arnoldc
/// HEY CHRISTMAS TREE x
/// YOU SET US UP @NO PROBLEMO
/// ```
/// is rewritten to:
/// ```mlir
/// %x = arith.constant 1 : i16
/// ```
struct DeclareIntLowering;

impl Rewrite for DeclareIntLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::DeclareIntLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arnold::DeclareIntOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<arnold::DeclareIntOp>().unwrap();
        op.operation().rd().rename_variables(&RENAMER)?;

        let successors = op.operation().rd().successors();
        let set_initial_value = successors.first().unwrap();
        let set_initial_value = set_initial_value.rd();
        let set_initial_value = set_initial_value
            .as_any()
            .downcast_ref::<arnold::SetInitialValueOp>()
            .expect("Expected SetInitialValueOp after DeclareIntOp");

        let operation = Operation::default();
        let new_op = arith::ConstantOp::from_operation(operation);
        new_op.set_parent(op.operation().rd().parent().clone().unwrap());
        new_op.set_value(set_initial_value.value());
        set_initial_value.remove();
        let new_op = Shared::new(new_op.into());
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
        Ok(op.as_any().is::<arnold::BeginMainOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<arnold::BeginMainOp>().unwrap();
        let identifier = "@main";
        let operation = op.operation();
        let mut new_op = func::FuncOp::from_operation_arc(operation.clone());
        new_op.set_identifier(identifier.to_string());
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

/// Lower `BECAUSE I'M GOING TO SAY PLEASE` (if).
///
/// Example:
/// ```arnoldc
/// BECAUSE I'M GOING TO SAY PLEASE x
///   TALK TO THE HAND "x was true"
/// BULLSHIT
///   TALK TO THE HAND "x was false"
/// YOU HAVE NO RESPECT FOR LOGIC
/// ```
/// is rewritten to:
/// ```mlir
/// scf.if %x {
///   experimental.printf("x was true")
/// } else {
///   experimental.printf("x was false")
/// }
/// ```
struct IfLowering;

impl Rewrite for IfLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::IfLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arnold::IfOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<arnold::IfOp>().unwrap();
        let operation = op.operation();
        let mut new_op = scf::IfOp::from_operation_arc(operation.clone());
        new_op.set_then(op.then().clone());
        new_op.set_els(op.els().clone());
        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

struct ModuleLowering;

impl ModuleLowering {
    fn constant_op(parent: &Shared<Block>) -> Shared<dyn Op> {
        let mut constant = Operation::default();
        constant.set_parent(Some(parent.clone()));
        constant.set_name(arith::ConstantOp::operation_name());
        let typ = IntegerType::new(32);
        let value = APInt::new(32, 0, true);
        let integer = IntegerAttr::new(typ, value);
        let name = parent.rd().unique_value_name("%");
        let result_type = Shared::new(typ.into());
        let result = constant.add_new_op_result(&name, result_type.clone());
        let constant = arith::ConstantOp::from_operation(constant);
        constant.set_value(Arc::new(integer));
        let constant = Shared::new(constant.into());
        result.set_defining_op(Some(constant.clone()));
        constant
    }
    fn return_op(parent: &Shared<Block>, constant: Shared<dyn Op>) -> Shared<dyn Op> {
        let typ = IntegerType::new(32);
        let result_type = Shared::new(typ.into());
        let mut ret = Operation::default();
        ret.set_parent(Some(parent.clone()));
        ret.set_name(func::ReturnOp::operation_name());
        ret.set_anonymous_result(result_type).unwrap();
        let value = constant.rd().result(0);
        let operand = OpOperand::new(value);
        let operand = Shared::new(operand.into());
        ret.set_operand(0, operand);
        let ret = func::ReturnOp::from_operation(ret);
        let ret = Shared::new(ret.into());
        ret
    }
    fn return_zero(func: Shared<dyn Op>) {
        let typ = IntegerType::new(32);
        func.rd()
            .operation()
            .wr()
            .set_anonymous_result(Shared::new(typ.into()))
            .unwrap();

        let ops = func.rd().ops();
        if ops.is_empty() {
            panic!("Expected ops to be non-empty");
        }
        let last = ops.last().unwrap();
        let block = last.rd().operation().rd().parent();
        let block = block.expect("no parent for operation");

        let constant = Self::constant_op(&block);
        last.rd().insert_after(constant.clone());

        let ret = Self::return_op(&block, constant.clone());
        constant.rd().insert_after(ret.clone());
    }
    fn returns_something(func: Shared<dyn Op>) -> bool {
        let func = func.rd();
        let func_op = func.as_any().downcast_ref::<func::FuncOp>().unwrap();
        let result = func_op.operation().rd().results();
        result.vec().rd().len() == 1
    }
    fn ensure_main_returns_zero(module: Shared<dyn Op>) -> Result<RewriteResult> {
        let ops = module.rd().ops();
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
        Ok(op.as_any().is::<xrcf::ir::ModuleOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        Self::ensure_main_returns_zero(op.clone())
    }
}

struct PrintLowering;

impl Rewrite for PrintLowering {
    fn name(&self) -> &'static str {
        "example_to_mlir::PrintLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<arnold::PrintOp>())
    }
    fn rewrite(&self, op: Shared<dyn Op>) -> Result<RewriteResult> {
        let op = op.rd();
        let op = op.as_any().downcast_ref::<arnold::PrintOp>().unwrap();
        let mut operation = Operation::default();
        operation.set_name(experimental::PrintfOp::operation_name());
        let operation = Shared::new(operation.into());
        let mut new_op = experimental::PrintfOp::from_operation_arc(operation.clone());
        new_op.set_parent(op.operation().rd().parent().clone().unwrap());

        let operand = op.text();
        let value = operand.rd().value();
        match &*value.rd() {
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
                new_op.operation().wr().set_operand(1, operand);
            }
            _ => panic!("expected constant or op result"),
        };

        let new_op = Shared::new(new_op.into());
        op.replace(new_op.clone());
        Ok(RewriteResult::Changed(ChangedOp::new(new_op)))
    }
}

pub struct ConvertArnoldToMLIR;

impl Pass for ConvertArnoldToMLIR {
    const NAME: &'static str = "convert-arnold-to-mlir";
    fn convert(op: Shared<dyn Op>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![
            &CallLowering,
            &DeclareIntLowering,
            &FuncLowering,
            &IfLowering,
            &ModuleLowering,
            &PrintLowering,
        ];
        apply_rewrites(op, &rewrites)
    }
}
