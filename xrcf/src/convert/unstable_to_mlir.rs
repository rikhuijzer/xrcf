use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::dialect::llvm;
use crate::dialect::llvm::PointerType;
use crate::ir::APInt;
use crate::ir::Block;
use crate::ir::IntegerAttr;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::StringAttr;
use anyhow::Result;
use dialect::unstable::PrintfOp;
use std::sync::Arc;
use std::sync::RwLock;

struct PrintLowering;

impl PrintLowering {
    /// Return a constant operation containing the [PrintOp]'s text.
    fn text_constant(parent: &Arc<RwLock<Block>>, op: &PrintfOp) -> (Arc<RwLock<dyn Op>>, usize) {
        let mut const_operation = Operation::default();
        const_operation.set_parent(Some(parent.clone()));
        let text = op.text().clone().unwrap();
        let text = text.c_string();
        let len = text.len();
        let name = parent.try_read().unwrap().unique_value_name();
        let result = const_operation.add_new_op_result(&name);

        let const_operation = Arc::new(RwLock::new(const_operation));
        let const_op = llvm::ConstantOp::from_operation(const_operation);
        const_op.set_value(Arc::new(StringAttr::new(text)));
        let const_op = Arc::new(RwLock::new(const_op));
        result.set_defining_op(Some(const_op.clone()));
        (const_op, len)
    }
    /// Return an [Op] which defines the length for the text `alloca`.
    fn len_specifier(parent: &Arc<RwLock<Block>>, len: usize) -> Arc<RwLock<dyn Op>> {
        let operation = Operation::default();
        let typ = IntegerType::from_str("i16");
        let name = parent.try_read().unwrap().unique_value_name();
        let result = operation.add_new_op_result(&name);
        let operation = Arc::new(RwLock::new(operation));
        let op = arith::ConstantOp::from_operation(operation);
        let len = APInt::from_str("i16", &len.to_string());
        op.set_value(Arc::new(IntegerAttr::new(typ, len)));
        let op = Arc::new(RwLock::new(op));
        result.set_defining_op(Some(op.clone()));
        op
    }
    fn alloca_op(parent: &Arc<RwLock<Block>>, len: Arc<RwLock<dyn Op>>) -> Arc<RwLock<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        let typ = llvm::PointerType::new();
        let name = parent.try_read().unwrap().unique_value_name();
        let result = operation.add_new_op_result(&name);
        let array_size = len.try_read().unwrap().result(0);
        let array_size = OpOperand::new(array_size);
        operation.set_operand(Arc::new(RwLock::new(array_size)));

        let operation = Arc::new(RwLock::new(operation));
        let mut op = llvm::AllocaOp::from_operation(operation);
        op.set_element_type("i8".to_string());
        let op = Arc::new(RwLock::new(op));
        result.set_defining_op(Some(op.clone()));
        op
    }
    fn store_op(
        parent: &Arc<RwLock<Block>>,
        text: Arc<RwLock<dyn Op>>,
        alloca: Arc<RwLock<dyn Op>>,
    ) -> Arc<RwLock<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));

        let operation = Arc::new(RwLock::new(operation));
        let mut op = llvm::StoreOp::from_operation(operation);

        let value = text.try_read().unwrap().result(0);
        let value = OpOperand::new(value);
        op.set_value(Arc::new(RwLock::new(value)));

        let addr = alloca.try_read().unwrap().result(0);
        let addr = OpOperand::new(addr);
        op.set_addr(Arc::new(RwLock::new(addr)));
        Arc::new(RwLock::new(op))
    }
    fn call_op(parent: &Arc<RwLock<Block>>, alloca: Arc<RwLock<dyn Op>>) -> Arc<RwLock<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        let addr = alloca.try_read().unwrap().result(0);
        let addr = OpOperand::new(addr);
        operation.set_operand(Arc::new(RwLock::new(addr)));

        let name = parent.try_read().unwrap().unique_value_name();
        let result = operation.add_new_op_result(&name);

        let operation = Arc::new(RwLock::new(operation));
        let mut op = func::CallOp::from_operation(operation);
        op.set_identifier("@printf".to_string());
        let op = Arc::new(RwLock::new(op));
        result.set_defining_op(Some(op.clone()));
        op
    }
    fn top_level_op(op: Arc<RwLock<dyn Op>>) -> Arc<RwLock<dyn Op>> {
        let mut out = op.clone();
        for i in 0..1000 {
            let parent_op = out.try_read().unwrap().parent_op();
            match parent_op {
                Some(parent_op) => out = parent_op,
                None => break,
            }
            if i == 999 {
                panic!("infinite loop");
            }
        }
        out
    }
    /// Whether the parent operation of `op` contains a `printf` function.
    fn contains_printf(top_level_op: Arc<RwLock<dyn Op>>) -> bool {
        let top_level_op = top_level_op.try_read().unwrap();
        let ops = top_level_op.ops();
        for op in ops {
            let op = op.try_read().unwrap();
            if op.is_func() {
                let func = op.as_any().downcast_ref::<func::FuncOp>().unwrap();
                if func.identifier() == Some("@printf".to_string()) {
                    return true;
                }
            }
        }
        false
    }
    /// Return an [Op] which defines (declares) the `printf` function.
    fn printf_func_def() -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        let result_type = crate::ir::IntegerType::from_str("i32");
        let result_type = Arc::new(RwLock::new(result_type));
        operation.set_anonymous_result(result_type)?;

        let operation = Arc::new(RwLock::new(operation));
        let mut op = func::FuncOp::from_operation(operation);
        op.set_identifier("@printf".to_string());
        op.set_sym_visibility(Some("private".to_string()));
        {
            let arg_type = PointerType::new();
            let arg_type = Arc::new(RwLock::new(arg_type));
            op.set_argument_from_type(arg_type)?;
        }
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
    /// Define the printf function if it is not already defined.
    fn define_printf(op: Arc<RwLock<dyn Op>>) -> Result<()> {
        let top_level_op = Self::top_level_op(op.clone());
        if !Self::contains_printf(top_level_op.clone()) {
            let op = top_level_op.try_read().unwrap();
            let ops = op.ops();
            let op = ops[0].clone();
            let op = op.try_read().unwrap();
            op.insert_before(Self::printf_func_def()?);
        }
        Ok(())
    }
}

impl Rewrite for PrintLowering {
    fn name(&self) -> &'static str {
        "unstable_to_mlir::PrintLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::unstable::PrintfOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op_clone = op.clone();
        let op_read = op_clone.try_read().unwrap();
        let op_read = op_read
            .as_any()
            .downcast_ref::<dialect::unstable::PrintfOp>()
            .unwrap();
        let parent = op_read.operation().try_read().unwrap().parent();
        let parent = parent.expect("no parent");
        let (text, len) = PrintLowering::text_constant(&parent, op_read);
        op_read.insert_before(text.clone());
        let len = PrintLowering::len_specifier(&parent, len);
        op_read.insert_before(len.clone());
        let alloca = PrintLowering::alloca_op(&parent, len);
        op_read.insert_before(alloca.clone());
        let store = PrintLowering::store_op(&parent, text.clone(), alloca.clone());
        op_read.insert_before(store);
        PrintLowering::define_printf(op)?;
        let call = PrintLowering::call_op(&parent, alloca);
        op_read.insert_before(call.clone());
        op_read.remove();

        Ok(RewriteResult::Changed(ChangedOp::new(text)))
    }
}

pub struct ConvertUnstableToMLIR;

impl Pass for ConvertUnstableToMLIR {
    const NAME: &'static str = "convert-unstable-to-mlir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&PrintLowering];
        apply_rewrites(op, &rewrites)
    }
}
