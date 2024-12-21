use crate::convert::apply_rewrites;
use crate::convert::ChangedOp;
use crate::convert::Pass;
use crate::convert::Rewrite;
use crate::convert::RewriteResult;
use crate::dialect;
use crate::dialect::arith;
use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::dialect::llvm;
use crate::dialect::llvm::PointerType;
use crate::ir::APInt;
use crate::ir::Block;
use crate::ir::BlockArgumentName;
use crate::ir::IntegerAttr;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::StringAttr;
use crate::ir::Value;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use dialect::experimental::PrintfOp;
use std::sync::Arc;
use std::sync::RwLock;

struct PrintLowering;

impl PrintLowering {
    /// Return a constant operation containing the [PrintOp]'s text.
    fn text_constant(parent: &Arc<RwLock<Block>>, op: &PrintfOp) -> (Arc<RwLock<dyn Op>>, usize) {
        let mut const_operation = Operation::default();
        const_operation.set_parent(Some(parent.clone()));
        let text = op.text().clone();
        let text = text.c_string();
        let len = text.len();
        let name = parent.rd().unique_value_name("%");
        let typ = llvm::ArrayType::for_bytes(&text);
        let typ = Shared::new(typ.into());
        let result = const_operation.add_new_op_result(&name, typ);

        let const_op = llvm::ConstantOp::from_operation(const_operation);
        const_op.set_value(Arc::new(StringAttr::new(text)));
        let const_op = Shared::new(const_op.into());
        result.set_defining_op(Some(const_op.clone()));
        (const_op, len)
    }
    /// Return an [Op] which defines the length for the text `alloca`.
    fn len_specifier(parent: &Arc<RwLock<Block>>, len: usize) -> Arc<RwLock<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        let typ = IntegerType::from_str("i16");
        let name = parent.rd().unique_value_name("%");
        let result_type = Shared::new(typ.into());
        let result = operation.add_new_op_result(&name, result_type);
        let op = arith::ConstantOp::from_operation(operation);
        let len = APInt::from_str("i16", &len.to_string());
        op.set_value(Arc::new(IntegerAttr::new(typ, len)));
        let op = Shared::new(op.into());
        result.set_defining_op(Some(op.clone()));
        op
    }
    fn alloca_op(parent: &Arc<RwLock<Block>>, len: Arc<RwLock<dyn Op>>) -> Arc<RwLock<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        let typ = llvm::PointerType::new();
        let name = parent.rd().unique_value_name("%");
        let result_type = Shared::new(typ.into());
        let result = operation.add_new_op_result(&name, result_type);
        let array_size = len.rd().result(0);
        let array_size = OpOperand::new(array_size);
        let array_size = Shared::new(array_size.into());
        operation.set_operand(0, array_size);

        let mut op = llvm::AllocaOp::from_operation(operation);
        op.set_element_type("i8".to_string());
        let op = Shared::new(op.into());
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

        let mut op = llvm::StoreOp::from_operation(operation);

        let value = text.rd().result(0);
        let value = OpOperand::new(value);
        op.set_value(Shared::new(value.into()));

        let addr = alloca.rd().result(0);
        let addr = OpOperand::new(addr);
        op.set_addr(Shared::new(addr.into()));
        Shared::new(op.into())
    }
    fn call_op(
        parent: &Arc<RwLock<Block>>,
        op: &PrintfOp,
        alloca: Arc<RwLock<dyn Op>>,
        set_varargs: bool,
    ) -> Arc<RwLock<dyn Op>> {
        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        {
            let text_addr = alloca.rd().result(0);
            let text_addr = OpOperand::new(text_addr);
            let text_addr = Shared::new(text_addr.into());
            operation.set_operand(0, text_addr);
        }
        if set_varargs {
            let var = op.operation().rd().operand(1);
            let var = var.expect("expected vararg");
            operation.set_operand(1, var);
        }
        let typ = IntegerType::from_str("i32");
        let name = parent.rd().unique_value_name("%");
        let result_type = Shared::new(typ.into());
        let result = operation.add_new_op_result(&name, result_type);

        // Going straight to llvm::CallOp instead of func::CallOp because func
        // does not support varargs.
        let mut op = llvm::CallOp::from_operation(operation);
        op.set_identifier("@printf".to_string());
        if set_varargs {
            let varargs = "!llvm.func<i32 (!llvm.ptr, ...)>";
            let varargs = llvm::FunctionType::from_str(varargs);
            let varargs = Shared::new(varargs.into());
            op.set_varargs(Some(varargs));
        }
        let op = Shared::new(op.into());
        result.set_defining_op(Some(op.clone()));
        op
    }
    fn top_level_op(op: Arc<RwLock<dyn Op>>) -> Arc<RwLock<dyn Op>> {
        let mut out = op.clone();
        for i in 0..1000 {
            let parent_op = out.rd().parent_op();
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
        let ops = top_level_op.rd().ops();
        for op in ops {
            let op = op.rd();
            if op.is_func() {
                let func = match op.as_any().downcast_ref::<llvm::FuncOp>() {
                    Some(func) => func,
                    None => continue,
                };
                if func.identifier() == Some("@printf".to_string()) {
                    return true;
                }
            }
        }
        false
    }
    /// Return an [Op] which defines (declares) the `printf` function.
    fn printf_func_def(
        parent: Arc<RwLock<Block>>,
        set_varargs: bool,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(Some(parent.clone()));
        let result_type = crate::ir::IntegerType::from_str("i32");
        let result_type = Shared::new(result_type.into());
        operation.set_anonymous_result(result_type)?;

        // Going straight to llvm::FuncOp instead of func::FuncOp because func
        // does not support varargs.
        let mut op = llvm::FuncOp::from_operation(operation);
        op.set_identifier("@printf".to_string());
        op.set_sym_visibility(Some("private".to_string()));
        {
            let arg_type = PointerType::new();
            let arg_type = Shared::new(arg_type.into());

            let name = BlockArgumentName::Anonymous;
            let name = Shared::new(name.into());
            let argument = crate::ir::BlockArgument::new(name, arg_type);
            let value = Value::BlockArgument(argument);
            let value = Shared::new(value.into());
            let operation = op.operation();
            operation.wr().set_argument(0, value);
        }
        if set_varargs {
            let value = Value::Variadic;
            let value = Shared::new(value.into());
            op.operation().wr().set_argument(1, value);
        }
        let op = Shared::new(op.into());
        Ok(op)
    }
    /// Define the printf function if it is not already defined.
    fn define_printf(op: Arc<RwLock<dyn Op>>, set_varargs: bool) -> Result<()> {
        let top_level_op = Self::top_level_op(op.clone());
        if !Self::contains_printf(top_level_op.clone()) {
            let ops = top_level_op.rd().ops();
            let op = ops[0].clone();
            let parent = op.rd().operation().rd().parent().unwrap();
            op.rd()
                .insert_before(Self::printf_func_def(parent, set_varargs)?);
        }
        Ok(())
    }
}

impl Rewrite for PrintLowering {
    fn name(&self) -> &'static str {
        "experimental_to_mlir::PrintLowering"
    }
    fn is_match(&self, op: &dyn Op) -> Result<bool> {
        Ok(op.as_any().is::<dialect::experimental::PrintfOp>())
    }
    fn rewrite(&self, op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let op_clone = op.clone();
        let set_varargs = 1 < op.rd().operation().rd().operands().vec().rd().len();
        let op_rd = op_clone.rd();
        let op_rd = op_rd
            .as_any()
            .downcast_ref::<dialect::experimental::PrintfOp>()
            .unwrap();
        let parent = op_rd.operation().rd().parent();
        let parent = parent.expect("no parent");
        let (text, len) = PrintLowering::text_constant(&parent, op_rd);
        op_rd.insert_before(text.clone());
        let len = PrintLowering::len_specifier(&parent, len);
        op_rd.insert_before(len.clone());
        let alloca = PrintLowering::alloca_op(&parent, len);
        op_rd.insert_before(alloca.clone());
        let store = PrintLowering::store_op(&parent, text.clone(), alloca.clone());
        op_rd.insert_before(store);
        PrintLowering::define_printf(op, set_varargs)?;
        let call = PrintLowering::call_op(&parent, op_rd, alloca, set_varargs);
        op_rd.insert_before(call.clone());
        op_rd.remove();

        Ok(RewriteResult::Changed(ChangedOp::new(text)))
    }
}

pub struct ConvertExperimentalToMLIR;

impl Pass for ConvertExperimentalToMLIR {
    const NAME: &'static str = "convert-experimental-to-mlir";
    fn convert(op: Arc<RwLock<dyn Op>>) -> Result<RewriteResult> {
        let rewrites: Vec<&dyn Rewrite> = vec![&PrintLowering];
        apply_rewrites(op, &rewrites)
    }
}
