use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::ir::display_region_inside_func;
use crate::ir::Attribute;
use crate::ir::IntegerAttr;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::StringAttr;
use crate::ir::Value;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// Interface for LLVM IR operations that have a constant value.
///
/// Unlike MLIR, LLVM IR does not have a separate constant operation.  So when
/// lowering from MLIR to LLVM IR, we store the constant value in the operation
/// itself. Next, we unset the link to the outdated constant operation, which
/// then is cleaned up by the dead code elimination.
pub trait OneConst: Op {
    fn const_value(&self) -> Arc<dyn Attribute>;
    fn set_const_value(&mut self, const_value: Arc<dyn Attribute>);
}

pub struct AddOp {
    operation: Arc<RwLock<Operation>>,
    const_value: Option<Arc<dyn Attribute>>,
}

impl Op for AddOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::add".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        AddOp {
            operation,
            const_value: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().try_read().unwrap();
        let results = operation.results();
        let results = results.vec();
        let results = results.try_read().unwrap();
        let result = results[0].try_read().unwrap();
        write!(f, "{result} = add")?;
        let result_types = operation.results().types();
        write!(f, " {result_types}")?;
        let operands = operation.operands();
        let operands = operands.vec();
        let operands = operands.try_read().unwrap();
        let operand = operands.get(0).unwrap();
        let operand = operand.try_read().unwrap();
        write!(f, " {operand}, ")?;
        let const_value = self.const_value();
        write!(f, "{const_value}")?;
        write!(f, "\n")
    }
}

impl OneConst for AddOp {
    fn const_value(&self) -> Arc<dyn Attribute> {
        self.const_value.clone().unwrap()
    }
    fn set_const_value(&mut self, const_value: Arc<dyn Attribute>) {
        self.const_value = Some(const_value);
    }
}

impl Display for AddOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct AllocaOp {
    operation: Arc<RwLock<Operation>>,
    const_value: Option<Arc<dyn Attribute>>,
    element_type: Option<String>,
}

impl AllocaOp {
    pub fn array_size(&self) -> Arc<dyn Attribute> {
        self.const_value.clone().unwrap()
    }
    pub fn set_element_type(&mut self, element_type: String) {
        self.element_type = Some(element_type);
    }
}

impl Op for AllocaOp {
    fn operation_name() -> OperationName {
        OperationName::new("alloca".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        AllocaOp {
            operation,
            const_value: None,
            element_type: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().try_read().unwrap();
        write!(f, "{} = ", operation.results())?;
        write!(f, "{} ", AllocaOp::operation_name())?;
        write!(f, "{}", self.element_type.as_ref().unwrap())?;
        let array_size = self.array_size();
        let array_size = array_size.as_any().downcast_ref::<IntegerAttr>().unwrap();
        let typ = array_size.typ();
        let value = array_size.value();
        write!(f, ", {typ} {value}, align 1")?;
        Ok(())
    }
}

impl OneConst for AllocaOp {
    fn const_value(&self) -> Arc<dyn Attribute> {
        self.const_value.clone().unwrap()
    }
    fn set_const_value(&mut self, const_value: Arc<dyn Attribute>) {
        self.const_value = Some(const_value);
    }
}

impl Display for AllocaOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct CallOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

impl Call for CallOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("call".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        CallOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().try_read().unwrap();
        write!(f, "{} = ", operation.results())?;
        let return_type = operation.result_type(0);
        let return_type = return_type.unwrap();
        let return_type = return_type.try_read().unwrap();
        write!(f, "call {return_type} {}(", self.identifier().unwrap())?;
        write!(f, "{} ", operation.operand_types())?;
        write!(f, "{})", operation.operands())?;
        Ok(())
    }
}

impl Display for CallOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct FuncOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

impl FuncOp {
    fn has_implementation(&self) -> bool {
        let operation = self.operation().try_read().unwrap();
        let region = operation.region();
        region.is_some()
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::func".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        FuncOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn is_func(&self) -> bool {
        true
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let return_type = self.return_type().unwrap();
        let return_type = return_type.try_read().unwrap();
        let fn_keyword = if self.has_implementation() {
            "define"
        } else {
            "declare"
        };
        write!(
            f,
            "{fn_keyword} {return_type} {}(",
            self.identifier().unwrap()
        )?;
        let arguments = self.arguments().unwrap();
        let arguments = arguments.vec();
        let arguments = arguments.try_read().unwrap();
        for argument in arguments.iter() {
            let argument = argument.try_read().unwrap();
            if let Value::BlockArgument(arg) = &*argument {
                let typ = arg.typ();
                let typ = typ.try_read().unwrap();
                match arg.name() {
                    Some(name) => write!(f, "{} {}", typ, name),
                    None => write!(f, "{}", typ),
                }?;
            } else {
                panic!("Expected BlockArgument");
            }
        }
        write!(f, ")")?;

        let operation = self.operation().try_read().unwrap();
        display_region_inside_func(f, &*operation, indent)
    }
}

impl Func for FuncOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct ModuleOp {
    operation: Arc<RwLock<Operation>>,
    module_id: String,
    source_filename: String,
    module_flags: String,
    debug_info: String,
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.mlir.module".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        ModuleOp {
            operation,
            module_id: "LLVMDialectModule".to_string(),
            source_filename: "LLVMDialectModule".to_string(),
            module_flags: "!{!0}".to_string(),
            debug_info: r#"!0 = !{i32 2, !"Debug Info Version", i32 3}"#.to_string(),
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, "; ModuleID = '{}'\n", self.module_id)?;
        write!(f, r#"source_filename = "{}""#, self.source_filename)?;
        write!(f, "\n\n")?;
        let operation = self.operation().try_read().unwrap();
        let region = operation.region();
        if let Some(region) = region {
            let region = region.try_read().unwrap();
            for block in region.blocks() {
                let block = block.try_read().unwrap();
                block.display(f, indent)?;
            }
        }
        write!(f, "\n\n!llvm.module.flags = {}", self.module_flags)?;
        write!(f, "\n\n{}", self.debug_info)?;
        Ok(())
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
    const_value: Option<Arc<dyn Attribute>>,
}

impl OneConst for ReturnOp {
    fn const_value(&self) -> Arc<dyn Attribute> {
        self.const_value.clone().unwrap()
    }
    fn set_const_value(&mut self, const_value: Arc<dyn Attribute>) {
        self.const_value = Some(const_value);
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::return".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        ReturnOp {
            operation,
            const_value: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        if let Some(const_value) = &self.const_value {
            let const_value = const_value.as_any().downcast_ref::<IntegerAttr>().unwrap();
            let typ = const_value.typ();
            write!(f, "ret {typ} {const_value}")
        } else {
            // Return is allowed to be a non-constant operand (for example, `ret i32 %1`).
            let operation = self.operation().try_read().unwrap();
            let operand = operation.operand(0).unwrap();
            let operand = operand.try_read().unwrap();
            let value = operand.value();
            let value = value.try_read().unwrap();
            let typ = value.typ();
            let typ = typ.try_read().unwrap();
            write!(f, "ret {typ} {value}")
        }
    }
}

impl Display for ReturnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct StoreOp {
    operation: Arc<RwLock<Operation>>,
    const_value: Option<Arc<dyn Attribute>>,
    len: Option<usize>,
}

impl StoreOp {
    pub fn addr(&self) -> Arc<RwLock<OpOperand>> {
        let operation = self.operation.try_read().unwrap();
        // The value was removed during lowering.
        let operand = operation.operand(0).unwrap();
        operand
    }
    pub fn set_len(&mut self, len: usize) {
        self.len = Some(len);
    }
    pub fn len(&self) -> Option<usize> {
        self.len
    }
}

impl OneConst for StoreOp {
    fn const_value(&self) -> Arc<dyn Attribute> {
        self.const_value.clone().unwrap()
    }
    fn set_const_value(&mut self, const_value: Arc<dyn Attribute>) {
        self.const_value = Some(const_value);
    }
}

impl Op for StoreOp {
    fn operation_name() -> OperationName {
        OperationName::new("store".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        StoreOp {
            operation,
            const_value: None,
            len: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "store ")?;
        let const_value = self.const_value();
        let const_value = const_value.as_any().downcast_ref::<StringAttr>().unwrap();
        let len = self.len().expect("len not set during lowering");
        write!(f, "[{len} x i8] ")?;
        write!(f, "c{const_value}, ")?;
        write!(f, "ptr {}, ", self.addr().try_read().unwrap())?;
        write!(f, "align 1")?;
        Ok(())
    }
}

impl Display for StoreOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
