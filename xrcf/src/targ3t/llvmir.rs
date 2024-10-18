use crate::dialect::func::Func;
use crate::ir::operation::display_region_inside_func;
use crate::ir::operation::OperationName;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Value;
use anyhow::Result;
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
    fn const_value(&self) -> String;
    fn set_const_value(&mut self, const_value: String);
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
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(ModuleOp {
            operation,
            module_id: "LLVMDialectModule".to_string(),
            source_filename: "LLVMDialectModule".to_string(),
            module_flags: "!{!0}".to_string(),
            debug_info: r#"!0 = !{i32 2, !"Debug Info Version", i32 3}"#.to_string(),
        })
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

pub struct FuncOp {
    operation: Arc<RwLock<Operation>>,
    identifier: String,
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::func".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(FuncOp {
            operation,
            identifier: "".to_string(),
        })
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
        write!(f, "define {return_type} {}(", self.identifier)?;
        let arguments = self.arguments().unwrap();
        let arguments = arguments.try_read().unwrap();
        for argument in arguments.iter() {
            let argument = argument.try_read().unwrap();
            if let Value::BlockArgument(arg) = &*argument {
                let typ = arg.typ();
                write!(f, "{} {}", typ, arg.name())?;
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
    fn identifier(&self) -> &str {
        &self.identifier
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = identifier;
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct AddOp {
    operation: Arc<RwLock<Operation>>,
    const_value: Option<String>,
}

impl Op for AddOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::add".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(AddOp {
            operation,
            const_value: None,
        })
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
        let results = results.try_read().unwrap();
        let result = results[0].try_read().unwrap();
        write!(f, "{result} = add")?;
        let result_types = operation.result_types();
        let result_types = result_types.try_read().unwrap();
        let result_type = result_types[0].try_read().unwrap().clone();
        write!(f, " {result_type}")?;
        let operands = operation.operands();
        let operands = operands.try_read().unwrap();
        let operand = operands[0].try_read().unwrap();
        write!(f, " {operand}, ")?;
        let const_value = self.const_value();
        write!(f, "{const_value}")?;
        write!(f, "\n")
    }
}

impl OneConst for AddOp {
    fn const_value(&self) -> String {
        self.const_value.clone().unwrap()
    }
    fn set_const_value(&mut self, const_value: String) {
        self.const_value = Some(const_value);
    }
}

impl Display for AddOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
    const_value: Option<String>,
}

impl OneConst for ReturnOp {
    fn const_value(&self) -> String {
        self.const_value.clone().unwrap()
    }
    fn set_const_value(&mut self, const_value: String) {
        self.const_value = Some(const_value);
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::llvmir::return".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(ReturnOp {
            operation,
            const_value: None,
        })
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        if let Some(const_value) = &self.const_value {
            write!(f, "ret i64 {const_value}")
        } else {
            let operation = self.operation().try_read().unwrap();
            let operand = operation.operand(0);
            let operand = operand.try_read().unwrap();
            let value = operand.value();
            let value = value.try_read().unwrap();
            let typ = value.typ();
            write!(f, "ret {typ} {value}")
        }
    }
}

impl Display for ReturnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
