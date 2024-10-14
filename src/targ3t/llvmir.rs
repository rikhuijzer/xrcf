use crate::ir::operation::display_region_inside_func;
use crate::ir::operation::OperationName;
use crate::ir::Op;
use crate::ir::Operation;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

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

impl FuncOp {
    pub fn set_identifier(&mut self, identifier: String) {
        self.identifier = identifier;
    }
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
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, "define i64 {}(", self.identifier)?;
        let operation = self.operation();
        let operation = operation.try_read().unwrap();
        let operands = operation.operands();
        let operands = operands.try_read().unwrap();
        for operand in operands.iter() {
            let operand = operand.try_read().unwrap();
            write!(f, "{}", operand)?;
        }
        write!(f, ")")?;

        display_region_inside_func(f, &*operation, indent)
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
    /// The constant value if the return value is a constant.
    ///
    /// LLVM does not supports constants as a separate operation.  So we use
    /// this field store the value if it is a constant to that dead code
    /// elimination can remove it.
    const_value: Option<String>,
}

impl ReturnOp {
    pub fn set_const_value(&mut self, const_value: String) {
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
            todo!("return without a constant value")
        }
    }
}

impl Display for ReturnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
