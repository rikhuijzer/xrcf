//! Cranelift IR (CLIR) operations.

use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::shared::Shared;
use crate::shared::SharedExt;
use cranelift_codegen::ir::Function;
use std::fmt::Display;
use std::fmt::Formatter;

pub struct FuncOp {
    operation: Shared<Operation>,
    func: Function,
}

impl FuncOp {
    pub fn func(&self) -> Function {
        self.func.clone()
    }
    pub fn set_func(&mut self, func: Function) {
        self.func = func;
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::clir::func".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        let func = Function::new();
        FuncOp { operation, func }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.func)
    }
}

impl Display for FuncOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct ModuleOp {
    operation: Shared<Operation>,
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("target::clir::module".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        ModuleOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "foo {}", self.operation.rd())
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
