//! WebAssembly Text format (`.wat`).
//!
//! This dialect holds operations that when printed are valid wat.

use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::shared::Shared;
use crate::shared::SharedExt;
use std::fmt::Display;
use std::fmt::Formatter;

#[derive(Clone, PartialEq)]
pub enum SymVisibility {
    Public,
    Private,
}

pub struct FuncOp {
    operation: Shared<Operation>,
    sym_visibility: Option<SymVisibility>,
}

impl FuncOp {
    pub fn sym_visibility(&self) -> Option<SymVisibility> {
        self.sym_visibility.clone()
    }
    pub fn set_sym_visibility(&mut self, visibility: Option<SymVisibility>) {
        self.sym_visibility = visibility;
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("func".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        Self {
            operation,
            sym_visibility: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let spaces = crate::ir::spaces(indent);
        write!(f, "{spaces}({} ", self.operation.rd().name())?;
        let identifier = "foo";
        if self.sym_visibility() == Some(SymVisibility::Public) {
            write!(f, "(export \"{identifier}\")")?;
        }
        writeln!(f, ")")?;
        Ok(())
    }
}

pub struct ModuleOp {
    operation: Shared<Operation>,
}

impl ModuleOp {
    pub fn functions(&self) -> Vec<Shared<dyn Op>> {
        self.ops()
    }
}

impl Op for ModuleOp {
    fn operation_name() -> OperationName {
        OperationName::new("module".to_string())
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
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        writeln!(f, "({}", self.operation.rd().name())?;
        let functions = self.functions();
        for function in functions {
            write!(f, "{}", crate::ir::spaces(indent))?;
            function.rd().display(f, indent + 1)?;
        }
        writeln!(f, ")")?;
        Ok(())
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
