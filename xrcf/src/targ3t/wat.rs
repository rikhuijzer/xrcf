//! WebAssembly Text format (`.wat`).
//!
//! This dialect holds operations that when printed are valid wat.

use crate::ir::canonicalize_identifier;
use crate::ir::BlockArgumentName;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Value;
use crate::ir::Values;
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
    identifier: Option<String>,
    sym_visibility: Option<SymVisibility>,
}

impl FuncOp {
    pub fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    pub fn set_identifier(&mut self, identifier: Option<String>) {
        self.identifier = identifier;
    }
    pub fn sym_visibility(&self) -> Option<SymVisibility> {
        self.sym_visibility.clone()
    }
    pub fn set_sym_visibility(&mut self, visibility: Option<SymVisibility>) {
        self.sym_visibility = visibility;
    }
    pub fn arguments(&self) -> Values {
        self.operation.rd().arguments()
    }
}

fn display_argument(value: &Value) -> String {
    match value {
        Value::BlockArgument(arg) => {
            if let BlockArgumentName::Name(name) = &*arg.name().rd() {
                format!("(param {} {})", name, arg.typ().rd())
            } else {
                panic!("invalid argument: {value}")
            }
        }
        _ => panic!("invalid argument: {value}"),
    }
}

fn display_arguments(values: &Values) -> String {
    let result = values
        .vec()
        .rd()
        .iter()
        .map(|value| display_argument(&value.rd()))
        .collect::<Vec<String>>();
    result.join(" ")
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("func".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        Self {
            operation,
            identifier: None,
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
        let spaces = crate::ir::spaces(2 * indent);
        write!(f, "{spaces}({} ", self.operation.rd().name())?;
        let identifier = self.identifier().expect("identifier not set");
        let identifier = canonicalize_identifier(&identifier);
        if self.sym_visibility() == Some(SymVisibility::Public) {
            write!(f, "(export \"{identifier}\") ")?;
        }
        write!(f, "{}", display_arguments(&self.arguments()))?;
        // TODO: display body
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
