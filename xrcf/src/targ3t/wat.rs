//! WebAssembly Text format (`.wat`).
//!
//! This dialect holds operations that when printed are valid wat.

use crate::ir::canonicalize_identifier;
use crate::ir::BlockArgumentName;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Prefixes;
use crate::ir::Type;
use crate::ir::Types;
use crate::ir::Value;
use crate::ir::Values;
use crate::shared::Shared;
use crate::shared::SharedExt;
use std::fmt::Display;
use std::fmt::Formatter;

const PREFIXES: Prefixes = Prefixes {
    argument: "$arg",
    block: "bb",
    ssa: "$",
};

#[derive(Clone, PartialEq)]
pub enum SymVisibility {
    Public,
    Private,
}

pub struct AddOp {
    operation: Shared<Operation>,
}

impl Op for AddOp {
    fn operation_name() -> OperationName {
        OperationName::new("add".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        AddOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn prefixes(&self) -> Prefixes {
        PREFIXES
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "(i32.add ")?;
        let operands = self.operation.rd().operands().vec();
        write!(f, "(local.get {}) ", operands.rd().first().unwrap().rd())?;
        write!(f, "(local.get {}))", operands.rd().get(1).unwrap().rd())?;
        Ok(())
    }
}

pub struct FuncOp {
    operation: Shared<Operation>,
    identifier: Option<String>,
    pub sym_visibility: Option<SymVisibility>,
}

impl FuncOp {
    pub fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    pub fn set_identifier(&mut self, identifier: Option<String>) {
        self.identifier = identifier;
    }
    pub fn arguments(&self) -> Values {
        self.operation.rd().arguments()
    }
}

fn display_argument(value: &Value) -> String {
    match value {
        Value::BlockArgument(arg) => {
            if let BlockArgumentName::Name(name) = &arg.name {
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

fn display_return_type(typ: Shared<dyn Type>) -> String {
    format!("(result {})", typ.rd())
}

fn display_return_types(types: &Types) -> String {
    let result = types
        .types
        .iter()
        .map(|typ| display_return_type(typ.clone()))
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
    fn is_func(&self) -> bool {
        true
    }
    fn prefixes(&self) -> Prefixes {
        PREFIXES
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        write!(f, "({} ", self.operation.rd().name())?;
        let identifier = self.identifier().expect("identifier not set");
        let identifier = canonicalize_identifier(&identifier);
        if self.sym_visibility == Some(SymVisibility::Public) {
            write!(f, "(export \"{identifier}\") ")?;
        }
        write!(f, "{}", display_arguments(&self.arguments()))?;
        let return_types = self.operation.rd().results().types();
        if !return_types.types.is_empty() {
            writeln!(f, " {}", display_return_types(&return_types))?;
        } else {
            writeln!(f)?;
        };
        for block in self.operation.rd().blocks().vec().rd().iter() {
            block.rd().display(f, indent + 1)?;
        }
        write!(f, "{})", crate::ir::spaces(indent))?;
        Ok(())
    }
}

pub struct ModuleOp {
    operation: Shared<Operation>,
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
    fn prefixes(&self) -> Prefixes {
        PREFIXES
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        writeln!(f, "({}", self.operation.rd().name())?;
        if let Some(region) = self.operation().rd().region() {
            region.rd().refresh_names();
            region.rd().display(f, indent)?;
        }
        write!(f, "{})", crate::ir::spaces(indent))?;
        Ok(())
    }
}

impl Display for ModuleOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

pub struct ReturnOp {
    operation: Shared<Operation>,
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("return".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        ReturnOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn prefixes(&self) -> Prefixes {
        PREFIXES
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation.rd().name())
    }
}

impl Display for ReturnOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
