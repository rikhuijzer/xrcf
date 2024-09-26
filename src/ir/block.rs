use crate::ir::Op;
use crate::ir::Region;
use crate::ir::Value;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

pub struct Block {
    label: Option<String>,
    arguments: Arc<Vec<Value>>,
    ops: Arc<Vec<Arc<dyn Op>>>,
    parent: Arc<RwLock<Option<Region>>>,
}

impl Block {
    pub fn new(
        label: Option<String>,
        arguments: Arc<Vec<Value>>,
        ops: Arc<Vec<Arc<dyn Op>>>,
        parent: Arc<RwLock<Option<Region>>>,
    ) -> Self {
        Self {
            label,
            arguments,
            ops,
            parent,
        }
    }
    pub fn arguments(&self) -> Arc<Vec<Value>> {
        self.arguments.clone()
    }
    pub fn ops(&self) -> Arc<Vec<Arc<dyn Op>>> {
        self.ops.clone()
    }
    pub fn ops_mut(&mut self) -> &mut Arc<Vec<Arc<dyn Op>>> {
        &mut self.ops
    }
    pub fn parent(&self) -> Arc<RwLock<Option<Region>>> {
        self.parent.clone()
    }
    pub fn first_use(&self, name: String) -> Option<&Value> {
        for op in self.ops.iter() {
            let operation = op.operation();
            let operands = operation.read().unwrap().operands();
            for operand in operands.iter() {
                match operand {
                    Value::BlockArgument(block_argument) => {
                        if block_argument.name() == name {
                            return Some(&block_argument);
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }
    pub fn display(&self, f: &mut std::fmt::Formatter<'_>, indent: i32) -> std::fmt::Result {
        if let Some(label) = &self.label {
            write!(f, "{} ", label)?;
        }
        for op in self.ops.iter() {
            let spaces = crate::ir::spaces(indent);
            write!(f, "{spaces}")?;
            op.display(f, indent)?;
        }
        Ok(())
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
