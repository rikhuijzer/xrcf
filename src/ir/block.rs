use crate::ir::Op;
use crate::ir::Region;
use crate::ir::Value;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

pub struct Block {
    label: Option<String>,
    arguments: Arc<Vec<Value>>,
    ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>,
    parent: Arc<RwLock<Option<Region>>>,
}

impl Block {
    pub fn new(
        label: Option<String>,
        arguments: Arc<Vec<Value>>,
        ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>,
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
    pub fn ops(&self) -> Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>> {
        self.ops.clone()
    }
    pub fn ops_mut(&mut self) -> &mut Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>> {
        &mut self.ops
    }
    pub fn parent(&self) -> Arc<RwLock<Option<Region>>> {
        self.parent.clone()
    }
    pub fn assignment(&self, name: String) -> Option<Arc<RwLock<Value>>> {
        let ops = self.ops();
        let ops = ops.read().unwrap();
        for op in ops.iter() {
            let op = op.read().unwrap();
            let operation = op.operation();
            let operands = operation.read().unwrap().operands();
            let operands = operands.read().unwrap();
            for operand in operands.iter() {
                let other_operand = operand.clone();
                let write = other_operand.write().unwrap();
                let value = other_operand.write().unwrap().value();
                let value = value.clone().read().unwrap();
                match value {
                    Value::BlockArgument(block_argument) => {
                        if block_argument.name() == name {
                            return Some(write.value().clone());
                        }
                    }
                    Value::OpResult(_) => {
                        return None;
                    }
                }
            }
        }
        None
    }
    pub fn display(&self, f: &mut std::fmt::Formatter<'_>, indent: i32) -> std::fmt::Result {
        if let Some(label) = &self.label {
            write!(f, "{} ", label)?;
        }
        let ops = self.ops();
        let ops = ops.read().unwrap();
        for op in ops.iter() {
            let spaces = crate::ir::spaces(indent);
            write!(f, "{spaces}")?;
            let op = op.read().unwrap();
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
