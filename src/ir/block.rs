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
    pub fn assignment_in_func_arguments(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        todo!()
    }
    pub fn assignment_in_ops(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        let ops = self.ops();
        let ops = ops.read().unwrap();
        println!("\nname: {}", name);
        for op in ops.iter() {
            let op = op.read().unwrap();
            let values = op.assignments();
            assert!(values.is_ok());
            let values = values.unwrap();
            let values = values.read().unwrap();
            let operation = op.operation();
            let operation = operation.read().unwrap();
            println!(
                "Found {} values for {name} and op {}",
                values.len(),
                operation.name()
            );
            for value in values.iter() {
                println!("iter operand");
                match &*value.read().unwrap() {
                    Value::BlockArgument(block_argument) => {
                        println!("block_argument: {}", block_argument.name());
                        if block_argument.name() == name {
                            return Some(value.clone());
                        }
                    }
                    Value::OpResult(op_result) => {
                        println!("op_result: {}", op_result.name());
                        if op_result.name() == name {
                            return Some(value.clone());
                        }
                    }
                }
            }
        }
        None
    }
    pub fn assignment(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        let from_arguments = self.assignment_in_func_arguments(name);
        match from_arguments {
            Some(value) => Some(value),
            None => self.assignment_in_ops(name),
        }
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
