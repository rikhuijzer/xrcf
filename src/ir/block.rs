use crate::dialect::func::FuncOp;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Value;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

pub struct Block {
    label: Option<String>,
    arguments: Arc<Vec<Value>>,
    ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>,
    /// This field does not have to be an `Arc<RwLock<..>>` because
    /// the `Block` is shared via `Arc<RwLock<..>>`.
    parent: Option<Arc<RwLock<Region>>>,
}

impl Block {
    pub fn new(
        label: Option<String>,
        arguments: Arc<Vec<Value>>,
        ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>,
        parent: Option<Arc<RwLock<Region>>>,
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
    pub fn parent(&self) -> Option<Arc<RwLock<Region>>> {
        self.parent.clone()
    }
    pub fn assignment_in_func_arguments(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        let region = self.parent();
        assert!(region.is_some());
        let region = region.unwrap();
        let region = region.read().unwrap();
        let parent = region.parent();
        assert!(
            parent.is_some(),
            "Found no parent for region {region} when searching for assignment of {name}"
        );
        let op = parent.unwrap();
        let op = &*op.read().unwrap();
        let operation = op.operation();
        let operation = operation.read().unwrap();
        if operation.name() == FuncOp::operation_name() {
            let arguments = operation.arguments();
            let arguments = arguments.read().unwrap();
            for argument in arguments.iter() {
                match &*argument.read().unwrap() {
                    Value::BlockArgument(block_argument) => {
                        if block_argument.name() == name {
                            return Some(argument.clone());
                        }
                    }
                    _ => panic!("Expected BlockArgument, but got OpResult"),
                }
            }
        } else {
            panic!(
                "Expected parent op to be FuncOp, but got {}",
                operation.name()
            );
        }
        None
    }
    pub fn assignment_in_ops(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        let ops = self.ops();
        let ops = ops.read().unwrap();
        for op in ops.iter() {
            let op = op.read().unwrap();
            let values = op.assignments();
            assert!(values.is_ok());
            let values = values.unwrap();
            let values = values.read().unwrap();
            for value in values.iter() {
                match &*value.read().unwrap() {
                    Value::BlockArgument(block_argument) => {
                        if block_argument.name() == name {
                            return Some(value.clone());
                        }
                    }
                    Value::OpResult(op_result) => {
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
    pub fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>, later: Arc<RwLock<Operation>>) {
        let ops = self.ops();
        let mut ops = ops.try_write().unwrap();
        let mut index = 0;
        for (i, current) in (&ops).iter().enumerate() {
            let current = current.try_read().unwrap();
            let current = current.operation();
            if Arc::ptr_eq(current, &later) {
                index = i;
                break;
            }
        }
        if index == 0 {
            panic!("Could not find op in block");
        }
        ops.insert(index, earlier);
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
