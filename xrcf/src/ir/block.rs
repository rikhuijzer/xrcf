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
        if op.is_func() {
            let arguments = operation.arguments();
            let arguments = arguments.vec();
            let arguments = arguments.try_read().unwrap();
            for argument in arguments.iter() {
                match &*argument.try_read().unwrap() {
                    Value::BlockArgument(block_argument) => {
                        if block_argument.name() == Some(name.to_string()) {
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
            let values = values.vec();
            let values = values.try_read().unwrap();
            for value in values.iter() {
                match &*value.try_read().unwrap() {
                    Value::BlockArgument(_block_argument) => {
                        // Ignore this case because we are looking for
                        // assignment in ops.
                        return None;
                    }
                    Value::FuncResult(_) => return None,
                    Value::OpResult(op_result) => {
                        if op_result.name().expect("OpResult has no name") == name {
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
    pub fn index_of(&self, op: Arc<RwLock<Operation>>) -> Option<usize> {
        let ops = self.ops();
        let ops = ops.read().unwrap();
        for (i, current) in (&ops).iter().enumerate() {
            let current = current.read().unwrap();
            let current = current.operation();
            if Arc::ptr_eq(current, &op) {
                return Some(i);
            }
        }
        None
    }
    pub fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>, later: Arc<RwLock<Operation>>) {
        let index = self.index_of(later);
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Could not find op in block");
            }
        };
        let ops = self.ops();
        let mut ops = ops.try_write().unwrap();
        ops.insert(index, earlier);
    }
    pub fn insert_after(&self, earlier: Arc<RwLock<Operation>>, later: Arc<RwLock<dyn Op>>) {
        let index = self.index_of(earlier);
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Could not find op in block");
            }
        };
        let ops = self.ops();
        let mut ops = ops.try_write().unwrap();
        ops.insert(index + 1, later);
    }
    pub fn replace(&self, old: Arc<RwLock<Operation>>, new: Arc<RwLock<dyn Op>>) {
        let index = self.index_of(old.clone());
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Replace could not find op in block");
            }
        };
        let ops = self.ops();
        let mut ops = ops.try_write().unwrap();
        ops[index] = new;
    }
    pub fn remove(&self, op: Arc<RwLock<Operation>>) {
        let index = self.index_of(op);
        match index {
            Some(index) => {
                let ops = self.ops();
                let mut ops = ops.try_write().unwrap();
                ops.remove(index);
            }
            None => {
                panic!("Remove could not find op in block");
            }
        };
    }
    /// Find a unique name for a value (for example, `%4 = ...`).
    pub fn unique_value_name(&self) -> String {
        let mut new_name: i32 = 0;
        let ops = self.ops();
        let ops = ops.try_read().unwrap();
        for op in ops.iter() {
            let op = op.try_read().unwrap();
            let operation = op.operation();
            let operation = operation.try_read().unwrap();
            let results = operation.results();
            let results = results.vec();
            let results = results.try_read().unwrap();
            for result in results.iter() {
                let result = result.try_read().unwrap();
                let name = match &*result {
                    Value::OpResult(res) => res.name().expect("failed to get name"),
                    Value::BlockArgument(arg) => arg.name().expect("failed to get name"),
                    Value::FuncResult(_) => continue,
                };
                let possible_name = format!("%{new_name}");
                if name == possible_name {
                    new_name += 1;
                } else {
                    return possible_name;
                }
            }
        }
        format!("%{new_name}")
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
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}
