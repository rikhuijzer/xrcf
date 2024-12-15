use crate::ir::BlockArgumentName;
use crate::ir::GuardedRegion;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Value;
use crate::ir::Values;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// A collection of [Op]s.
///
/// A block is always nested below a [Region].
pub struct Block {
    label: Option<String>,
    arguments: Values,
    ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>,
    /// This field does not have to be an `Arc<RwLock<..>>` because
    /// the `Block` is shared via `Arc<RwLock<..>>`.
    parent: Option<Arc<RwLock<Region>>>,
}

fn canonicalize_label(label: &str) -> String {
    label.trim_start_matches('^').to_string()
}

/// Two blocks are equal if they point to the same object.
///
/// This is used for things like finding `successors`.
impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Block {
    pub fn new(
        label: Option<String>,
        arguments: Values,
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
    pub fn arguments(&self) -> Values {
        self.arguments.clone()
    }
    pub fn ops(&self) -> Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>> {
        self.ops.clone()
    }
    pub fn set_ops(&mut self, ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>) {
        self.ops = ops;
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<Region>>>) {
        self.parent = parent;
    }
    pub fn ops_mut(&mut self) -> &mut Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>> {
        &mut self.ops
    }
    pub fn label(&self) -> Option<String> {
        self.label.clone()
    }
    pub fn set_label(&mut self, label: Option<String>) {
        self.label = label;
    }
    pub fn parent(&self) -> Option<Arc<RwLock<Region>>> {
        self.parent.clone()
    }
    /// Return callers of this block (i.e., ops that point to the current block).
    ///
    /// For example, when called on the block `^merge` in
    /// ```mlir
    /// ^else:
    ///   %c1 = arith.constant 1 : i32
    ///   llvm.br ^merge(%c1 : i32)
    /// ^merge(%result : i32):
    /// ```
    /// this method will return the operation `llvm.br`.
    pub fn callers(&self) -> Option<Vec<Arc<RwLock<dyn Op>>>> {
        let label = self.label();
        if label.is_none() {
            return None;
        }
        let label = label.unwrap();
        let predecessors = self.predecessors();
        let predecessors = predecessors.expect("expected predecessors");
        let mut callers = vec![];
        for predecessor in predecessors.iter() {
            let predecessor = predecessor.try_read().unwrap();
            let ops = predecessor.ops();
            let ops = ops.try_read().unwrap();
            for op in ops.iter() {
                let op_read = op.try_read().unwrap();
                let dest = op_read.block_destination();
                if dest.is_some() {
                    let dest = dest.unwrap();
                    let dest = dest.try_read().unwrap();
                    if canonicalize_label(&dest.name()) == canonicalize_label(&label) {
                        callers.push(op.clone());
                    }
                }
            }
        }
        Some(callers)
    }
    /// Return predecessors of the current block.
    ///
    /// Returns all known blocks if the current block cannot be found in the
    /// parent region. This is because the current block may currently be in the
    /// process of being parsed (i.e., not yet ready to be added to the
    /// collection of blocks).
    pub fn predecessors(&self) -> Option<Vec<Arc<RwLock<Block>>>> {
        let region = self.parent();
        let region = region.expect("no parent");
        let region = region.try_read().unwrap();
        let index = region.index_of(self);
        let blocks = region.blocks();
        let blocks = blocks.vec();
        let predecessors = blocks.try_read().unwrap();
        let predecessors = match index {
            Some(index) => predecessors[..index].to_vec(),
            None => predecessors.clone(),
        };
        Some(predecessors)
    }
    /// Return successors of the current block.
    ///
    /// Panics if the current block cannot be found in the parent region.
    pub fn successors(&self) -> Option<Vec<Arc<RwLock<Block>>>> {
        let region = self.parent();
        let region = region.expect("no parent");
        let region = region.try_read().unwrap();
        let index = region.index_of(self);
        let blocks = region.blocks();
        let blocks = blocks.vec();
        let successors = blocks.try_read().unwrap();
        let successors = match index {
            Some(index) => successors[index + 1..].to_vec(),
            None => panic!("Expected block to be in region"),
        };
        Some(successors)
    }
    /// Return assignment of a value in the parent op's function arguments.
    ///
    /// Returns a [Value] if the parent operation is a function and contains an
    /// assignment for the given `name`. Return `None` if no assignment is
    /// found.
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
        let op = &*op.try_read().unwrap();
        let operation = op.operation();
        let operation = operation.try_read().unwrap();
        if op.is_func() {
            let arguments = operation.arguments();
            let arguments = arguments.vec();
            let arguments = arguments.try_read().unwrap();
            for argument in arguments.iter() {
                match &*argument.try_read().unwrap() {
                    Value::BlockArgument(block_argument) => {
                        let current_name = block_argument.name();
                        let current_name = current_name.try_read().unwrap();
                        if let BlockArgumentName::Name(current_name) = &*current_name {
                            if current_name == name {
                                return Some(argument.clone());
                            }
                        }
                    }
                    _ => panic!("Expected BlockArgument"),
                }
            }
        } else {
            // We are not in a function, but for example inside a region inside
            // `scf.if`. As far as I know, the regions inside `scf.if` do not
            // have access to the arguments of the parent function.
            return None;
        }
        None
    }
    pub fn assignment_in_ops(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        let ops = self.ops();
        let ops = ops.try_read().unwrap();
        for op in ops.iter() {
            let op = op.try_read().unwrap();
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
                    Value::BlockLabel(_) => continue,
                    Value::Constant(_) => continue,
                    Value::FuncResult(_) => return None,
                    Value::OpResult(op_result) => {
                        let current_name = op_result.name();
                        let current_name = current_name.try_read().unwrap();
                        if let Some(current_name) = &*current_name {
                            if current_name == name {
                                return Some(value.clone());
                            }
                        }
                    }
                    Value::Variadic => continue,
                }
            }
        }
        None
    }
    /// Return assignment of a value in the block arguments.
    ///
    /// Returns a [Value] if the block contains an assignment for the given
    /// `name`. Return `None` if no assignment is found.
    pub fn assignment_in_block_arguments(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        let arguments = self.arguments();
        let arguments = arguments.vec();
        let arguments = arguments.try_read().unwrap();
        for argument in arguments.iter() {
            match &*argument.try_read().unwrap() {
                Value::BlockArgument(block_argument) => {
                    let current_name = block_argument.name();
                    let current_name = current_name.try_read().unwrap();
                    if let BlockArgumentName::Name(current_name) = &*current_name {
                        if current_name == name {
                            return Some(argument.clone());
                        }
                    }
                }
                _ => panic!("Expected BlockArgument"),
            }
        }
        None
    }
    pub fn assignment(&self, name: &str) -> Option<Arc<RwLock<Value>>> {
        // Check the current block first because it is most likely to contain
        // the assignment.
        if let Some(value) = self.assignment_in_func_arguments(name) {
            return Some(value);
        }
        if let Some(value) = self.assignment_in_ops(name) {
            return Some(value);
        }
        if let Some(value) = self.assignment_in_block_arguments(name) {
            return Some(value);
        }

        // Check the predecessors.
        //
        // Taking dominance into account is not implemented yet. Dominance will
        // be necessary later in order to verify whether an operand dominates
        // the use. "dominate" in LLVM/MLIR essentially means whether the code
        // that assigns a variable was executed before the use of the variable.
        // An exception to this is during an if-else statement where the
        // assignment is in the "then" block and the use is after the end of the
        // if-else statement. In this case, the value should not be used since
        // it is not guaranteed to have been initialized.
        //
        // A talk with much more details is called "(Correctly) Extending
        // Dominance to MLIR Regions" and is available at
        // https://www.youtube.com/watch?v=VJORFvHJKWE.
        let predecessors = self.predecessors();
        let predecessors = predecessors.expect("expected predecessors");
        for predecessor in predecessors.iter() {
            let predecessor = predecessor.try_read().unwrap();
            if let Some(value) = predecessor.assignment_in_func_arguments(name) {
                return Some(value);
            }
            if let Some(value) = predecessor.assignment_in_ops(name) {
                return Some(value);
            }
            if let Some(value) = predecessor.assignment_in_block_arguments(name) {
                return Some(value);
            }
        }
        None
    }
    /// Return index of `op` in `self`.
    ///
    /// Returns `None` if `op` is not found in `self`.
    pub fn index_of(&self, op: &Operation) -> Option<usize> {
        let ops = self.ops();
        let ops = ops.try_read().unwrap();
        ops.iter().position(|current| {
            let current = current.try_read().unwrap();
            let current = current.operation();
            let current = &*current.try_read().unwrap();
            std::ptr::eq(current, op)
        })
    }
    pub fn index_of_arc(&self, op: Arc<RwLock<Operation>>) -> Option<usize> {
        self.index_of(&*op.try_read().unwrap())
    }
    /// Move the blocks that belong to `region` before `self`.
    ///
    /// The caller is in charge of transferring the control flow to the region
    /// and pass it the correct block arguments.
    pub fn inline_region_before(&self, region: Arc<RwLock<Region>>) {
        let parent = self.parent();
        let parent = parent.expect("no parent");
        let blocks = parent.blocks();
        blocks.splice(self, region.blocks());
    }
    pub fn insert_op(&self, op: Arc<RwLock<dyn Op>>, index: usize) {
        let ops = self.ops();
        let mut ops = ops.try_write().unwrap();
        ops.insert(index, op);
    }
    pub fn insert_after(&self, earlier: Arc<RwLock<Operation>>, later: Arc<RwLock<dyn Op>>) {
        let index = self.index_of_arc(earlier.clone());
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Could not find op in block during insert_after");
            }
        };
        self.insert_op(later, index + 1);
    }
    pub fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>, later: Arc<RwLock<Operation>>) {
        let index = self.index_of_arc(later);
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Could not find op in block during insert_before");
            }
        };
        self.insert_op(earlier, index);
    }
    pub fn replace(&self, old: Arc<RwLock<Operation>>, new: Arc<RwLock<dyn Op>>) {
        let index = self.index_of_arc(old);
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Replace could not find op in block during replace");
            }
        };
        let ops = self.ops();
        let mut ops = ops.try_write().unwrap();
        ops[index] = new;
    }
    pub fn remove(&self, op: Arc<RwLock<Operation>>) {
        let index = self.index_of_arc(op);
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
    fn set_arguments(&mut self, arguments: Values) {
        self.arguments = arguments;
    }
    pub fn used_names_without_predecessors(&self) -> Vec<String> {
        let ops = self.ops();
        let ops = ops.try_read().unwrap();
        let mut used_names = vec![];
        for op in ops.iter() {
            let op = op.try_read().unwrap();
            let operation = op.operation();
            let operation = operation.try_read().unwrap();
            let result_names = operation.result_names();
            used_names.extend(result_names);
        }
        used_names
    }
    fn used_names_with_predecessors(&self) -> Vec<String> {
        let predecessors = self.predecessors();
        let mut used_names = self.used_names_without_predecessors();
        if let Some(predecessors) = predecessors {
            for predecessor in predecessors.iter() {
                let predecessor = predecessor.try_read().unwrap();
                used_names.extend(Self::used_names_without_predecessors(&predecessor));
            }
        }
        used_names
    }
    /// Find a unique name for a value (for example, `%4 = ...`).
    pub fn unique_value_name(&self, prefix: &str) -> String {
        let used_names = self.used_names_with_predecessors();
        let mut new_name: i32 = -1;
        for name in used_names.iter() {
            let name = name.trim_start_matches(prefix);
            if let Ok(num) = name.parse::<i32>() {
                // Ensure new_name is greater than any used name.
                // This is required by LLVM.
                new_name = new_name.max(num);
            }
        }
        new_name += 1;
        format!("{prefix}{new_name}")
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        if let Some(label) = &self.label {
            let label_indent = if indent > 0 { indent - 1 } else { 0 };
            let spaces = crate::ir::spaces(label_indent);
            write!(f, "{spaces}{label}")?;
            let arguments = self.arguments();
            if !arguments.is_empty() {
                write!(f, "({arguments})")?;
            }
            write!(f, ":\n")?;
        }
        let ops = self.ops();
        let ops = ops.try_read().unwrap();
        for op in ops.iter() {
            let spaces = crate::ir::spaces(indent);
            write!(f, "{spaces}")?;
            let op = op.try_read().unwrap();
            op.display(f, indent)?;
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Default for Block {
    fn default() -> Self {
        let label = None;
        let arguments = Values::default();
        let ops = Arc::new(RwLock::new(vec![]));
        let parent = None;
        Self::new(label, arguments, ops, parent)
    }
}

impl Display for Block {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

#[must_use = "the object inside `UnsetBlock` should be further initialized, see the setter methods"]
pub struct UnsetBlock {
    block: Arc<RwLock<Block>>,
}

impl UnsetBlock {
    pub fn new(block: Arc<RwLock<Block>>) -> Self {
        Self { block }
    }
    pub fn block(&self) -> Arc<RwLock<Block>> {
        self.block.clone()
    }
    pub fn set_parent(&self, parent: Option<Arc<RwLock<Region>>>) -> Arc<RwLock<Block>> {
        let mut block = self.block.try_write().unwrap();
        block.set_parent(parent);
        self.block.clone()
    }
}

pub trait GuardedBlock {
    fn arguments(&self) -> Values;
    fn callers(&self) -> Option<Vec<Arc<RwLock<dyn Op>>>>;
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result;
    fn index_of(&self, op: &Operation) -> Option<usize>;
    fn index_of_arc(&self, op: Arc<RwLock<Operation>>) -> Option<usize>;
    fn inline_region_before(&self, region: Arc<RwLock<Region>>);
    fn insert_after(&self, earlier: Arc<RwLock<Operation>>, later: Arc<RwLock<dyn Op>>);
    fn label(&self) -> Option<String>;
    fn ops(&self) -> Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>;
    fn parent(&self) -> Option<Arc<RwLock<Region>>>;
    fn predecessors(&self) -> Option<Vec<Arc<RwLock<Block>>>>;
    fn remove(&self, op: Arc<RwLock<Operation>>);
    fn set_arguments(&self, arguments: Values);
    fn set_label(&self, label: Option<String>);
    fn set_ops(&self, ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>);
    fn successors(&self) -> Option<Vec<Arc<RwLock<Block>>>>;
    fn unique_value_name(&self, prefix: &str) -> String;
}

impl GuardedBlock for Arc<RwLock<Block>> {
    fn arguments(&self) -> Values {
        self.try_read().unwrap().arguments()
    }
    fn callers(&self) -> Option<Vec<Arc<RwLock<dyn Op>>>> {
        self.try_read().unwrap().callers()
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        self.try_read().unwrap().display(f, indent)
    }
    fn index_of(&self, op: &Operation) -> Option<usize> {
        self.try_read().unwrap().index_of(op)
    }
    fn index_of_arc(&self, op: Arc<RwLock<Operation>>) -> Option<usize> {
        self.try_read().unwrap().index_of_arc(op)
    }
    fn inline_region_before(&self, region: Arc<RwLock<Region>>) {
        self.try_read().unwrap().inline_region_before(region);
    }
    fn insert_after(&self, earlier: Arc<RwLock<Operation>>, later: Arc<RwLock<dyn Op>>) {
        self.try_write().unwrap().insert_after(earlier, later);
    }
    fn label(&self) -> Option<String> {
        self.try_read().unwrap().label()
    }
    fn ops(&self) -> Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>> {
        self.try_read().unwrap().ops()
    }
    fn parent(&self) -> Option<Arc<RwLock<Region>>> {
        self.try_read().unwrap().parent()
    }
    fn predecessors(&self) -> Option<Vec<Arc<RwLock<Block>>>> {
        self.try_read().unwrap().predecessors()
    }
    fn remove(&self, op: Arc<RwLock<Operation>>) {
        self.try_write().unwrap().remove(op);
    }
    fn set_arguments(&self, arguments: Values) {
        self.try_write().unwrap().set_arguments(arguments);
    }
    fn set_label(&self, label: Option<String>) {
        self.try_write().unwrap().set_label(label);
    }
    fn set_ops(&self, ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>) {
        self.try_write().unwrap().set_ops(ops);
    }
    fn successors(&self) -> Option<Vec<Arc<RwLock<Block>>>> {
        self.try_read().unwrap().successors()
    }
    fn unique_value_name(&self, prefix: &str) -> String {
        self.try_read().unwrap().unique_value_name(prefix)
    }
}

#[derive(Clone)]
pub struct Blocks {
    vec: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>,
}

impl Blocks {
    pub fn new(vec: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>) -> Self {
        Self { vec }
    }
    pub fn vec(&self) -> Arc<RwLock<Vec<Arc<RwLock<Block>>>>> {
        self.vec.clone()
    }
    /// Return the index of `block` in `self`.
    ///
    /// Returns `None` if `block` is not found in `self`.
    pub fn index_of(&self, block: &Block) -> Option<usize> {
        let vec = self.vec();
        let vec = vec.try_read().unwrap();
        vec.iter().position(|b| {
            let b = &*b.try_read().unwrap();
            std::ptr::eq(b, block)
        })
    }
    /// Move `blocks` before `before` in `self`.
    pub fn splice(&self, before: &Block, blocks: Blocks) {
        let blocks = blocks.vec();
        let blocks = blocks.try_read().unwrap();
        let index = self.index_of(before);
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Could not find block in blocks during splice");
            }
        };
        let vec = self.vec();
        let mut vec = vec.try_write().unwrap();
        vec.splice(index..index, blocks.iter().cloned());
    }
}
