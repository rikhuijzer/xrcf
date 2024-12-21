use crate::ir::BlockArgumentName;
use crate::ir::GuardedOp;
use crate::ir::GuardedOpOperand;
use crate::ir::GuardedOperation;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Value;
use crate::ir::Values;
use crate::shared::Shared;
use crate::shared::SharedExt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone, PartialEq)]
pub enum BlockName {
    /// The block has no label.
    Unnamed,
    /// The name of the block.
    Name(String),
    /// The block has a label, but it is not set yet.
    Unset,
}

/// A collection of [Op]s.
///
/// A block is always nested below a [Region].
pub struct Block {
    /// The label of the block.
    ///
    /// The label is only used during parsing to see which operands point to
    /// this block. During printing, a new name is generated.
    label: Arc<RwLock<BlockName>>,
    /// The prefix for the label (defaults to `^`).
    label_prefix: String,
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
        label: Arc<RwLock<BlockName>>,
        arguments: Values,
        ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>,
        parent: Option<Arc<RwLock<Region>>>,
    ) -> Self {
        Self {
            label,
            label_prefix: "^".to_string(),
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
    pub fn label(&self) -> Arc<RwLock<BlockName>> {
        self.label.clone()
    }
    pub fn label_prefix(&self) -> String {
        self.label_prefix.clone()
    }
    pub fn set_label(&self, label: BlockName) {
        *self.label.wr() = label;
    }
    pub fn set_label_prefix(&mut self, label_prefix: String) {
        self.label_prefix = label_prefix;
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
        let label = match &*self.label().rd() {
            BlockName::Unnamed => return None,
            // We can find callers via `Value::BlockLabel`.
            BlockName::Name(label) => Some(label.clone()),
            // We can still find callers via `Value::BlockPtr`.
            BlockName::Unset => None,
        };
        let mut callers = vec![];
        for p in self.predecessors().expect("no predecessors") {
            for op in p.rd().ops().rd().iter() {
                for operand in op.operation().operands().into_iter() {
                    match &*operand.value().rd() {
                        Value::BlockPtr(block_ptr) => {
                            let current = block_ptr.block();
                            let current = &*current.rd();
                            if std::ptr::eq(current, self) {
                                callers.push(op.clone());
                            };
                        }
                        Value::BlockLabel(block_label) => {
                            if let Some(label) = &label {
                                let current = canonicalize_label(&block_label.name());
                                if current == canonicalize_label(label) {
                                    callers.push(op.clone());
                                }
                            }
                        }
                        _ => {}
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
        let region = region.rd();
        let index = region.index_of(self);
        let blocks = region.blocks();
        let predecessors = match index {
            Some(index) => blocks.into_iter().take(index).collect(),
            None => blocks.into_iter().collect(),
        };
        Some(predecessors)
    }
    /// Return successors of the current block.
    ///
    /// Panics if the current block cannot be found in the parent region.
    pub fn successors(&self) -> Option<Vec<Arc<RwLock<Block>>>> {
        let region = self.parent();
        let region = region.expect("no parent");
        let region = region.rd();
        let index = region.index_of(self);
        let blocks = region.blocks();
        let successors = match index {
            Some(index) => blocks.into_iter().skip(index + 1).collect(),
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
        let region = region.rd();
        let parent = region.parent();
        assert!(
            parent.is_some(),
            "Found no parent for region {region} when searching for assignment of {name}"
        );
        let op = parent.unwrap();
        let op = &*op.rd();
        let operation = op.operation();
        if op.is_func() {
            for argument in operation.rd().arguments().into_iter() {
                match &*argument.rd() {
                    Value::BlockArgument(arg) => match &*arg.name().rd() {
                        BlockArgumentName::Name(curr) => {
                            if curr == name {
                                return Some(argument.clone());
                            }
                        }
                        _ => {}
                    },
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
        for op in self.ops().rd().iter() {
            for value in op.rd().assignments().unwrap().into_iter() {
                match &*value.rd() {
                    Value::BlockArgument(_block_argument) => {
                        // Ignore this case because we are looking for
                        // assignment in ops.
                        return None;
                    }
                    Value::BlockLabel(_) => continue,
                    Value::BlockPtr(_) => continue,
                    Value::Constant(_) => continue,
                    Value::FuncResult(_) => return None,
                    Value::OpResult(op_result) => {
                        if let Some(curr) = &*op_result.name().rd() {
                            if curr == name {
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
        for argument in self.arguments().into_iter() {
            match &*argument.rd() {
                Value::BlockArgument(arg) => {
                    if let BlockArgumentName::Name(curr) = &*arg.name().rd() {
                        if curr == name {
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
        for p in self.predecessors().unwrap().iter() {
            let p = p.rd();
            if let Some(value) = p.assignment_in_func_arguments(name) {
                return Some(value);
            }
            if let Some(value) = p.assignment_in_ops(name) {
                return Some(value);
            }
            if let Some(value) = p.assignment_in_block_arguments(name) {
                return Some(value);
            }
        }
        None
    }
    /// Return index of `op` in `self`.
    ///
    /// Returns `None` if `op` is not found in `self`.
    pub fn index_of(&self, op: &Operation) -> Option<usize> {
        self.ops().rd().iter().position(|current| {
            let current = current.rd();
            let current = current.operation();
            let current = &*current.rd();
            std::ptr::eq(current, op)
        })
    }
    /// Move the blocks that belong to `region` before `self`.
    ///
    /// The caller is in charge of transferring the control flow to the region
    /// and pass it the correct block arguments.
    pub fn inline_region_before(&self, region: Arc<RwLock<Region>>) {
        self.parent()
            .expect("no parent")
            .rd()
            .blocks()
            .splice(self, region.rd().blocks());
    }
    pub fn insert_op(&self, op: Arc<RwLock<dyn Op>>, index: usize) {
        self.ops.wr().insert(index, op);
    }
    pub fn insert_after(&self, earlier: Arc<RwLock<Operation>>, later: Arc<RwLock<dyn Op>>) {
        match self.index_of(&*earlier.rd()) {
            Some(index) => self.insert_op(later, index + 1),
            None => {
                panic!("Could not find op in block during insert_after");
            }
        }
    }
    pub fn insert_before(&self, earlier: Arc<RwLock<dyn Op>>, later: Arc<RwLock<Operation>>) {
        match self.index_of(&*later.rd()) {
            Some(index) => self.insert_op(earlier, index),
            None => panic!("could not find op in block"),
        }
    }
    pub fn replace(&self, old: Arc<RwLock<Operation>>, new: Arc<RwLock<dyn Op>>) {
        match self.index_of(&*old.rd()) {
            Some(index) => self.ops().wr()[index] = new,
            None => panic!("could not find op in block"),
        }
    }
    pub fn remove(&self, op: Arc<RwLock<Operation>>) {
        match self.index_of(&*op.rd()) {
            Some(index) => self.ops().wr().remove(index),
            None => panic!("could not find op in block"),
        };
    }
    fn set_arguments(&mut self, arguments: Values) {
        self.arguments = arguments;
    }
    pub fn used_names(&self) -> Vec<String> {
        let ops = self.ops();
        let ops = ops.rd();
        let mut used_names = vec![];
        for op in ops.iter() {
            used_names.extend(op.rd().operation().rd().result_names());
        }
        used_names
    }
    fn used_names_with_predecessors(&self) -> Vec<String> {
        let mut used_names = self.used_names();
        if let Some(predecessors) = self.predecessors() {
            for p in predecessors.iter() {
                used_names.extend(p.rd().used_names());
            }
        }
        used_names
    }
    /// Find a unique name for a value (for example, `%4 = ...`).
    pub fn unique_value_name(&self, prefix: &str) -> String {
        let mut new_name: i32 = -1;
        for name in self.used_names_with_predecessors().iter() {
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
        if let BlockName::Name(name) = &*self.label.rd() {
            let label_indent = if indent > 0 { indent - 1 } else { 0 };
            write!(f, "{}{name}", crate::ir::spaces(label_indent))?;
            let arguments = self.arguments();
            if !arguments.is_empty() {
                write!(f, "({arguments})")?;
            }
            write!(f, ":\n")?;
        }
        for op in self.ops().rd().iter() {
            write!(f, "{}", crate::ir::spaces(indent))?;
            op.rd().display(f, indent)?;
            write!(f, "\n")?;
        }
        Ok(())
    }
}

impl Default for Block {
    fn default() -> Self {
        let label = Shared::new(BlockName::Unnamed.into());
        let arguments = Values::default();
        let ops = Shared::new(vec![].into());
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
        self.block.wr().set_parent(parent);
        self.block.clone()
    }
}

pub trait GuardedBlock {
    fn arguments(&self) -> Values;
    fn callers(&self) -> Option<Vec<Arc<RwLock<dyn Op>>>>;
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result;
    fn index_of(&self, op: &Operation) -> Option<usize>;
    fn inline_region_before(&self, region: Arc<RwLock<Region>>);
    fn insert_after(&self, earlier: Arc<RwLock<Operation>>, later: Arc<RwLock<dyn Op>>);
    fn label(&self) -> Arc<RwLock<BlockName>>;
    fn label_prefix(&self) -> String;
    fn ops(&self) -> Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>;
    fn parent(&self) -> Option<Arc<RwLock<Region>>>;
    fn predecessors(&self) -> Option<Vec<Arc<RwLock<Block>>>>;
    fn remove(&self, op: Arc<RwLock<Operation>>);
    fn set_arguments(&self, arguments: Values);
    fn set_label(&self, label: BlockName);
    fn set_label_prefix(&self, label_prefix: String);
    fn set_ops(&self, ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>);
    fn successors(&self) -> Option<Vec<Arc<RwLock<Block>>>>;
    fn unique_value_name(&self, prefix: &str) -> String;
}

impl GuardedBlock for Arc<RwLock<Block>> {
    fn arguments(&self) -> Values {
        self.rd().arguments()
    }
    fn callers(&self) -> Option<Vec<Arc<RwLock<dyn Op>>>> {
        self.rd().callers()
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        self.rd().display(f, indent)
    }
    fn index_of(&self, op: &Operation) -> Option<usize> {
        self.rd().index_of(op)
    }
    fn inline_region_before(&self, region: Arc<RwLock<Region>>) {
        self.rd().inline_region_before(region);
    }
    fn insert_after(&self, earlier: Arc<RwLock<Operation>>, later: Arc<RwLock<dyn Op>>) {
        self.wr().insert_after(earlier, later);
    }
    fn label(&self) -> Arc<RwLock<BlockName>> {
        self.rd().label()
    }
    fn label_prefix(&self) -> String {
        self.rd().label_prefix()
    }
    fn ops(&self) -> Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>> {
        self.rd().ops()
    }
    fn parent(&self) -> Option<Arc<RwLock<Region>>> {
        self.rd().parent()
    }
    fn predecessors(&self) -> Option<Vec<Arc<RwLock<Block>>>> {
        self.rd().predecessors()
    }
    fn remove(&self, op: Arc<RwLock<Operation>>) {
        self.wr().remove(op);
    }
    fn set_arguments(&self, arguments: Values) {
        self.wr().set_arguments(arguments);
    }
    fn set_label(&self, label: BlockName) {
        self.wr().set_label(label);
    }
    fn set_label_prefix(&self, label_prefix: String) {
        self.wr().set_label_prefix(label_prefix);
    }
    fn set_ops(&self, ops: Arc<RwLock<Vec<Arc<RwLock<dyn Op>>>>>) {
        self.wr().set_ops(ops);
    }
    fn successors(&self) -> Option<Vec<Arc<RwLock<Block>>>> {
        self.rd().successors()
    }
    fn unique_value_name(&self, prefix: &str) -> String {
        self.rd().unique_value_name(prefix)
    }
}

#[derive(Clone)]
pub struct Blocks {
    vec: Arc<RwLock<Vec<Arc<RwLock<Block>>>>>,
}

impl IntoIterator for Blocks {
    type Item = Arc<RwLock<Block>>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let vec = self.vec.rd();
        vec.clone().into_iter()
    }
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
        let vec = vec.rd();
        if vec.is_empty() {
            panic!("Trying to find block in empty set of blocks");
        }
        vec.iter().position(|b| {
            let b = &*b.rd();
            std::ptr::eq(b, block)
        })
    }
    fn transfer(&self, before: &Block, blocks: Blocks) {
        let index = self.index_of(before);
        let index = match index {
            Some(index) => index,
            None => {
                panic!("Could not find block in blocks during transfer");
            }
        };
        let vec = self.vec();
        let mut vec = vec.wr();
        let blocks = blocks.vec();
        let mut blocks = blocks.wr();
        vec.splice(index..index, blocks.iter().cloned());
        {
            let parent = before.parent();
            for block in blocks.iter() {
                let mut block = block.wr();
                block.set_parent(parent.clone());
            }
        }
        blocks.clear();
    }
    /// Move `blocks` before `before` in `self`.
    ///
    /// This also handles side-effects like updating parents.
    pub fn splice(&self, before: &Block, blocks: Blocks) {
        self.transfer(before, blocks);
    }
}
