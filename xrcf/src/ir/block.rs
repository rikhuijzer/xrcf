use crate::ir::BlockArgumentName;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Value;
use crate::ir::Values;
use crate::shared::Shared;
use crate::shared::SharedExt;
use std::fmt::Display;
use std::fmt::Formatter;

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
    pub label: BlockName,
    arguments: Values,
    pub ops: Vec<Shared<dyn Op>>,
    /// Region has to be `Shared` to allow the parent to be moved.
    pub parent: Option<Shared<Region>>,
}

/// Canonicalize a block label.
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
        label: BlockName,
        arguments: Values,
        ops: Vec<Shared<dyn Op>>,
        parent: Option<Shared<Region>>,
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
    pub fn callers(&self) -> Option<Vec<Shared<dyn Op>>> {
        let label = match &self.label {
            BlockName::Unnamed => return None,
            // We can find callers via `Value::BlockLabel`.
            BlockName::Name(label) => Some(label.clone()),
            // We can still find callers via `Value::BlockPtr`.
            BlockName::Unset => None,
        };
        let mut callers = vec![];
        for p in self.predecessors().expect("no predecessors") {
            for op in p.rd().ops.iter() {
                for operand in op.rd().operation().rd().operands().into_iter() {
                    match &*operand.rd().value().rd() {
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
    pub fn predecessors(&self) -> Option<Vec<Shared<Block>>> {
        let region = self.parent.as_ref();
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
    pub fn successors(&self) -> Option<Vec<Shared<Block>>> {
        let region = self.parent.as_ref();
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
    pub fn assignment_in_func_arguments(&self, name: &str) -> Option<Shared<Value>> {
        let region = self.parent.clone();
        let region = region.expect("no parent");
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
                    #[allow(clippy::single_match)]
                    Value::BlockArgument(arg) => match &arg.name {
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
    pub fn assignment_in_ops(&self, name: &str) -> Option<Shared<Value>> {
        for op in self.ops.iter() {
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
                        if let Some(curr) = &op_result.name() {
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
    pub fn assignment_in_block_arguments(&self, name: &str) -> Option<Shared<Value>> {
        for argument in self.arguments().into_iter() {
            match &*argument.rd() {
                Value::BlockArgument(arg) => {
                    if let BlockArgumentName::Name(curr) = &arg.name {
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
    pub fn assignment(&self, name: &str) -> Option<Shared<Value>> {
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
        self.ops.iter().position(|current| {
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
    pub fn inline_region_before(&self, region: Shared<Region>) {
        self.parent
            .as_ref()
            .expect("no parent")
            .rd()
            .blocks()
            .splice(self, region.rd().blocks());
    }
    pub fn insert_op(&mut self, op: Shared<dyn Op>, index: usize) {
        self.ops.insert(index, op);
    }
    pub fn insert_after(&mut self, earlier: Shared<Operation>, later: Shared<dyn Op>) {
        match self.index_of(&earlier.rd()) {
            Some(index) => self.insert_op(later, index + 1),
            None => {
                panic!("Could not find op in block during insert_after");
            }
        }
    }
    pub fn insert_before(&mut self, earlier: Shared<dyn Op>, later: Shared<Operation>) {
        match self.index_of(&later.rd()) {
            Some(index) => self.insert_op(earlier, index),
            None => panic!("could not find op in block"),
        }
    }
    pub fn replace(&mut self, old: Shared<Operation>, new: Shared<dyn Op>) {
        match self.index_of(&old.rd()) {
            Some(index) => self.ops[index] = new,
            None => panic!("could not find op in block"),
        }
    }
    pub fn remove(&mut self, op: Shared<Operation>) {
        match self.index_of(&op.rd()) {
            Some(index) => self.ops.remove(index),
            None => panic!("could not find op in block"),
        };
    }
    pub fn set_arguments(&mut self, arguments: Values) {
        self.arguments = arguments;
    }
    pub fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        if let BlockName::Name(name) = &self.label {
            let label_indent = if indent > 0 { indent - 1 } else { 0 };
            write!(f, "{}{name}", crate::ir::spaces(label_indent))?;
            let arguments = self.arguments();
            if !arguments.is_empty() {
                write!(f, "({arguments})")?;
            }
            writeln!(f, ":")?;
        }
        for op in self.ops.iter() {
            write!(f, "{}", crate::ir::spaces(indent))?;
            op.rd().display(f, indent)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Default for Block {
    fn default() -> Self {
        let label = BlockName::Unnamed;
        let arguments = Values::default();
        let ops = vec![];
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
    block: Shared<Block>,
}

impl UnsetBlock {
    pub fn new(block: Shared<Block>) -> Self {
        Self { block }
    }
    pub fn block(&self) -> Shared<Block> {
        self.block.clone()
    }
    pub fn set_parent(&self, parent: Option<Shared<Region>>) -> Shared<Block> {
        self.block.wr().parent = parent;
        self.block.clone()
    }
}

#[derive(Clone)]
pub struct Blocks {
    vec: Shared<Vec<Shared<Block>>>,
}

impl IntoIterator for Blocks {
    type Item = Shared<Block>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let vec = self.vec.rd();
        vec.clone().into_iter()
    }
}

impl Blocks {
    pub fn new(vec: Shared<Vec<Shared<Block>>>) -> Self {
        Self { vec }
    }
    pub fn vec(&self) -> Shared<Vec<Shared<Block>>> {
        self.vec.clone()
    }
    /// Return the index of `block` in `self`.
    ///
    /// Returns `None` if `block` is not found in `self`.
    pub fn index_of(&self, block: &Block) -> Option<usize> {
        let vec = self.vec();
        let vec = vec.rd();
        if vec.is_empty() {
            return None;
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
            let parent = before.parent.clone();
            for block in blocks.iter() {
                let mut block = block.wr();
                block.parent = parent.clone();
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
