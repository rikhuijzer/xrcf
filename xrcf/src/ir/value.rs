use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOp;
use crate::ir::GuardedOperation;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::TypeConvert;
use crate::ir::Types;
use crate::ir::VariableRenamer;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Display;
use std::sync::Arc;
use std::sync::RwLock;

#[derive(Clone, PartialEq, Eq)]
pub enum BlockArgumentName {
    /// The name of the block argument.
    Name(String),
    /// Anonymous block arguments are used for functions without an implementation.
    Anonymous,
}

/// An argument in a block or function.
pub struct BlockArgument {
    /// The name of the block argument.
    ///
    /// The name is only used during parsing to see which operands point to this
    /// argument. During printing, a new name is generated.
    name: Option<BlockArgumentName>,
    typ: Arc<RwLock<dyn Type>>,
    /// The operation for which this [BlockArgument] is an argument.
    ///
    /// This is used by [value.users] to find the users of this
    /// [BlockArgument].
    parent: Option<Arc<RwLock<Block>>>,
}

impl BlockArgument {
    pub fn new(name: Option<BlockArgumentName>, typ: Arc<RwLock<dyn Type>>) -> Self {
        BlockArgument {
            name,
            typ,
            parent: None,
        }
    }
    pub fn name(&self) -> Option<BlockArgumentName> {
        self.name.clone()
    }
    pub fn parent(&self) -> Option<Arc<RwLock<Block>>> {
        self.parent.clone()
    }
    pub fn set_name(&mut self, name: Option<BlockArgumentName>) {
        self.name = name;
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<Block>>>) {
        self.parent = parent;
    }
    pub fn set_typ(&mut self, typ: Arc<RwLock<dyn Type>>) {
        self.typ = typ;
    }
    pub fn typ(&self) -> Arc<RwLock<dyn Type>> {
        self.typ.clone()
    }
}

impl Display for BlockArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let typ = self.typ.try_read().unwrap();
        if let Some(BlockArgumentName::Anonymous) = &self.name {
            write!(f, "{}", typ)
        } else {
            let parent = self.parent();
            let parent = parent.expect("no parent");
            let new_name = parent.unique_value_name("%arg");
            write!(f, "{} : {}", new_name, typ)
        }
    }
}

pub struct BlockLabel {
    name: String,
}

impl BlockLabel {
    pub fn new(name: String) -> Self {
        BlockLabel { name }
    }
    pub fn name(&self) -> String {
        self.name.clone()
    }
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }
}

impl Display for BlockLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A constant value, for example a constant integer.
///
/// This is useful for situations where a operand is replaced by a constant
/// value. Due to [Constant] being a [Value], it can be placed inside the
/// [Operation] `operands` field. In turn, this allows us to keep track of
/// the order of the operands.
pub struct Constant {
    value: Arc<dyn Attribute>,
}

impl Constant {
    pub fn new(value: Arc<dyn Attribute>) -> Self {
        Constant { value }
    }
    pub fn typ(&self) -> Arc<RwLock<dyn Type>> {
        self.value.typ()
    }
    pub fn value(&self) -> Arc<dyn Attribute> {
        self.value.clone()
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}

/// An unnamed result of an operation, such as a function.
///
/// This result does not specify a name since, for example, the following is
/// invalid:
///
/// ```mlir
/// %0 = func.func @foo() -> %0 : i64
/// ```
///
/// The reason that this is a [Value] is that it provides a way to set a type
/// for a function result. The alternative would be to move the [Type]s to a
/// separate [Operation] `return_types` field, but that is more error prone
/// since the field then has to be set manually each time an operation is
/// created.  This makes it more error prone than having it included in the
/// `results` field.
pub struct AnonymousResult {
    typ: Arc<RwLock<dyn Type>>,
}

impl AnonymousResult {
    pub fn new(typ: Arc<RwLock<dyn Type>>) -> Self {
        AnonymousResult { typ }
    }
    pub fn typ(&self) -> Arc<RwLock<dyn Type>> {
        self.typ.clone()
    }
    pub fn set_typ(&mut self, typ: Arc<RwLock<dyn Type>>) {
        self.typ = typ;
    }
}

impl Display for AnonymousResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.typ.try_read().unwrap())
    }
}

/// A named result of an operation.
///
/// For example, in the following code:
/// ```mlir
/// %0 = arith.addi %1, %2 : i32
/// ```
/// `%0` is the result of the operation and has the name `%0`. The `defining_op`
/// is `arith.addi` and the `typ` is `i32`.
pub struct OpResult {
    name: Option<String>,
    typ: Option<Arc<RwLock<dyn Type>>>,
    defining_op: Option<Arc<RwLock<dyn Op>>>,
}

impl OpResult {
    pub fn new(
        name: Option<String>,
        typ: Option<Arc<RwLock<dyn Type>>>,
        defining_op: Option<Arc<RwLock<dyn Op>>>,
    ) -> Self {
        OpResult {
            name,
            typ,
            defining_op,
        }
    }
    pub fn name(&self) -> Option<String> {
        self.name.clone()
    }
    pub fn typ(&self) -> Option<Arc<RwLock<dyn Type>>> {
        self.typ.clone()
    }
    pub fn defining_op(&self) -> Option<Arc<RwLock<dyn Op>>> {
        self.defining_op.clone()
    }
    pub fn set_name(&mut self, name: &str) {
        self.name = Some(name.to_string());
    }
    pub fn set_typ(&mut self, typ: Arc<RwLock<dyn Type>>) {
        self.typ = Some(typ);
    }
    pub fn set_defining_op(&mut self, op: Option<Arc<RwLock<dyn Op>>>) {
        self.defining_op = op;
    }
}

impl Default for OpResult {
    fn default() -> Self {
        Self {
            name: None,
            typ: None,
            defining_op: None,
        }
    }
}

impl Display for OpResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self.name.as_ref().expect("OpResult has no name");
        write!(f, "{}", name)
    }
}

#[must_use = "the object should be further initialized, see the setter methods"]
pub struct UnsetOpResult {
    result: Arc<RwLock<Value>>,
}

impl UnsetOpResult {
    pub fn new(result: Arc<RwLock<Value>>) -> Self {
        assert!(
            matches!(&*result.try_read().unwrap(), Value::OpResult(_)),
            "Expected OpResult"
        );
        UnsetOpResult { result }
    }
    pub fn set_defining_op(&self, op: Option<Arc<RwLock<dyn Op>>>) {
        self.result.try_write().unwrap().set_defining_op(op);
    }
    pub fn set_typ(&self, typ: Arc<RwLock<dyn Type>>) {
        self.result.try_write().unwrap().set_type(typ);
    }
}

#[must_use = "the object should be further initialized, see the setter methods"]
pub struct UnsetOpResults {
    results: Values,
}

impl UnsetOpResults {
    pub fn new(results: Values) -> Self {
        UnsetOpResults { results }
    }
    pub fn values(&self) -> Values {
        self.results.clone()
    }
    pub fn set_defining_op(&self, op: Arc<RwLock<dyn Op>>) {
        let values = self.values();
        values.set_defining_op(op);
    }
    pub fn set_types(&self, types: Vec<Arc<RwLock<dyn Type>>>) {
        let results = self.values();
        let results = results.vec();
        let results = results.try_read().unwrap();
        assert!(types.len() == results.len());
        for (result, typ) in results.iter().zip(types) {
            let mut result = result.try_write().unwrap();
            result.set_type(typ);
        }
    }
}

pub enum Users {
    /// The operation defines no `OpResult`s.
    HasNoOpResults,
    /// The operation defines `OpResult`s (and can still have zero users).
    OpOperands(Vec<Arc<RwLock<OpOperand>>>),
}

impl Users {
    pub fn len(&self) -> usize {
        match self {
            Users::HasNoOpResults => 0,
            Users::OpOperands(users) => users.len(),
        }
    }
}

pub struct Variadic;

impl Display for Variadic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "...")
    }
}

/// An instance of a value (SSA, variadic, or constant) in the IR.
///
/// The primary purpose of a [Value] is to be pointed to by operands. This means
/// that a [Value] is typically an [OpResult]. Next, [OpOperand]s point to this
/// [Value]. So, in the following example:
///
/// ```mlir
/// %x = arith.constant 1 : i64
/// %y = arith.constant 2 : i64
/// %z = arith.addi %x, %y
/// ```
///
/// The [OpOperand] `%x` in the last line points to the [OpResult] defined by
/// the [Operation] in the first line. There are multiple reasons for pointing
/// directly to the place where the SSA variable is created, one of which is to
/// verify during parsing that the SSA variable is created before it is used.
///
/// We also express a [Constant] as a [Value] because it allows us to keep track
/// of the order of the operands in the [Operation] `operands` field.
pub enum Value {
    BlockArgument(BlockArgument),
    BlockLabel(BlockLabel),
    Constant(Constant),
    FuncResult(AnonymousResult),
    OpResult(OpResult),
    Variadic,
}

impl Value {
    /// The name of the value.
    ///
    /// Returns `None` for block arguments or func results that do not have a
    /// name.
    ///
    /// During printing, a new name is generated automatically. This is because
    /// some names might collide during rewriting (for example, when moving a
    /// set of variables out of a region/scope). These collides are no problem
    /// because even though the name is the same, they are different [Value]s.
    /// At the same time, LLVM wants SSA values to have unique and monotonically
    /// increasing names. We can solve both these problems by just generating
    /// new names in the end, that is, during printing.
    pub fn name(&self) -> Option<String> {
        match self {
            Value::BlockArgument(arg) => match arg.name() {
                Some(BlockArgumentName::Name(name)) => Some(name.clone()),
                Some(BlockArgumentName::Anonymous) => None,
                None => None,
            },
            Value::BlockLabel(label) => Some(label.name.clone()),
            Value::Constant(_) => None,
            Value::FuncResult(_) => None,
            Value::OpResult(result) => result.name.clone(),
            Value::Variadic => None,
        }
    }
    pub fn typ(&self) -> Result<Arc<RwLock<dyn Type>>> {
        match self {
            Value::BlockArgument(arg) => Ok(arg.typ.clone()),
            Value::BlockLabel(_) => todo!(),
            Value::Constant(constant) => Ok(constant.typ()),
            Value::FuncResult(result) => Ok(result.typ.clone()),
            Value::OpResult(result) => match result.typ() {
                Some(typ) => Ok(typ),
                None => Err(anyhow::anyhow!("Type was not set for OpResult {}", self)),
            },
            Value::Variadic => todo!(),
        }
    }
    pub fn set_type(&mut self, typ: Arc<RwLock<dyn Type>>) {
        match self {
            Value::BlockArgument(arg) => arg.set_typ(typ),
            Value::BlockLabel(_) => todo!(),
            Value::Constant(_) => todo!(),
            Value::FuncResult(result) => result.set_typ(typ),
            Value::OpResult(result) => result.set_typ(typ),
            Value::Variadic => todo!(),
        }
    }
    pub fn set_defining_op(&mut self, op: Option<Arc<RwLock<dyn Op>>>) {
        match self {
            Value::BlockArgument(_) => panic!("Cannot set defining op for BlockArgument"),
            Value::BlockLabel(_) => todo!("Cannot set defining op for BlockLabel"),
            Value::Constant(_) => panic!("Cannot set defining op for Constant"),
            Value::FuncResult(_) => panic!("It is not necessary to set this defining op"),
            Value::OpResult(op_res) => op_res.set_defining_op(op),
            Value::Variadic => panic!("Cannot set defining op for Variadic"),
        }
    }
    pub fn set_parent(&mut self, parent: Option<Arc<RwLock<Block>>>) {
        match self {
            Value::BlockArgument(arg) => arg.set_parent(parent),
            Value::BlockLabel(_) => todo!(),
            Value::Constant(_) => todo!(),
            Value::FuncResult(_) => todo!(),
            Value::OpResult(_) => todo!(),
            Value::Variadic => todo!(),
        }
    }
    pub fn set_name(&mut self, name: &str) {
        match self {
            Value::BlockArgument(arg) => {
                let arg_name = BlockArgumentName::Name(name.to_string());
                arg.set_name(Some(arg_name));
            }
            Value::BlockLabel(label) => label.set_name(name.to_string()),
            Value::Constant(_) => panic!("Cannot set name for Constant"),
            Value::FuncResult(_) => panic!("It is not necessary to set this name"),
            Value::OpResult(result) => result.set_name(name),
            Value::Variadic => panic!("Cannot set name for Variadic"),
        }
    }
    fn find_users(&self, ops: &Vec<Arc<RwLock<dyn Op>>>) -> Vec<Arc<RwLock<OpOperand>>> {
        let mut out = Vec::new();
        for op in ops.iter() {
            let operation = op.operation();
            let operation = operation.try_read().unwrap();
            let operands = operation.operands().vec();
            let operands = operands.try_read().unwrap();
            for operand in operands.iter() {
                let operand_clone = operand.clone();
                let operand_clone = operand_clone.try_read().unwrap();
                let value = operand_clone.value();
                let value = value.try_read().unwrap();
                if std::ptr::eq(&*value as *const Value, self as *const Value) {
                    out.push(operand.clone());
                }
            }
        }
        out
    }
    fn block_arg_users(&self, arg: &BlockArgument) -> Vec<Arc<RwLock<OpOperand>>> {
        let parent = arg.parent();
        let parent = if parent.is_some() {
            parent.unwrap()
        } else {
            panic!("BlockArgument {arg} has no parent operation");
        };
        let successors = parent.successors().unwrap();
        let parent = parent.try_read().unwrap();
        let ops = parent.ops();
        let mut ops = ops.try_read().unwrap().clone();
        for successor in successors.iter() {
            let current_ops = successor.ops();
            let current_ops = current_ops.try_read().unwrap().clone();
            ops.extend(current_ops);
        }
        self.find_users(&ops)
    }
    fn op_result_users(&self, op_res: &OpResult) -> Vec<Arc<RwLock<OpOperand>>> {
        let op = op_res.defining_op();
        let op = if op.is_some() {
            op.unwrap()
        } else {
            panic!("Defining op not set for OpResult {op_res}");
        };
        let ops = op.operation().successors();
        self.find_users(&ops)
    }
    pub fn users(&self) -> Users {
        match self {
            Value::BlockArgument(arg) => Users::OpOperands(self.block_arg_users(arg)),
            Value::BlockLabel(_) => todo!(),
            Value::Constant(_) => todo!("so this is empty? not sure yet"),
            Value::FuncResult(_) => todo!(),
            Value::OpResult(op_res) => Users::OpOperands(self.op_result_users(op_res)),
            Value::Variadic => Users::HasNoOpResults,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::BlockArgument(arg) => write!(f, "{arg}"),
            Value::BlockLabel(label) => write!(f, "{label}"),
            Value::Constant(constant) => write!(f, "{constant}"),
            Value::FuncResult(result) => write!(f, "{result}"),
            Value::OpResult(result) => write!(f, "{result}"),
            Value::Variadic => write!(f, "..."),
        }
    }
}

pub trait GuardedValue {
    fn rename(&self, new_name: &str);
    fn set_parent(&self, parent: Option<Arc<RwLock<Block>>>);
    fn typ(&self) -> Result<Arc<RwLock<dyn Type>>>;
}

impl GuardedValue for Arc<RwLock<Value>> {
    fn rename(&self, new_name: &str) {
        let mut value = self.try_write().unwrap();
        value.set_name(new_name);
    }
    fn set_parent(&self, parent: Option<Arc<RwLock<Block>>>) {
        let mut value = self.try_write().unwrap();
        value.set_parent(parent);
    }
    fn typ(&self) -> Result<Arc<RwLock<dyn Type>>> {
        let value = self.try_read().unwrap();
        value.typ()
    }
}

/// Vector of values.
///
/// Used to store operation results and function arguments. This naming is
/// identical to MLIR values. Also there, a value can be a block argument or an
/// operation result.
#[derive(Clone)]
pub struct Values {
    values: Arc<RwLock<Vec<Arc<RwLock<Value>>>>>,
}

impl Values {
    pub fn from_vec(values: Vec<Arc<RwLock<Value>>>) -> Self {
        Values {
            values: Arc::new(RwLock::new(values)),
        }
    }
    pub fn vec(&self) -> Arc<RwLock<Vec<Arc<RwLock<Value>>>>> {
        self.values.clone()
    }
    pub fn names(&self) -> Vec<String> {
        self.values
            .try_read()
            .unwrap()
            .iter()
            .map(|value| value.try_read().unwrap().name().unwrap())
            .collect()
    }
    /// The number of values.
    ///
    /// At some point, this preferably is replaced by some collection trait.
    /// But this seems to not be available yet?
    pub fn len(&self) -> usize {
        self.values.try_read().unwrap().len()
    }
    pub fn rename_variables(&self, renamer: &dyn VariableRenamer) -> Result<()> {
        let values = self.values.try_read().unwrap();
        for value in values.iter() {
            let mut value = value.try_write().unwrap();
            let name = value.name().unwrap();
            value.set_name(&renamer.rename(&name));
        }
        Ok(())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn types(&self) -> Types {
        let values = self.values.try_read().unwrap();
        let types = values
            .iter()
            .map(|value| value.typ().unwrap())
            .collect::<Vec<Arc<RwLock<dyn Type>>>>();
        Types::from_vec(types)
    }
    pub fn update_types(&mut self, types: Vec<Arc<RwLock<dyn Type>>>) -> Result<()> {
        let values = self.values.try_read().unwrap();
        if values.len() != types.len() {
            return Err(anyhow::anyhow!(
                "Expected {} types, but got {}",
                values.len(),
                types.len()
            ));
        }
        for (i, value) in values.iter().enumerate() {
            let mut container = value.try_write().unwrap();
            container.set_type(types[i].clone());
        }
        Ok(())
    }
    /// Set the defining op for op results.
    ///
    /// This can be used when replacing an operation with another operation to
    /// set the defining op for the results to point to the new operation.
    ///
    /// Calling this on block arguments (like function arguments) will panic since
    /// block arguments do not specify a defining op.
    pub fn set_defining_op(&self, op: Arc<RwLock<dyn Op>>) {
        let results = self.values.read().unwrap();
        for result in results.iter() {
            let mut mut_result = result.try_write().unwrap();
            match &mut *mut_result {
                Value::BlockArgument(_) => {
                    panic!("Trying to set defining op for block argument")
                }
                Value::BlockLabel(_) => {
                    panic!("Trying to set defining op for block label")
                }
                Value::Constant(_) => {
                    panic!("Trying to set defining op for constant")
                }
                Value::FuncResult(_) => {
                    panic!("Trying to set defining op for func result")
                }
                Value::OpResult(res) => res.set_defining_op(Some(op.clone())),
                Value::Variadic => panic!("Trying to set defining op for variadic"),
            }
        }
    }
    /// Convert all types in-place via the given type converter.
    ///
    /// This is commonly used to convert types from one dialect to another.
    /// Note that it should only be called to convert `operation.result_types()`.
    /// To modify `operation.operand_types()`, call this method on the
    /// `operation`s that define the operands.
    pub fn convert_types<T: TypeConvert>(&self) -> Result<()> {
        let values = self.values.try_read().unwrap();
        for value in values.iter() {
            let mut value = value.try_write().unwrap();
            let typ = value.typ().unwrap();
            let typ = T::convert_type(&typ)?;
            value.set_type(typ);
        }
        Ok(())
    }
}

impl Default for Values {
    fn default() -> Self {
        Values {
            values: Arc::new(RwLock::new(vec![])),
        }
    }
}

impl Display for Values {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let joined = self
            .values
            .try_read()
            .unwrap()
            .iter()
            .map(|o| o.try_read().unwrap().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{joined}")
    }
}

/// Call to a destination of type block.
///
/// For example, `^merge(%c4: i32)` in:
///
/// ```mlir
/// %c4 = arith.constant 4 : i32
/// cr.br ^merge(%c4: i32)
/// ```
///
/// or `^then, ^else` in:
///
/// ```mlir
/// cr.cond_br %cond, ^then, ^else
/// ```
///
/// This data structure is used by ops such as `cf.cond_br` to keep track of
/// multiple destinations.
///
/// To allow other parts of the codebase to still recognize uses of some
/// [OpResult] (like `%c4` in the first example), block destinations do not
/// contain the [OpOperand]s. The operands are instead stored as operands of the
/// operation. This is possible because the block destinations in `cr.cond_br` do
/// not take arguments (let's hope this assumption keeps standing over time or
/// we need to rewrite this).
///
/// Unlike variables ([OpResult]s), block destinations do not contain a pointer
/// to the block. The reason is that the block definition may appear after the
/// block destination. Put differently, whereas functions and variables have to
/// be defined before calling them, blocks don't have to. This means that in
/// order to parse a block destination, we would need two passes. This is
/// currently not implemented. The solution would probably to add an
/// `Option<Arc<RwLock<Block>>>` to this struct and set it later.
pub struct BlockDest {
    name: String,
}

impl BlockDest {
    pub fn new(name: &str) -> Self {
        BlockDest {
            name: name.to_string(),
        }
    }
    /// The name of the destination block (e.g., `^merge`).
    pub fn name(&self) -> String {
        self.name.clone()
    }
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
}

impl Display for BlockDest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

// Putting these on the parser to allow method discovery via `parser.parse_`.
impl<T: ParserDispatch> Parser<T> {
    /// Parse `%arg0 : i64,`, `i64,`, or `...`.
    pub fn parse_function_argument(&mut self) -> Result<Arc<RwLock<Value>>> {
        if self.check(TokenKind::PercentIdentifier) {
            let identifier = self.expect(TokenKind::PercentIdentifier)?;
            let name = identifier.lexeme.clone();
            let _colon = self.expect(TokenKind::Colon)?;
            let typ = T::parse_type(self)?;
            let name = BlockArgumentName::Name(name);
            let arg = Value::BlockArgument(BlockArgument::new(Some(name), typ));
            let operand = Arc::new(RwLock::new(arg));
            if self.check(TokenKind::Comma) {
                self.advance();
            }
            return Ok(operand);
        }
        if self.check(TokenKind::IntType) || self.check(TokenKind::Exclamation) {
            let typ = T::parse_type(self)?;
            let name = None;
            let arg = Value::BlockArgument(BlockArgument::new(name, typ));
            let operand = Arc::new(RwLock::new(arg));
            if self.check(TokenKind::Comma) {
                self.advance();
            }
            return Ok(operand);
        }
        if self.check(TokenKind::Dot) {
            self.expect(TokenKind::Dot)?;
            self.expect(TokenKind::Dot)?;
            self.expect(TokenKind::Dot)?;
            let variadic = Value::Variadic;
            return Ok(Arc::new(RwLock::new(variadic)));
        }
        Err(anyhow::anyhow!("Expected function argument"))
    }
    pub fn is_function_argument(&mut self) -> bool {
        // For example, `@add(%arg0 : i32)`.
        let perc = self.check(TokenKind::PercentIdentifier);
        // For example, `@add(i32)`.
        let int = self.check(TokenKind::IntType);
        // For example, `@printf(!llvm.ptr)`.
        let excl = self.check(TokenKind::Exclamation);
        // For example, `@printf(ptr, ...)`.
        let dot = self.check(TokenKind::Dot);
        perc || int || excl || dot
    }
    /// Parse operation result.
    pub fn parse_op_result(&mut self, token_kind: TokenKind) -> Result<UnsetOpResult> {
        let identifier = self.expect(token_kind)?;
        let name = identifier.lexeme.clone();
        let mut op_result = OpResult::default();
        op_result.set_name(&name);
        let result = Value::OpResult(op_result);
        Ok(UnsetOpResult::new(Arc::new(RwLock::new(result))))
    }
    pub fn parse_op_result_into(
        &mut self,
        token_kind: TokenKind,
        operation: &mut Operation,
    ) -> Result<UnsetOpResult> {
        let result = self.parse_op_result(token_kind)?.result;
        let results = Values::from_vec(vec![result.clone()]);
        operation.set_results(results.clone());
        Ok(UnsetOpResult::new(result))
    }
    /// Parse operation results.
    ///
    /// Can parse `%0` in `%0 = ...` when `token_kind` is
    /// `TokenKind::PercentIdentifier`, or `x` in `x = ...` when `token_kind` is
    /// `TokenKind::BareIdentifier`.
    pub fn parse_op_results(&mut self, token_kind: TokenKind) -> Result<UnsetOpResults> {
        let mut results = vec![];
        while self.check(token_kind) {
            let result = self.parse_op_result(token_kind)?.result;
            results.push(result);
            if self.check(TokenKind::Comma) {
                let _comma = self.advance();
            }
        }
        let results = Arc::new(RwLock::new(results));
        let values = Values { values: results };
        Ok(UnsetOpResults::new(values))
    }
    /// Parse results (e.g., `%0 = ...`) into an operation.
    ///
    /// This returns the results to allow setting the defining op on them.
    ///
    /// Setting the type is not necessary since the type
    pub fn parse_op_results_into(
        &mut self,
        token_kind: TokenKind,
        operation: &mut Operation,
    ) -> Result<UnsetOpResults> {
        let results = self.parse_op_results(token_kind)?;
        let values = results.values();
        operation.set_results(values.clone());
        Ok(results)
    }
    /// Parse `(%arg0 : i64, %arg1 : i64)`, or `(i64, !llvm.ptr)`.
    pub fn parse_function_arguments(&mut self) -> Result<Values> {
        let _lparen = self.expect(TokenKind::LParen)?;
        let mut operands = vec![];
        while self.is_function_argument() {
            operands.push(self.parse_function_argument()?);
        }
        let _rparen = self.expect(TokenKind::RParen);
        let values = Values::from_vec(operands);
        Ok(values)
    }
    /// Parse a destination of type block (e.g., `^merge(%c4: i32)`).
    ///
    /// Example:
    /// ```mlir
    /// cr.br ^merge(%c4: i32)
    /// ```
    /// or
    /// ```mlir
    /// cr.br ^exit
    /// ```
    pub fn parse_block_dest(&mut self) -> Result<BlockDest> {
        let name = self.expect(TokenKind::CaretIdentifier)?;
        let name = name.lexeme.clone();
        Ok(BlockDest { name })
    }
}
