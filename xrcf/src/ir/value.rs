use crate::frontend::Parser;
use crate::frontend::ParserDispatch;
use crate::frontend::TokenKind;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::BlockName;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::Type;
use crate::ir::TypeConvert;
use crate::ir::Types;
use crate::ir::VariableRenamer;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Display;
use std::sync::Arc;

pub enum BlockArgumentName {
    /// Anonymous block arguments are used for functions without an implementation.
    Anonymous,
    /// The name of the block argument.
    Name(String),
    /// The name does not have to be set since we generate unique names anyway.
    Unset,
}

/// An argument in a block or function.
pub struct BlockArgument {
    /// The name of the block argument.
    ///
    /// The name is only used during parsing to see which operands point to this
    /// argument. During printing, a new name is generated.
    name: Shared<BlockArgumentName>,
    typ: Shared<dyn Type>,
    /// The operation for which this [BlockArgument] is an argument.
    ///
    /// This is used by [value.users] to find the users of this
    /// [BlockArgument].
    parent: Option<Shared<Block>>,
}

impl BlockArgument {
    pub fn new(name: Shared<BlockArgumentName>, typ: Shared<dyn Type>) -> Self {
        BlockArgument {
            name,
            typ,
            parent: None,
        }
    }
    pub fn name(&self) -> Shared<BlockArgumentName> {
        self.name.clone()
    }
    pub fn parent(&self) -> Option<Shared<Block>> {
        self.parent.clone()
    }
    pub fn set_name(&self, name: BlockArgumentName) {
        *self.name.wr() = name;
    }
    pub fn set_parent(&mut self, parent: Option<Shared<Block>>) {
        self.parent = parent;
    }
    pub fn set_typ(&mut self, typ: Shared<dyn Type>) {
        self.typ = typ;
    }
    pub fn typ(&self) -> Shared<dyn Type> {
        self.typ.clone()
    }
}

impl Display for BlockArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let typ = self.typ.rd();
        let name = self.name();
        let name_read = name.rd();
        match &*name_read {
            BlockArgumentName::Anonymous => write!(f, "{typ}"),
            BlockArgumentName::Name(name) => {
                write!(f, "{name} : {typ}")
            }
            BlockArgumentName::Unset => {
                // Name should normally be set during region printing.
                write!(f, "<UNSET NAME> : {typ}")
            }
        }
    }
}

/// A label to a block.
///
/// This is a temporary data structure that holds a label until the block is
/// parsed. At that point, the [BlockLabel]s are replaced by [BlockPtr]s.
///
/// Note that because this is a [Value], it can be directly stored in
/// `operation.operands`. To model labels with operands like
/// ```mlir
/// cr.br ^merge(%c4: i32)
/// ```
/// store the `^merge` as a [BlockLabel] and the `%c4` as an [OpResult]. This
/// "encoding" in the operands field has the benefit that other code can easily
/// check if, for example, a variable is being used.
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

/// A pointer to a block (replaces the block label).
///
/// This is essentially a sort of interface so that [OpOperand]'s
/// pointer (`Arc<RwLock<Value>>`) can point to a block.
///
/// To model labels with operands like
/// ```mlir
/// cr.br ^merge(%c4: i32)
/// ```
/// store the `^merge` as a [BlockPtr] and the `%c4` as an [OpResult]. This
/// "encoding" in the operands field has the benefit that other code can easily
/// check if, for example, a variable is being used.
pub struct BlockPtr {
    block: Shared<Block>,
}

impl BlockPtr {
    pub fn new(block: Shared<Block>) -> Self {
        BlockPtr { block }
    }
    pub fn block(&self) -> Shared<Block> {
        self.block.clone()
    }
}

impl Display for BlockPtr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // By default, block pointers are prefixed with `^`. Ops which want
        // different printing can do that themselves (it is known which operands
        // are pointers so this is hopefully not too much of a problem).
        write!(f, "^{}", self.block.rd())
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
    pub fn typ(&self) -> Shared<dyn Type> {
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
    typ: Shared<dyn Type>,
}

impl AnonymousResult {
    pub fn new(typ: Shared<dyn Type>) -> Self {
        AnonymousResult { typ }
    }
    pub fn typ(&self) -> Shared<dyn Type> {
        self.typ.clone()
    }
    pub fn set_typ(&mut self, typ: Shared<dyn Type>) {
        self.typ = typ;
    }
}

impl Display for AnonymousResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.typ.rd())
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
    /// The name of the result.
    ///
    /// Does not necessarily have to be set because new names are generated
    /// anyway.
    name: Shared<Option<String>>,
    typ: Option<Shared<dyn Type>>,
    defining_op: Option<Shared<dyn Op>>,
}

impl OpResult {
    pub fn new(
        name: Shared<Option<String>>,
        typ: Option<Shared<dyn Type>>,
        defining_op: Option<Shared<dyn Op>>,
    ) -> Self {
        OpResult {
            name,
            typ,
            defining_op,
        }
    }
    pub fn name(&self) -> Shared<Option<String>> {
        self.name.clone()
    }
    pub fn typ(&self) -> Option<Shared<dyn Type>> {
        self.typ.clone()
    }
    pub fn defining_op(&self) -> Option<Shared<dyn Op>> {
        self.defining_op.clone()
    }
    pub fn set_name(&self, name: &str) {
        *self.name.wr() = Some(name.to_string());
    }
    pub fn set_typ(&mut self, typ: Shared<dyn Type>) {
        self.typ = Some(typ);
    }
    pub fn set_defining_op(&mut self, op: Option<Shared<dyn Op>>) {
        self.defining_op = op;
    }
}

impl Default for OpResult {
    fn default() -> Self {
        Self {
            name: Shared::new(None.into()),
            typ: None,
            defining_op: None,
        }
    }
}

impl Display for OpResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name.rd().clone().unwrap())
    }
}

#[must_use = "the object should be further initialized, see the setter methods"]
pub struct UnsetOpResult {
    result: Shared<Value>,
}

impl UnsetOpResult {
    pub fn new(result: Shared<Value>) -> Self {
        assert!(
            matches!(&*result.rd(), Value::OpResult(_)),
            "Expected OpResult"
        );
        UnsetOpResult { result }
    }
    pub fn set_defining_op(&self, op: Option<Shared<dyn Op>>) {
        self.result.wr().set_defining_op(op);
    }
    pub fn set_typ(&self, typ: Shared<dyn Type>) {
        self.result.wr().set_type(typ);
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
    pub fn set_defining_op(&self, op: Shared<dyn Op>) {
        let values = self.values();
        values.set_defining_op(op);
    }
    pub fn set_types(&self, types: Vec<Shared<dyn Type>>) {
        let results = self.values().into_iter();
        assert!(types.len() == results.len());
        for (result, typ) in results.zip(types) {
            result.wr().set_type(typ);
        }
    }
}

pub enum Users {
    /// The operation defines no `OpResult`s.
    HasNoOpResults,
    /// The operation defines `OpResult`s (and can still have zero users).
    OpOperands(Vec<Shared<OpOperand>>),
}

impl Users {
    pub fn len(&self) -> usize {
        match self {
            Users::HasNoOpResults => 0,
            Users::OpOperands(users) => users.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    /// A block argument (e.g., `add_one(%arg0: i64)`).
    BlockArgument(BlockArgument),
    /// A block label (will be replaced by a pointer once the block is parsed).
    BlockLabel(BlockLabel),
    /// A pointer to a block (replaces the block label).
    ///
    /// This is essentially a sort of interface so that [OpOperand]'s
    /// pointer (`Arc<RwLock<Value>>`) can point to a block.
    BlockPtr(BlockPtr),
    /// A constant value (e.g., `arith.constant 1 : i64`).
    Constant(Constant),
    FuncResult(AnonymousResult),
    /// A result of an operation (e.g., `%0 = ...`).
    OpResult(OpResult),
    /// A variadic value.
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
            Value::BlockArgument(arg) => match &*arg.name().rd() {
                BlockArgumentName::Anonymous => None,
                BlockArgumentName::Name(name) => Some(name.clone()),
                BlockArgumentName::Unset => None,
            },
            Value::BlockLabel(label) => Some(label.name.clone()),
            Value::BlockPtr(ptr) => match &*ptr.block().rd().label().rd() {
                BlockName::Name(name) => Some(name.clone()),
                BlockName::Unnamed => None,
                BlockName::Unset => None,
            },
            Value::Constant(_) => None,
            Value::FuncResult(_) => None,
            Value::OpResult(result) => result.name().rd().clone(),
            Value::Variadic => None,
        }
    }
    pub fn typ(&self) -> Result<Shared<dyn Type>> {
        match self {
            Value::BlockArgument(arg) => Ok(arg.typ.clone()),
            Value::BlockLabel(_) => panic!("BlockLabel has no type"),
            Value::BlockPtr(_) => panic!("BlockPtr has no type"),
            Value::Constant(constant) => Ok(constant.typ()),
            Value::FuncResult(result) => Ok(result.typ.clone()),
            Value::OpResult(result) => match result.typ() {
                Some(typ) => Ok(typ),
                None => Err(anyhow::anyhow!("Type was not set for OpResult {}", self)),
            },
            Value::Variadic => panic!("Variadic has no type"),
        }
    }
    pub fn set_type(&mut self, typ: Shared<dyn Type>) {
        match self {
            Value::BlockArgument(arg) => arg.set_typ(typ),
            Value::BlockLabel(_) => todo!(),
            Value::BlockPtr(_) => todo!(),
            Value::Constant(_) => todo!(),
            Value::FuncResult(result) => result.set_typ(typ),
            Value::OpResult(result) => result.set_typ(typ),
            Value::Variadic => todo!(),
        }
    }
    pub fn set_defining_op(&mut self, op: Option<Shared<dyn Op>>) {
        match self {
            Value::BlockArgument(_) => panic!("Cannot set defining op for BlockArgument"),
            Value::BlockLabel(_) => todo!("Cannot set defining op for BlockLabel"),
            Value::BlockPtr(_) => todo!("Cannot set defining op for BlockPtr"),
            Value::Constant(_) => panic!("Cannot set defining op for Constant"),
            Value::FuncResult(_) => panic!("It is not necessary to set this defining op"),
            Value::OpResult(op_res) => op_res.set_defining_op(op),
            Value::Variadic => panic!("Cannot set defining op for Variadic"),
        }
    }
    pub fn set_parent(&mut self, parent: Option<Shared<Block>>) {
        match self {
            Value::BlockArgument(arg) => arg.set_parent(parent),
            Value::BlockLabel(_) => todo!(),
            Value::BlockPtr(_) => todo!(),
            Value::Constant(_) => todo!(),
            Value::FuncResult(_) => todo!(),
            Value::OpResult(_) => todo!(),
            Value::Variadic => todo!(),
        }
    }
    pub fn set_name(&mut self, name: &str) {
        match self {
            Value::BlockArgument(arg) => {
                arg.set_name(BlockArgumentName::Name(name.to_string()));
            }
            Value::BlockLabel(label) => label.set_name(name.to_string()),
            Value::BlockPtr(_) => todo!(),
            Value::Constant(_) => panic!("Cannot set name for Constant"),
            Value::FuncResult(_) => panic!("It is not necessary to set this name"),
            Value::OpResult(result) => result.set_name(name),
            Value::Variadic => panic!("Cannot set name for Variadic"),
        }
    }
    pub fn from_block(block: Shared<Block>) -> Self {
        let ptr = BlockPtr::new(block);
        Value::BlockPtr(ptr)
    }
    fn find_users(&self, ops: &[Shared<dyn Op>]) -> Vec<Shared<OpOperand>> {
        let mut out = Vec::new();
        for op in ops.iter() {
            for operand in op.rd().operation().rd().operands().into_iter() {
                let value = operand.rd().value();
                if std::ptr::eq(&*value.rd() as *const Value, self as *const Value) {
                    out.push(operand.clone());
                }
            }
        }
        out
    }
    fn block_arg_users(&self, arg: &BlockArgument) -> Vec<Shared<OpOperand>> {
        let parent = arg.parent();
        let parent = if let Some(parent) = parent {
            parent
        } else {
            panic!("BlockArgument {arg} has no parent operation");
        };
        let mut ops = parent.rd().ops().rd().clone();
        for successor in parent.rd().successors().unwrap().iter() {
            ops.extend(successor.rd().ops().rd().clone());
        }
        self.find_users(&ops)
    }
    fn op_result_users(&self, op_res: &OpResult) -> Vec<Shared<OpOperand>> {
        let op = op_res.defining_op();
        let op = if op_res.defining_op().is_some() {
            op.unwrap()
        } else {
            panic!("Defining op not set for OpResult {op_res}");
        };
        let ops = op.rd().operation().rd().successors();
        self.find_users(&ops)
    }
    pub fn users(&self) -> Users {
        match self {
            Value::BlockArgument(arg) => Users::OpOperands(self.block_arg_users(arg)),
            Value::BlockLabel(_) => todo!(),
            Value::BlockPtr(_) => todo!(),
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
            Value::BlockPtr(ptr) => write!(f, "{ptr}"),
            Value::Constant(constant) => write!(f, "{constant}"),
            Value::FuncResult(result) => write!(f, "{result}"),
            Value::OpResult(result) => write!(f, "{result}"),
            Value::Variadic => write!(f, "..."),
        }
    }
}

/// Vector of values.
///
/// Used to store operation results and function arguments. This naming is
/// identical to MLIR values. Also there, a value can be a block argument or an
/// operation result.
#[derive(Clone)]
pub struct Values {
    values: Shared<Vec<Shared<Value>>>,
}

impl IntoIterator for Values {
    type Item = Shared<Value>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.rd().clone().into_iter()
    }
}

impl Values {
    pub fn from_vec(values: Vec<Shared<Value>>) -> Self {
        Values {
            values: Shared::new(values.into()),
        }
    }
    pub fn vec(&self) -> Shared<Vec<Shared<Value>>> {
        self.values.clone()
    }
    pub fn names(&self) -> Vec<String> {
        self.values
            .try_read()
            .unwrap()
            .iter()
            .map(|value| value.rd().name().unwrap())
            .collect()
    }
    /// The number of values.
    ///
    /// At some point, this preferably is replaced by some collection trait.
    /// But this seems to not be available yet?
    pub fn len(&self) -> usize {
        self.values.rd().len()
    }
    pub fn rename_variables(&self, renamer: &dyn VariableRenamer) -> Result<()> {
        let values = self.values.rd();
        for value in values.iter() {
            let name = value.rd().name().unwrap();
            value.wr().set_name(&renamer.rename(&name));
        }
        Ok(())
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn types(&self) -> Types {
        let types = self
            .values
            .rd()
            .iter()
            .map(|value| value.rd().typ().unwrap())
            .collect::<Vec<Shared<dyn Type>>>();
        Types::from_vec(types)
    }
    pub fn update_types(&mut self, types: Vec<Shared<dyn Type>>) -> Result<()> {
        let values = self.values.rd().clone().into_iter();
        if values.len() != types.len() {
            return Err(anyhow::anyhow!(
                "Expected {} types, but got {}",
                values.len(),
                types.len()
            ));
        }
        for (i, value) in values.enumerate() {
            value.wr().set_type(types[i].clone());
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
    pub fn set_defining_op(&self, op: Shared<dyn Op>) {
        for result in self.values.rd().iter() {
            match &mut *result.wr() {
                Value::BlockArgument(_) => {
                    panic!("Trying to set defining op for block argument")
                }
                Value::BlockLabel(_) => {
                    panic!("Trying to set defining op for block label")
                }
                Value::BlockPtr(_) => panic!("Trying to set defining op for block ptr"),
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
        for value in self.values.rd().iter() {
            let typ = value.rd().typ().unwrap();
            value.wr().set_type(T::convert_type(&typ)?);
        }
        Ok(())
    }
}

impl Default for Values {
    fn default() -> Self {
        Values {
            values: Shared::new(vec![].into()),
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
            .map(|o| o.rd().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, "{joined}")
    }
}

// Putting these on the parser to allow method discovery via `parser.parse_`.
impl<T: ParserDispatch> Parser<T> {
    /// Parse `%arg0 : i64,`, `i64,`, or `...`.
    pub fn parse_function_argument(&mut self) -> Result<Shared<Value>> {
        if self.check(TokenKind::PercentIdentifier) {
            let identifier = self.expect(TokenKind::PercentIdentifier)?;
            let name = identifier.lexeme.clone();
            let _colon = self.expect(TokenKind::Colon)?;
            let typ = T::parse_type(self)?;
            let name = BlockArgumentName::Name(name);
            let name = Shared::new(name.into());
            let arg = Value::BlockArgument(BlockArgument::new(name, typ));
            let operand = Shared::new(arg.into());
            if self.check(TokenKind::Comma) {
                self.advance();
            }
            return Ok(operand);
        }
        if self.check(TokenKind::IntType) || self.check(TokenKind::Exclamation) {
            let typ = T::parse_type(self)?;
            let name = BlockArgumentName::Anonymous;
            let name = Shared::new(name.into());
            let arg = Value::BlockArgument(BlockArgument::new(name, typ));
            let operand = Shared::new(arg.into());
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
            return Ok(Shared::new(variadic.into()));
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
        let op_result = OpResult::default();
        op_result.set_name(&name);
        let result = Value::OpResult(op_result);
        Ok(UnsetOpResult::new(Shared::new(result.into())))
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
        let results = Shared::new(results.into());
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
        operation.set_results(results.values().clone());
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
    pub fn parse_block_dest(&mut self) -> Result<OpOperand> {
        let name = self.expect(TokenKind::CaretIdentifier)?;
        let name = name.lexeme.clone();
        let value = Value::BlockLabel(BlockLabel::new(name));
        let value = Shared::new(value.into());
        Ok(OpOperand::new(value))
    }
}
