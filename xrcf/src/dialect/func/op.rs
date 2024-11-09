use crate::ir::AnyType;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOp;
use crate::ir::GuardedOperation;
use crate::ir::GuardedRegion;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::OpWithoutParent;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Region;
use crate::ir::StringAttr;
use crate::ir::Type;
use crate::ir::Value;
use crate::ir::Values;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

/// `func.call`
///
/// ```ebnf
/// `func.call` $callee `(` $operands `)` attr-dict `:` `(` type($operands) `)` -> type($results)
/// ```
pub struct CallOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
    varargs: Option<Arc<RwLock<dyn Type>>>,
}

pub trait Call: Op {
    fn identifier(&self) -> Option<String>;
    fn set_identifier(&mut self, identifier: String);
    fn varargs(&self) -> Option<Arc<RwLock<dyn Type>>>;
    fn set_varargs(&mut self, varargs: Option<Arc<RwLock<dyn Type>>>);
    fn display_call_op(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let operation = self.operation().read().unwrap();
        let results = operation.results();
        let has_results = !results.vec().try_read().unwrap().is_empty();
        if has_results {
            write!(f, "{} = ", operation.results())?;
        }
        write!(f, "{}", operation.name())?;
        write!(f, " {}", self.identifier().unwrap())?;
        write!(f, "({})", operation.operands())?;
        if let Some(varargs) = self.varargs() {
            let varargs = varargs.try_read().unwrap();
            write!(f, " vararg({})", varargs)?;
        }
        write!(f, " : ")?;
        write!(f, "({})", operation.operand_types())?;
        write!(f, " -> ")?;
        if has_results {
            let result_type = operation.result_type(0).expect("no result type");
            let result_type = result_type.try_read().unwrap();
            write!(f, "{}", result_type)?;
        } else {
            write!(f, "()")?;
        }
        Ok(())
    }
    /// Parse a call op such as `llvm.call`.
    ///
    /// Examples:
    ///
    /// ```mlir
    /// llvm.call @hello() : () -> ()
    ///
    /// llvm.call @printf(%0) : (!llvm.ptr) -> i32
    ///
    /// llvm.call @printf(%0, %1) vararg(!llvm.func<i32 (ptr, ...)>) :
    ///   (!llvm.ptr, i32) -> i32
    /// ```
    fn parse_call_op<T: ParserDispatch, O: Call + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
        allow_varargs: bool,
    ) -> Result<Arc<RwLock<O>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let results = parser.parse_op_results_into(&mut operation)?;
        parser.parse_operation_name_into::<O>(&mut operation)?;
        let identifier = parser.expect(TokenKind::AtIdentifier)?;
        let identifier = identifier.lexeme.clone();

        parser.expect(TokenKind::LParen)?;
        let operands = parser.parse_op_operands_into(parent.unwrap(), &mut operation)?;
        parser.expect(TokenKind::RParen)?;

        // vararg(!llvm.func<i32 (ptr, ...)>)
        let varargs = if allow_varargs && parser.peek().lexeme == "vararg" {
            let _vararg = parser.advance();
            parser.expect(TokenKind::LParen)?;
            let varargs = parser.parse_type()?;
            parser.expect(TokenKind::RParen)?;
            Some(varargs)
        } else {
            None
        };
        parser.expect(TokenKind::Colon)?;

        // (i32) or (!llvm.ptr, i32)
        parser.expect(TokenKind::LParen)?;
        if !operands.vec().try_read().unwrap().is_empty() {
            parser.parse_types_for_op_operands(operands)?;
        }
        parser.expect(TokenKind::RParen)?;

        parser.expect(TokenKind::Arrow)?;

        if parser.empty_type() {
            parser.parse_empty_type()?;
        } else {
            let result_type = T::parse_type(parser)?;
            operation.set_result_type(0, result_type)?;
        }

        let mut op = O::from_operation(operation);
        op.set_identifier(identifier);
        op.set_varargs(varargs);
        let op = Arc::new(RwLock::new(op));
        results.set_defining_op(op.clone());
        Ok(op)
    }
}

impl Call for CallOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
    fn varargs(&self) -> Option<Arc<RwLock<dyn Type>>> {
        self.varargs.clone()
    }
    fn set_varargs(&mut self, varargs: Option<Arc<RwLock<dyn Type>>>) {
        self.varargs = varargs;
    }
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("func.call".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        CallOp {
            operation,
            identifier: None,
            varargs: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        self.display_call_op(f)
    }
}

impl CallOp {
    pub fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
}

impl Parse for CallOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let allow_varargs = false;
        let call_op = CallOp::parse_call_op::<T, CallOp>(parser, parent, allow_varargs)?;
        Ok(call_op)
    }
}

pub trait Func: Op {
    fn identifier(&self) -> Option<String>;
    fn set_identifier(&mut self, identifier: String);
    fn set_argument_from_type(&mut self, index: usize, typ: Arc<RwLock<dyn Type>>) -> Result<()> {
        let argument = crate::ir::BlockArgument::new(None, typ);
        let value = Value::BlockArgument(argument);
        let value = Arc::new(RwLock::new(value));
        let operation = self.operation();
        operation.set_argument(index, value);
        Ok(())
    }
    fn sym_visibility(&self) -> Option<String> {
        let operation = self.operation();
        let attributes = operation.attributes();
        let attribute = attributes.get("sym_visibility");
        match attribute {
            Some(attribute) => Some(attribute.to_string()),
            // It is legal to not have set visibility.
            None => None,
        }
    }
    /// Set the symbol visibility.
    ///
    /// It is legal to not have set visibility.
    fn set_sym_visibility(&mut self, visibility: Option<String>) {
        if let Some(visibility) = visibility {
            let operation = self.operation();
            let operation = operation.try_write().unwrap();
            let attributes = operation.attributes();
            let attribute = StringAttr::from_str(&visibility);
            let attribute = Arc::new(attribute);
            attributes.insert("sym_visibility", attribute);
        }
    }
    fn arguments(&self) -> Result<Values> {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        let arguments = operation.arguments();
        Ok(arguments.clone())
    }
    fn return_types(&self) -> Vec<Arc<RwLock<dyn Type>>> {
        self.operation().results().types().vec()
    }
    fn return_type(&self) -> Result<Arc<RwLock<dyn Type>>> {
        let return_types = self.return_types();
        assert!(!return_types.is_empty(), "Expected result types to be set");
        assert!(return_types.len() == 1, "Expected single result type");
        Ok(return_types[0].clone())
    }
}

/// `func.func`
///
/// Note that the operands of the function are internally
/// represented by `BlockArgument`s, but the textual form is inline.
pub struct FuncOp {
    identifier: Option<String>,
    sym_visibility: Option<String>,
    operation: Arc<RwLock<Operation>>,
}

impl FuncOp {
    /// Insert `op` into the region of `self`, while creating a region if necessary.
    pub fn insert_op(&self, op: Arc<RwLock<dyn Op>>) -> OpWithoutParent {
        let read = op.try_read().unwrap();
        let ops = read.ops();
        if ops.is_empty() {
            let operation = self.operation();
            let region = operation.region();
            if region.is_some() {
                panic!("Expected region to be empty");
            }
            let ops = vec![op.clone()];
            let ops = Arc::new(RwLock::new(ops));
            let mut region = Region::default();
            let without_parent = region.add_new_block();
            let region = Arc::new(RwLock::new(region));
            let block = without_parent.set_parent(Some(region.clone()));
            block.set_ops(ops);
            operation.set_region(Some(region));
        } else {
            let last = ops.last().unwrap();
            last.insert_after(op.clone());
        }
        OpWithoutParent::new(op.clone())
    }
}

impl Func for FuncOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
    fn sym_visibility(&self) -> Option<String> {
        self.sym_visibility.clone()
    }
    fn set_sym_visibility(&mut self, visibility: Option<String>) {
        self.sym_visibility = visibility;
    }
}

impl FuncOp {
    fn display_visibility(op: &dyn Op) -> Option<String> {
        if let Some(func_op) = op.as_any().downcast_ref::<FuncOp>() {
            let visibility = func_op.sym_visibility.clone();
            return visibility;
        }
        None
    }
    pub fn display_func(
        op: &dyn Op,
        identifier: String,
        f: &mut Formatter<'_>,
        indent: i32,
    ) -> std::fmt::Result {
        let name = op.operation().try_read().unwrap().name();
        write!(f, "{name} ")?;
        if let Some(visibility) = FuncOp::display_visibility(op) {
            write!(f, "{visibility} ")?;
        }
        write!(f, "{identifier}(")?;
        let arguments = op.operation().arguments();
        write!(f, "{}", arguments)?;
        write!(f, ")")?;
        let operation = op.operation();
        let result_types = operation.results().types();
        if !result_types.vec().is_empty() {
            write!(f, " -> {}", result_types)?;
        }
        let attributes = operation.attributes();
        if !attributes.is_empty() {
            write!(f, " attributes {attributes}")?;
        }
        let region = op.operation().region();
        if let Some(region) = region {
            region.display(f, indent)?;
        }
        Ok(())
    }
    fn try_parse_func_visibility<T: ParserDispatch>(
        parser: &mut Parser<T>,
        expected_name: &OperationName,
    ) -> Option<String> {
        if expected_name == &FuncOp::operation_name() {
            if parser.check(TokenKind::BareIdentifier) {
                let token = parser.advance();
                let sym_visibility = token.lexeme.clone();
                return Some(sym_visibility);
            }
        }
        None
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("func.func".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        FuncOp {
            identifier: None,
            sym_visibility: None,
            operation,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_func(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn assignments(&self) -> Result<Values> {
        let operation = self.operation();
        let arguments = operation.arguments();
        Ok(arguments.clone())
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let identifier = self.identifier.clone();
        FuncOp::display_func(self, identifier.unwrap(), f, indent)
    }
}

impl<T: ParserDispatch> Parser<T> {
    fn result_types(&mut self) -> Result<Vec<Arc<RwLock<dyn Type>>>> {
        let mut result_types: Vec<Arc<RwLock<dyn Type>>> = vec![];
        if !self.check(TokenKind::Arrow) {
            return Ok(result_types);
        } else {
            let _arrow = self.advance();
            while self.check(TokenKind::IntType) {
                let typ = self.advance();
                let typ = AnyType::new(&typ.lexeme);
                let typ = Arc::new(RwLock::new(typ));
                result_types.push(typ);
            }
        }
        Ok(result_types)
    }
    pub fn parse_func<F: Func + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<F>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent);
        parser.parse_operation_name_into::<F>(&mut operation)?;
        let expected_name = F::operation_name();
        let visibility = FuncOp::try_parse_func_visibility(parser, &expected_name);
        let identifier = parser.expect(TokenKind::AtIdentifier)?;
        let identifier = identifier.lexeme.clone();
        operation.set_arguments(parser.parse_function_arguments()?);
        operation.set_anonymous_results(parser.result_types()?)?;
        let mut op = F::from_operation(operation);
        op.set_identifier(identifier);
        op.set_sym_visibility(visibility);
        let op = Arc::new(RwLock::new(op));
        let has_implementation = parser.check(TokenKind::LBrace);
        if has_implementation {
            let region = parser.region(op.clone())?;
            let op_rd = op.try_read().unwrap();
            op_rd.operation().set_region(Some(region.clone()));
            region.set_parent(Some(op.clone()));
        }

        Ok(op)
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let op = Parser::<T>::parse_func::<FuncOp>(parser, parent)?;
        Ok(op)
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
}

impl ReturnOp {
    pub fn display_return(
        op: &dyn Op,
        name: &str,
        f: &mut Formatter<'_>,
        _indent: i32,
    ) -> std::fmt::Result {
        let operation = op.operation();
        write!(f, "{name}")?;
        let operands = operation.operands().vec();
        let operands = operands.try_read().unwrap();
        if !operands.is_empty() {
            for operand in operands.iter() {
                write!(f, " {}", operand.read().unwrap())?;
            }
            let result_types = operation.results().types();
            write!(f, " : {}", result_types)?;
        }
        Ok(())
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("return".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        ReturnOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let name = Self::operation_name().to_string();
        ReturnOp::display_return(self, &name, f, _indent)
    }
}

impl<T: ParserDispatch> Parser<T> {
    pub fn return_op<O: Op>(
        &mut self,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<O>>> {
        let mut operation = Operation::default();
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        self.parse_operation_name_into::<O>(&mut operation)?;
        let has_operands = !self.check(TokenKind::RBrace);
        if has_operands {
            operation.set_operands(self.parse_op_operands(parent.clone().unwrap())?);
            self.expect(TokenKind::Colon)?;
            let return_type = self.expect(TokenKind::IntType)?;
            let return_type = IntegerType::from_str(&return_type.lexeme);
            let result_type = Arc::new(RwLock::new(return_type));
            operation.set_anonymous_result(result_type)?;
        }
        let op = O::from_operation(operation);
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Parse for ReturnOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let op = Parser::<T>::return_op::<ReturnOp>(parser, parent)?;
        Ok(op)
    }
}
