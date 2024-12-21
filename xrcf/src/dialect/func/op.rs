use crate::frontend::Parse;
use crate::frontend::Parser;
use crate::frontend::ParserDispatch;
use crate::frontend::TokenKind;
use crate::ir::AnyType;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::IntegerType;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::Region;
use crate::ir::StringAttr;
use crate::ir::Type;
use crate::ir::UnsetOp;
use crate::ir::Values;
use crate::shared::Shared;
use crate::shared::SharedExt;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;

const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

/// `func.call`
///
/// ```ebnf
/// `func.call` $callee `(` $operands `)` attr-dict `:` `(` type($operands) `)` -> type($results)
/// ```
pub struct CallOp {
    operation: Shared<Operation>,
    identifier: Option<String>,
    varargs: Option<Shared<dyn Type>>,
}

pub trait Call: Op {
    fn identifier(&self) -> Option<String>;
    fn set_identifier(&mut self, identifier: String);
    fn varargs(&self) -> Option<Shared<dyn Type>>;
    fn set_varargs(&mut self, varargs: Option<Shared<dyn Type>>);
    fn display_call_op(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let operation = self.operation().rd();
        let results = operation.results();
        let has_results = !results.vec().rd().is_empty();
        if has_results {
            write!(f, "{} = ", results)?;
        }
        write!(f, "{}", operation.name())?;
        write!(f, " {}", self.identifier().unwrap())?;
        write!(f, "({})", operation.operands())?;
        if let Some(varargs) = self.varargs() {
            write!(f, " vararg({})", varargs.rd())?;
        }
        write!(f, " : ")?;
        write!(f, "({})", operation.operand_types())?;
        write!(f, " -> ")?;
        if has_results {
            write!(
                f,
                "{}",
                operation.result_type(0).expect("no result type").rd()
            )?;
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
        parent: Option<Shared<Block>>,
        allow_varargs: bool,
    ) -> Result<Shared<O>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let results = parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        if parser.check(TokenKind::Equal) {
            parser.advance();
        }
        parser.parse_operation_name_into::<O>(&mut operation)?;
        let identifier = parser.expect(TokenKind::AtIdentifier)?;
        let identifier = identifier.lexeme.clone();

        parser.expect(TokenKind::LParen)?;
        let operands =
            parser.parse_op_operands_into(parent.clone().unwrap(), TOKEN_KIND, &mut operation)?;
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
        if !operands.vec().rd().is_empty() {
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
        let op = Shared::new(op.into());
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
    fn varargs(&self) -> Option<Shared<dyn Type>> {
        self.varargs.clone()
    }
    fn set_varargs(&mut self, varargs: Option<Shared<dyn Type>>) {
        self.varargs = varargs;
    }
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("func.call".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        CallOp {
            operation,
            identifier: None,
            varargs: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
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
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let allow_varargs = false;
        let call_op = CallOp::parse_call_op::<T, CallOp>(parser, parent, allow_varargs)?;
        Ok(call_op)
    }
}

pub trait Func: Op {
    fn identifier(&self) -> Option<String>;
    fn set_identifier(&mut self, identifier: String);
    fn sym_visibility(&self) -> Option<String> {
        let operation = self.operation();
        let attributes = operation.rd().attributes();
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
            let operation = operation.wr();
            let attributes = operation.attributes();
            let attribute = StringAttr::from_str(&visibility);
            let attribute = Arc::new(attribute);
            attributes.insert("sym_visibility", attribute);
        }
    }
    fn arguments(&self) -> Result<Values> {
        Ok(self.operation().rd().arguments().clone())
    }
    fn return_types(&self) -> Vec<Shared<dyn Type>> {
        self.operation().rd().results().types().vec()
    }
    fn return_type(&self) -> Result<Shared<dyn Type>> {
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
    operation: Shared<Operation>,
}

impl FuncOp {
    /// Insert `op` into the region of `self`, while creating a region if necessary.
    pub fn insert_op(&self, op: Shared<dyn Op>) -> UnsetOp {
        let read = op.rd();
        let ops = read.ops();
        if ops.is_empty() {
            let operation = self.operation();
            let region = operation.rd().region();
            if region.is_some() {
                panic!("Expected region to be empty");
            }
            let ops = vec![op.clone()];
            let ops = Shared::new(ops.into());
            let region = Region::default();
            let without_parent = region.add_empty_block();
            let region = Shared::new(region.into());
            let block = without_parent.set_parent(Some(region.clone()));
            block.wr().set_ops(ops);
            operation.wr().set_region(Some(region));
        } else {
            ops.last().unwrap().rd().insert_after(op.clone());
        }
        UnsetOp::new(op.clone())
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
        write!(f, "{} ", op.operation().rd().name())?;
        if let Some(visibility) = FuncOp::display_visibility(op) {
            write!(f, "{visibility} ")?;
        }
        write!(f, "{identifier}(")?;
        write!(f, "{}", op.operation().rd().arguments())?;
        write!(f, ")")?;
        let operation = op.operation();
        let result_types = operation.rd().results().types();
        if !result_types.vec().is_empty() {
            write!(f, " -> {}", result_types)?;
        }
        let attributes = operation.rd().attributes();
        if !attributes.is_empty() {
            write!(f, " attributes {attributes}")?;
        }
        if let Some(region) = op.operation().rd().region() {
            region.rd().display(f, indent)?;
        }
        Ok(())
    }
    fn try_parse_func_visibility<T: ParserDispatch>(
        parser: &mut Parser<T>,
        expected_name: &OperationName,
    ) -> Option<String> {
        if expected_name == &FuncOp::operation_name() {
            if parser.check(TokenKind::BareIdentifier) {
                let sym_visibility = parser.advance().lexeme.clone();
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
    fn new(operation: Shared<Operation>) -> Self {
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
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn assignments(&self) -> Result<Values> {
        let operation = self.operation();
        let arguments = operation.rd().arguments();
        Ok(arguments.clone())
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let identifier = self.identifier.clone();
        FuncOp::display_func(self, identifier.unwrap(), f, indent)
    }
}

impl<T: ParserDispatch> Parser<T> {
    fn result_types(&mut self) -> Result<Vec<Shared<dyn Type>>> {
        let mut result_types: Vec<Shared<dyn Type>> = vec![];
        if !self.check(TokenKind::Arrow) {
            return Ok(result_types);
        } else {
            let _arrow = self.advance();
            while self.check(TokenKind::IntType) {
                let typ = self.advance();
                let typ = AnyType::new(&typ.lexeme);
                let typ = Shared::new(typ.into());
                result_types.push(typ);
            }
        }
        Ok(result_types)
    }
    pub fn parse_func<F: Func + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<F>> {
        let mut operation = Operation::default();
        operation.set_parent(parent);
        parser.parse_operation_name_into::<F>(&mut operation)?;
        let expected_name = F::operation_name();
        let visibility = FuncOp::try_parse_func_visibility(parser, &expected_name);
        let identifier = parser.expect(TokenKind::AtIdentifier)?;
        let identifier = identifier.lexeme.clone();
        let arguments = parser.parse_function_arguments()?;
        operation.set_arguments(arguments.clone());
        operation.set_anonymous_results(parser.result_types()?)?;
        let mut op = F::from_operation(operation);
        op.set_identifier(identifier);
        op.set_sym_visibility(visibility);
        let op = Shared::new(op.into());
        let has_implementation = parser.check(TokenKind::LBrace);
        if has_implementation {
            let region = parser.parse_region(op.clone())?;
            let op_rd = op.rd();
            op_rd.operation().wr().set_region(Some(region.clone()));
            region.wr().set_parent(Some(op.clone()));

            let block = region.rd().blocks().into_iter().next().unwrap();
            for argument in arguments.into_iter() {
                argument.wr().set_parent(Some(block.clone()));
            }
        }

        Ok(op)
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let op = Parser::<T>::parse_func::<FuncOp>(parser, parent)?;
        Ok(op)
    }
}

pub struct ReturnOp {
    operation: Shared<Operation>,
}

impl ReturnOp {
    pub fn display_return(op: &dyn Op, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = op.operation();
        let name = operation.rd().name();
        write!(f, "{name}")?;
        let operands = operation.rd().operands().into_iter();
        if operands.len() != 0 {
            for operand in operands {
                write!(f, " {}", operand.rd())?;
            }
            write!(f, " : {}", operation.rd().results().types())?;
        }
        Ok(())
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("return".to_string())
    }
    fn new(operation: Shared<Operation>) -> Self {
        ReturnOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Shared<Operation> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        ReturnOp::display_return(self, f, _indent)
    }
}

impl<T: ParserDispatch> Parser<T> {
    pub fn return_op<O: Op>(&mut self, parent: Option<Shared<Block>>) -> Result<Shared<O>> {
        let mut operation = Operation::default();
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        self.parse_operation_name_into::<O>(&mut operation)?;
        let has_operands = !self.check(TokenKind::RBrace);
        if has_operands {
            operation.set_operands(self.parse_op_operands(parent.clone().unwrap(), TOKEN_KIND)?);
            self.expect(TokenKind::Colon)?;
            let return_type = self.expect(TokenKind::IntType)?;
            let return_type = IntegerType::from_str(&return_type.lexeme);
            let result_type = Shared::new(return_type.into());
            operation.set_anonymous_result(result_type)?;
        }
        let op = O::from_operation(operation);
        let op = Shared::new(op.into());
        Ok(op)
    }
}

impl Parse for ReturnOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Shared<Block>>,
    ) -> Result<Shared<dyn Op>> {
        let op = Parser::<T>::return_op::<ReturnOp>(parser, parent)?;
        Ok(op)
    }
}
