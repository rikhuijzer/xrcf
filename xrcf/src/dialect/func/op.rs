use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::PlaceholderType;
use crate::ir::StrAttr;
use crate::ir::Type;
use crate::ir::Types;
use crate::ir::Values;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use crate::parser::Parse;
use crate::parser::Parser;
use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

pub trait Func: Op {
    fn identifier(&self) -> Option<String>;
    fn set_identifier(&mut self, identifier: String);
    fn sym_visibility(&self) -> Option<String> {
        let operation = self.operation();
        let operation = operation.read().unwrap();
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
            let attribute = StrAttr::new(&visibility);
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
    fn return_types(&self) -> Types {
        let operation = self.operation();
        let operation = operation.read().unwrap();
        let return_types = operation.result_types();
        return_types.clone()
    }
    fn return_type(&self) -> Result<Arc<RwLock<dyn Type>>> {
        let return_types = self.return_types();
        let return_types = return_types.vec();
        let return_types = return_types.try_read().unwrap();
        assert!(!return_types.is_empty(), "Expected result types to be set");
        assert!(return_types.len() == 1, "Expected single result type");
        let typ = return_types[0].clone();
        Ok(typ)
    }
}

/// `func.call`
///
/// ```ebnf
/// `func.call` $callee `(` $operands `)` attr-dict `:` `(` type($operands) `)` -> type($results)
/// ```
pub struct CallOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

pub trait Call: Op {
    fn identifier(&self) -> Option<String>;
    fn set_identifier(&mut self, identifier: String);
    fn display_call_op(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let operation = self.operation().read().unwrap();
        write!(f, "{} = ", operation.results())?;
        write!(f, "{}", operation.name())?;
        write!(f, " {}", self.identifier().unwrap())?;
        write!(f, "({})", operation.operands())?;
        write!(f, " : ")?;
        write!(f, "({})", operation.operand_types())?;
        write!(f, " -> ")?;
        let result_type = operation.result_type().unwrap();
        let result_type = result_type.try_read().unwrap();
        write!(f, "{}", result_type)?;
        Ok(())
    }
    fn parse_call_op<T: ParserDispatch, O: Call + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<O>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let results = parser.parse_op_results_into(&mut operation)?;
        parser.parse_operation_name_into::<O>(&mut operation)?;
        let identifier = parser.expect(TokenKind::AtIdentifier)?;
        let identifier = identifier.lexeme.clone();

        parser.expect(TokenKind::LParen)?;
        let operand = parser.parse_op_operand_into(parent.unwrap(), &mut operation)?;
        parser.expect(TokenKind::RParen)?;

        parser.expect(TokenKind::Colon)?;

        parser.expect(TokenKind::LParen)?;
        let operand_type = T::parse_type(parser)?;
        parser.verify_type(operand, operand_type)?;
        parser.expect(TokenKind::RParen)?;

        parser.expect(TokenKind::Arrow)?;
        let result_type = T::parse_type(parser)?;
        operation.set_result_type(result_type);

        let operation = Arc::new(RwLock::new(operation));
        let mut op = O::from_operation_without_verify(operation.clone(), O::operation_name())?;
        op.set_identifier(identifier);
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
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("func.call".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(CallOp {
            operation,
            identifier: None,
        })
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
        let call_op = CallOp::parse_call_op::<T, CallOp>(parser, parent)?;
        Ok(call_op)
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
        let arguments = op.operation().try_read().unwrap().arguments();
        write!(f, "{}", arguments)?;
        write!(f, ")")?;
        let operation = op.operation();
        let operation = operation.try_read().unwrap();
        let result_types = operation.result_types();
        let result_types = result_types.vec();
        let result_types = result_types.try_read().unwrap();
        if !result_types.is_empty() {
            if result_types.len() == 1 {
                write!(
                    f,
                    " -> {}",
                    result_types.get(0).unwrap().try_read().unwrap()
                )?;
            } else {
                write!(
                    f,
                    " -> ({})",
                    result_types
                        .iter()
                        .map(|t| t.try_read().unwrap().to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                )?;
            }
        }
        let attributes = operation.attributes();
        if !attributes.is_empty() {
            write!(f, " attributes {attributes}")?;
        }
        let region = op.operation().try_read().unwrap().region();
        if let Some(region) = region {
            let region = region.try_read().unwrap();
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
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(FuncOp {
            identifier: None,
            sym_visibility: None,
            operation,
        })
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
        let operation = operation.read().unwrap();
        let arguments = operation.arguments();
        Ok(arguments.clone())
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let identifier = self.identifier.clone();
        FuncOp::display_func(self, identifier.unwrap(), f, indent)
    }
}

impl<T: ParserDispatch> Parser<T> {
    fn result_types(&mut self) -> Result<Types> {
        let mut result_types: Vec<Arc<RwLock<dyn Type>>> = vec![];
        if !self.check(TokenKind::Arrow) {
            return Ok(Types::new(vec![]));
        } else {
            let _arrow = self.advance();
            while self.check(TokenKind::IntType) {
                let typ = self.advance();
                let typ = PlaceholderType::new(&typ.lexeme);
                let typ = Arc::new(RwLock::new(typ));
                result_types.push(typ);
            }
        }
        Ok(Types::new(result_types))
    }
    pub fn parse_func<F: Func + 'static>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
        expected_name: OperationName,
    ) -> Result<Arc<RwLock<F>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent);
        parser.parse_operation_name_into::<F>(&mut operation)?;
        let visibility = FuncOp::try_parse_func_visibility(parser, &expected_name);
        let identifier = parser.expect(TokenKind::AtIdentifier)?;
        let identifier = identifier.lexeme.clone();
        operation.set_name(expected_name.clone());
        operation.set_arguments(parser.parse_function_arguments()?);
        operation.set_result_types(parser.result_types()?);
        let operation = Arc::new(RwLock::new(operation));
        let mut op = F::from_operation_without_verify(operation.clone(), expected_name)?;
        op.set_identifier(identifier);
        op.set_sym_visibility(visibility);
        let op = Arc::new(RwLock::new(op));
        let has_implementation = parser.check(TokenKind::LBrace);
        if has_implementation {
            let region = parser.region(op.clone())?;
            let mut operation = operation.write().unwrap();
            operation.set_region(Some(region.clone()));

            let mut region = region.write().unwrap();
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
        let expected_name = FuncOp::operation_name();
        let op = Parser::<T>::parse_func::<FuncOp>(parser, parent, expected_name)?;
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
        let operation = operation.read().unwrap();
        write!(f, "{name}")?;
        let operands = operation.operands().vec();
        let operands = operands.try_read().unwrap();
        for operand in operands.iter() {
            write!(f, " {}", operand.read().unwrap())?;
        }
        let result_types = operation.result_types();
        let result_types = result_types.vec();
        let result_types = result_types.try_read().unwrap();
        assert!(!result_types.is_empty(), "Expected result types to be set");
        let result_types = result_types
            .iter()
            .map(|t| t.read().unwrap().to_string())
            .collect::<Vec<String>>()
            .join(", ");
        write!(f, " : {}", result_types)?;
        Ok(())
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("return".to_string())
    }
    fn from_operation_without_verify(
        operation: Arc<RwLock<Operation>>,
        name: OperationName,
    ) -> Result<Self> {
        operation.try_write().unwrap().set_name(name);
        Ok(ReturnOp { operation })
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
        expected_name: OperationName,
    ) -> Result<Arc<RwLock<O>>> {
        tracing::debug!("Parsing return op");
        let mut operation = Operation::default();
        assert!(parent.is_some());
        operation.set_parent(parent.clone());
        self.parse_operation_name_into::<O>(&mut operation)?;
        operation.set_operands(self.parse_op_operands(parent.clone().unwrap())?);
        let _colon = self.expect(TokenKind::Colon)?;
        let return_type = self.expect(TokenKind::IntType)?;
        let return_type = PlaceholderType::new(&return_type.lexeme);
        let result_type = Arc::new(RwLock::new(return_type));
        operation.set_result_type(result_type);
        let operation = Arc::new(RwLock::new(operation));
        let op = O::from_operation_without_verify(operation.clone(), expected_name)?;
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Parse for ReturnOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let expected_name = ReturnOp::operation_name();
        let op = Parser::<T>::return_op::<ReturnOp>(parser, parent, expected_name)?;
        Ok(op)
    }
}
