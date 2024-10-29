use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use xrcf::ir::StringAttr;
use std::sync::RwLock;
use xrcf::dialect::func::Func;
use xrcf::ir::Block;
use xrcf::ir::Op;
use xrcf::ir::Operation;
use xrcf::ir::OperationName;
use xrcf::parser::Parse;
use xrcf::parser::Parser;
use xrcf::parser::ParserDispatch;
use xrcf::parser::TokenKind;

pub struct CallOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

impl CallOp {
    pub fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
}

impl Op for CallOp {
    fn operation_name() -> OperationName {
        OperationName::new("call".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        CallOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}()", self.identifier().unwrap())
    }
}

impl Parse for CallOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        let identifier = parser.expect(TokenKind::BareIdentifier)?;
        let identifier = identifier.lexeme.clone();
        parser.expect(TokenKind::LParen)?;
        parser.expect(TokenKind::RParen)?;
        let operation = Arc::new(RwLock::new(operation));
        let op = CallOp {
            operation: operation.clone(),
            identifier: Some(identifier),
        };
        Ok(Arc::new(RwLock::new(op)))
    }
}

pub struct FuncOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
}

impl Func for FuncOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("def".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        FuncOp {
            operation,
            identifier: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let operation = self.operation.try_read().unwrap();
        write!(f, "{} ", operation.name())?;
        write!(f, "{}", self.identifier().unwrap())?;
        let region = operation.region().unwrap();
        let region = region.try_read().unwrap();
        write!(f, "()")?;
        region.display(f, indent)?;
        Ok(())
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());

        parser.parse_operation_name_into::<FuncOp>(&mut operation)?;
        let identifier = parser.expect(TokenKind::BareIdentifier)?;
        let identifier = identifier.lexeme.clone();

        let operation = Arc::new(RwLock::new(operation));
        let op = FuncOp {
            operation: operation.clone(),
            identifier: Some(identifier),
        };
        let op = Arc::new(RwLock::new(op));
        parser.expect(TokenKind::LParen)?;
        parser.expect(TokenKind::RParen)?;
        let region = parser.region(op.clone())?;
        let mut operation = operation.write().unwrap();
        operation.set_region(Some(region.clone()));
        Ok(op)
    }
}

pub struct PrintOp {
    operation: Arc<RwLock<Operation>>,
    text: Option<StringAttr>,
}

impl PrintOp {
    pub fn text(&self) -> Option<StringAttr> {
        self.text.clone()
    }
}

impl Op for PrintOp {
    fn operation_name() -> OperationName {
        OperationName::new("print".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        PrintOp {
            operation,
            text: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "print")?;
        write!(f, "({})", self.text().unwrap())?;
        Ok(())
    }
}

impl Parse for PrintOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<PrintOp>(&mut operation)?;
        let operation = Arc::new(RwLock::new(operation));
        parser.expect(TokenKind::LParen)?;
        let text = parser.parse_string()?;
        parser.expect(TokenKind::RParen)?;
        let op = PrintOp {
            operation: operation.clone(),
            text: Some(text),
        };

        Ok(Arc::new(RwLock::new(op)))
    }
}
