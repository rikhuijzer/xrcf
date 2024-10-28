use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::llvm;
use crate::dialect::llvm::LLVM;
use crate::dialect::unstable;
use crate::ir::Block;
use crate::ir::IntegerType;
use crate::ir::ModuleOp;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Type;
use crate::ir::TypeParse;
use crate::parser::scanner::Scanner;
use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

/// Interface to add custom operations to the parser.
///
/// Downstream crates can implement this trait to support custom parsing. The
/// default implementation can only know about operations defined in this crate.
/// To support custom operations, implement this trait with custom logic, see
/// `DefaultParserDispatch` for an example.
pub trait ParserDispatch {
    /// Parse an operation.
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>>
    where
        Self: Sized;
    /// Parse a type.
    fn parse_type(parser: &mut Parser<Self>) -> Result<Arc<RwLock<dyn Type>>>
    where
        Self: Sized;
}

/// Default operation parser.
///
/// This parser knows about all operations defined in this crate.  For
/// operations in external dialects, define another parser dispatcher and use
/// it.
pub struct DefaultParserDispatch;

impl ParserDispatch for DefaultParserDispatch {
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let name = if parser.peek_n(1).unwrap().kind == TokenKind::Equal {
            // Ignore result name and '=' (e.g., `%0 = <op name>`).
            parser.peek_n(2).unwrap()
        } else {
            // Ignore nothing (e.g., `<op name> %0, %1`).
            parser.peek()
        };
        match name.lexeme.clone().as_str() {
            "arith.addi" => <arith::AddiOp as Parse>::op(parser, parent),
            "arith.constant" => <arith::ConstantOp as Parse>::op(parser, parent),
            "func.call" => <func::CallOp as Parse>::op(parser, parent),
            "func.func" => <func::FuncOp as Parse>::op(parser, parent),
            "llvm.add" => <llvm::AddOp as Parse>::op(parser, parent),
            "llvm.alloca" => <llvm::AllocaOp as Parse>::op(parser, parent),
            "llvm.call" => <llvm::CallOp as Parse>::op(parser, parent),
            "llvm.func" => <llvm::FuncOp as Parse>::op(parser, parent),
            "llvm.mlir.constant" => <llvm::ConstantOp as Parse>::op(parser, parent),
            "llvm.mlir.global" => <llvm::GlobalOp as Parse>::op(parser, parent),
            "llvm.return" => <llvm::ReturnOp as Parse>::op(parser, parent),
            "llvm.store" => <llvm::StoreOp as Parse>::op(parser, parent),
            "module" => <ModuleOp as Parse>::op(parser, parent),
            "return" => <func::ReturnOp as Parse>::op(parser, parent),
            "unstable.printf" => <unstable::PrintfOp as Parse>::op(parser, parent),
            _ => {
                let msg = parser.error(&name, &format!("Unknown operation: {}", name.lexeme));
                return Err(anyhow::anyhow!(msg));
            }
        }
    }
    fn parse_type(parser: &mut Parser<Self>) -> Result<Arc<RwLock<dyn Type>>> {
        if parser.check(TokenKind::IntType) {
            let typ = parser.advance();
            let typ = IntegerType::from_str(&typ.lexeme);
            return Ok(Arc::new(RwLock::new(typ)));
        }
        let text = parser.parse_type_text()?;
        if text.starts_with("!llvm") {
            return LLVM::parse_type(&text);
        }
        todo!("Not yet implemented for {}", text)
    }
}

/// Interface to define parsing of operations.
///
/// Downstream crates can implement this trait to support parsing of custom
/// operations.
pub trait Parse {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>>
    where
        Self: Sized;
}

pub struct Parser<T: ParserDispatch> {
    src: String,
    tokens: Vec<Token>,
    current: usize,
    parse_op: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
enum Dialects {
    Builtin,
    LLVM,
}

impl<T: ParserDispatch> Parser<T> {
    pub fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }
    pub fn previous_n(&self, n: usize) -> &Token {
        &self.tokens[self.current - n]
    }
    pub fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }
    pub fn peek(&self) -> &Token {
        self.tokens.get(self.current).unwrap()
    }
    pub fn peek_n(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.current + n)
    }
    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }
    pub fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        self.peek().kind == kind
    }
    pub fn error(&self, token: &Token, msg: &str) -> String {
        let msg = Scanner::error(&self.src, &token.location, msg);
        format!("\n\n{msg}\n")
    }
    pub fn report_token_error(&self, token: &Token, expected: TokenKind) -> Result<Token> {
        let msg = format!(
            "Expected {:?}, but got `{}` of kind {:?}",
            expected, token.lexeme, token.kind
        );
        let msg = Scanner::error(&self.src, &token.location, &msg);
        Err(anyhow::anyhow!(format!("\n\n{msg}\n")))
    }
    pub fn expect(&mut self, kind: TokenKind) -> Result<Token> {
        if self.check(kind) {
            self.advance();
            Ok(self.previous().clone())
        } else {
            self.report_token_error(self.peek(), kind)
        }
    }
    pub fn block(&mut self, parent: Option<Arc<RwLock<Region>>>) -> Result<Arc<RwLock<Block>>> {
        assert!(
            parent.is_some(),
            "Expected parent region to be passed when parsing a block"
        );
        // Not all blocks have a label.
        // let label = self.expect(TokenKind::PercentIdentifier)?;
        // let label = label.lexeme.clone();
        // println!("label: {}", label);
        // let _equal = self.expect(TokenKind::Equal)?;
        let arguments = Arc::new(vec![]);
        let ops = vec![];
        let ops = Arc::new(RwLock::new(ops));
        let block = Block::new(None, arguments, ops.clone(), parent);
        let block = Arc::new(RwLock::new(block));
        loop {
            if self.peek().kind == TokenKind::RBrace {
                break;
            }
            let parent = Some(block.clone());
            let op = T::parse_op(self, parent)?;
            let mut ops = ops.write().unwrap();
            ops.push(op.clone());
        }
        if ops.read().unwrap().is_empty() {
            let token = self.peek();
            let msg = self.error(&token, "Could not find operations in block");
            return Err(anyhow::anyhow!(msg));
        }
        let ops = block.read().unwrap().ops();
        let ops = ops.read().unwrap();
        for op in ops.iter() {
            let op = op.read().unwrap();
            let mut operation = op.operation().write().unwrap();
            operation.set_parent(Some(block.clone()));
        }
        Ok(block)
    }
    pub fn match_kinds(&mut self, kinds: &[TokenKind]) -> bool {
        for kind in kinds {
            if self.check(*kind) {
                self.advance();
                return true;
            }
        }
        false
    }
    pub fn region(&mut self, parent: Arc<RwLock<dyn Op>>) -> Result<Arc<RwLock<Region>>> {
        let mut region = Region::default();
        region.set_parent(Some(parent.clone()));
        let region = Arc::new(RwLock::new(region));
        let _lbrace = self.expect(TokenKind::LBrace)?;
        let block = self.block(Some(region.clone()))?;
        let blocks = vec![block];
        region.write().unwrap().set_blocks(blocks);
        self.advance();
        Ok(region)
    }
    /// Return true if the next token could be the start of an operation.
    fn peek_op(&self) -> bool {
        self.check(TokenKind::BareIdentifier)
    }
    pub fn parse_keyword(&mut self, keyword: &str) -> Result<()> {
        let token = self.expect(TokenKind::BareIdentifier)?;
        if token.lexeme != keyword {
            let msg = self.error(&token, &format!("Expected keyword: {}", keyword));
            return Err(anyhow::anyhow!(msg));
        }
        Ok(())
    }
    pub fn parse(src: &str) -> Result<Arc<RwLock<dyn Op>>> {
        let mut parser = Parser::<T> {
            src: src.to_string(),
            tokens: Scanner::scan(src)?,
            current: 0,
            parse_op: std::marker::PhantomData,
        };
        let op = T::parse_op(&mut parser, None)?;
        let opp = op.clone();
        let opp = opp.read().unwrap();
        let casted = opp.as_any().downcast_ref::<ModuleOp>();
        let op: Arc<RwLock<dyn Op>> = if let Some(_module_op) = casted {
            op
        } else {
            let module_region = Region::default();
            let module_region = Arc::new(RwLock::new(module_region));
            let mut ops = vec![op.clone()];

            if parser.peek_op() {
                while parser.peek_op() {
                    let op = T::parse_op(&mut parser, None)?;
                    ops.push(op.clone());
                }
            }
            let ops = Arc::new(RwLock::new(ops));
            let arguments = Arc::new(vec![]);
            let block = Block::new(None, arguments, ops.clone(), Some(module_region.clone()));
            let block = Arc::new(RwLock::new(block));
            {
                let ops = ops.read().unwrap();
                for child_op in ops.iter() {
                    let child_op = child_op.try_read().unwrap();
                    let mut child_operation = child_op.operation().try_write().unwrap();
                    child_operation.set_parent(Some(block.clone()));
                }
            }
            module_region
                .write()
                .unwrap()
                .blocks_mut()
                .push(block.clone());
            let mut module_operation = Operation::default();
            module_operation.set_name(ModuleOp::operation_name());
            module_operation.set_region(Some(module_region.clone()));
            let module_operation = Arc::new(RwLock::new(module_operation));
            let module_op = ModuleOp::from_operation(module_operation);
            let module_op = Arc::new(RwLock::new(module_op));
            module_region
                .write()
                .unwrap()
                .set_parent(Some(module_op.clone()));
            module_op
        };
        Ok(op)
    }
    /// Parse a type to a string (for example, `!llvm.array<i32>`).
    fn parse_type_text(&mut self) -> Result<String> {
        let mut typ = String::new();
        if self.check(TokenKind::Exclamation) {
            let _exclamation = self.advance();
            typ.push('!');
        }
        if self.check(TokenKind::BareIdentifier) {
            let text = self.advance();
            typ.push_str(&text.lexeme);
        }
        if self.check(TokenKind::Less) {
            while !self.check(TokenKind::Greater) {
                let vector_element_typ = self.advance();
                typ.push_str(&vector_element_typ.lexeme);
            }
            let _greater = self.advance();
            typ.push('>');
        }
        Ok(typ)
    }
}
