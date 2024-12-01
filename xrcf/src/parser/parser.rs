use crate::dialect::arith;
use crate::dialect::cf;
use crate::dialect::experimental;
use crate::dialect::func;
use crate::dialect::llvm;
use crate::dialect::llvm::LLVM;
use crate::ir::Attribute;
use crate::ir::Block;
use crate::ir::BooleanAttr;
use crate::ir::GuardedBlock;
use crate::ir::GuardedOp;
use crate::ir::GuardedOperation;
use crate::ir::GuardedRegion;
use crate::ir::IntegerType;
use crate::ir::ModuleOp;
use crate::ir::Op;
use crate::ir::Operation;
use crate::ir::Region;
use crate::ir::Type;
use crate::ir::TypeParse;
use crate::ir::Value;
use crate::ir::Values;
use crate::parser::scanner::Scanner;
use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

/// Interface to add custom operations to the parser.
///
/// Clients can implement this trait to support custom parsing. The default
/// implementation can only know about operations defined in this crate.  To
/// support custom operations, implement this trait with custom logic, see
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
    /// Return true if the next token is a boolean.
    ///
    /// Clients can implement this trait to support custom boolean parsing.
    fn is_boolean<T: ParserDispatch>(parser: &mut Parser<T>) -> bool {
        let peek = parser.peek();
        peek.kind == TokenKind::BareIdentifier && (peek.lexeme == "true" || peek.lexeme == "false")
    }
    /// Parse a MLIR-style boolean such as `true : i1` or `false : i1`.
    ///
    /// Clients can implement this trait to support custom boolean parsing.
    fn parse_boolean<T: ParserDispatch>(parser: &mut Parser<T>) -> Result<Arc<dyn Attribute>> {
        let token = parser.expect(TokenKind::BareIdentifier)?;
        let value = BooleanAttr::from_str(&token.lexeme);
        Ok(Arc::new(value))
    }
}

/// Default operation parser.
///
/// This parser knows about all operations defined in this crate. For
/// operations in external dialects, define another parser dispatcher and use
/// it.
pub struct DefaultParserDispatch;

pub fn default_dispatch<T: ParserDispatch>(
    name: Token,
    parser: &mut Parser<T>,
    parent: Option<Arc<RwLock<Block>>>,
) -> Result<Arc<RwLock<dyn Op>>> {
    match name.lexeme.clone().as_str() {
        "arith.addi" => <arith::AddiOp as Parse>::op(parser, parent),
        "arith.constant" => <arith::ConstantOp as Parse>::op(parser, parent),
        "cf.br" => <cf::BranchOp as Parse>::op(parser, parent),
        "cf.cond_br" => <cf::CondBranchOp as Parse>::op(parser, parent),
        "experimental.printf" => <experimental::PrintfOp as Parse>::op(parser, parent),
        "func.call" => <func::CallOp as Parse>::op(parser, parent),
        "func.func" => <func::FuncOp as Parse>::op(parser, parent),
        "llvm.add" => <llvm::AddOp as Parse>::op(parser, parent),
        "llvm.alloca" => <llvm::AllocaOp as Parse>::op(parser, parent),
        "llvm.br" => <llvm::BranchOp as Parse>::op(parser, parent),
        "llvm.call" => <llvm::CallOp as Parse>::op(parser, parent),
        "llvm.cond_br" => <llvm::CondBranchOp as Parse>::op(parser, parent),
        "llvm.func" => <llvm::FuncOp as Parse>::op(parser, parent),
        "llvm.mlir.constant" => <llvm::ConstantOp as Parse>::op(parser, parent),
        "llvm.mlir.global" => <llvm::GlobalOp as Parse>::op(parser, parent),
        "llvm.return" => <llvm::ReturnOp as Parse>::op(parser, parent),
        "llvm.store" => <llvm::StoreOp as Parse>::op(parser, parent),
        "module" => <ModuleOp as Parse>::op(parser, parent),
        "return" => <func::ReturnOp as Parse>::op(parser, parent),
        _ => {
            let msg = parser.error(&name, &format!("Unknown operation: {}", name.lexeme));
            return Err(anyhow::anyhow!(msg));
        }
    }
}

impl ParserDispatch for DefaultParserDispatch {
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let name = if parser.peek_n(1).unwrap().kind == TokenKind::Equal {
            // Ignore result name and '=' (e.g., `%0 = <op name>`).
            parser.peek_n(2).unwrap().clone()
        } else {
            // Ignore nothing (e.g., `<op name> %0, %1`).
            parser.peek().clone()
        };
        default_dispatch(name, parser, parent)
    }
    fn parse_type(parser: &mut Parser<Self>) -> Result<Arc<RwLock<dyn Type>>> {
        if parser.check(TokenKind::IntType) {
            let typ = parser.advance();
            let typ = IntegerType::from_str(&typ.lexeme);
            return Ok(Arc::new(RwLock::new(typ)));
        }
        let text = parser.parse_type_text()?;
        if text.is_empty() {
            panic!("Expected type but got empty string");
        }
        if text.starts_with("!llvm") {
            return LLVM::parse_type(&text);
        }
        todo!("Not yet implemented for '{text}'")
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
    fn is_block_definition(&self) -> bool {
        self.peek().kind == TokenKind::CaretIdentifier
    }
    fn is_region_end(&self) -> bool {
        self.peek().kind == TokenKind::RBrace
    }
    pub fn block(&mut self, parent: Option<Arc<RwLock<Region>>>) -> Result<Arc<RwLock<Block>>> {
        assert!(
            parent.is_some(),
            "Expected parent region to be passed when parsing a block"
        );

        let (label, arguments) = if self.is_block_definition() {
            let label = self.expect(TokenKind::CaretIdentifier)?;
            let label = label.lexeme.to_string();
            let arguments = if self.check(TokenKind::LParen) {
                self.parse_function_arguments()?
            } else {
                Values::default()
            };
            self.expect(TokenKind::Colon)?;
            (Some(label), arguments)
        } else {
            let values = Values::default();
            (None, values)
        };

        let ops = vec![];
        let ops = Arc::new(RwLock::new(ops));
        let block = Block::new(label, arguments.clone(), ops.clone(), parent);
        let block = Arc::new(RwLock::new(block));
        for argument in arguments.vec().try_read().unwrap().iter() {
            let mut argument = argument.try_write().unwrap();
            if let Value::BlockArgument(arg) = &mut *argument {
                arg.set_parent(Some(block.clone()));
            } else {
                panic!("Expected a block argument");
            }
        }
        while !self.is_region_end() && !self.is_block_definition() {
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
        let ops = block.ops();
        let ops = ops.try_read().unwrap();
        for op in ops.iter() {
            op.operation().set_parent(Some(block.clone()));
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
        self.expect(TokenKind::LBrace)?;
        let blocks = vec![];
        let blocks = Arc::new(RwLock::new(blocks));
        region.set_blocks(blocks.clone());
        while !self.is_region_end() {
            let block = self.block(Some(region.clone()))?;
            let mut blocks = blocks.try_write().unwrap();
            blocks.push(block);
        }
        self.expect(TokenKind::RBrace)?;
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
    pub fn empty_type(&self) -> bool {
        let lparen = self.check(TokenKind::LParen);
        let rparen = {
            let peek = self.peek_n(1);
            peek.is_some() && peek.unwrap().kind == TokenKind::RParen
        };
        lparen && rparen
    }
    pub fn parse_empty_type(&mut self) -> Result<()> {
        if self.empty_type() {
            self.expect(TokenKind::LParen)?;
            self.expect(TokenKind::RParen)?;
            Ok(())
        } else {
            let msg = self.error(self.peek(), "Expected empty type");
            Err(anyhow::anyhow!(msg))
        }
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
            let arguments = Values::default();
            let block = Block::new(None, arguments, ops.clone(), Some(module_region.clone()));
            let block = Arc::new(RwLock::new(block));
            {
                let ops = ops.read().unwrap();
                for child_op in ops.iter() {
                    child_op.operation().set_parent(Some(block.clone()));
                }
            }
            module_region
                .write()
                .unwrap()
                .blocks()
                .try_write()
                .unwrap()
                .push(block.clone());
            let mut module_operation = Operation::default();
            module_operation.set_name(ModuleOp::operation_name());
            module_operation.set_region(Some(module_region.clone()));
            let module_op = ModuleOp::from_operation(module_operation);
            let module_op = Arc::new(RwLock::new(module_op));
            module_region.set_parent(Some(module_op.clone()));
            module_op
        };
        Ok(op)
    }
    pub fn parse_type(&mut self) -> Result<Arc<RwLock<dyn Type>>> {
        T::parse_type(self)
    }
    pub fn is_boolean(&mut self) -> bool {
        T::is_boolean(self)
    }
    pub fn parse_boolean(&mut self) -> Result<Arc<dyn Attribute>> {
        T::parse_boolean(self)
    }
    /// Parse a type to a string.
    ///
    /// This is used to parse types without having to backtrack or do multiple
    /// peeks. Instead, this method provides a full string which then can be
    /// passed around.
    ///
    /// Examples:
    /// ```mlir
    /// !llvm.array<i32>
    ///
    /// !llvm.ptr
    /// ```
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
