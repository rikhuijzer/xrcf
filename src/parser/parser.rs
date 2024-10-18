use crate::dialect::arith;
use crate::dialect::func;
use crate::dialect::llvmir;
use crate::ir::operation;
use crate::ir::Block;
use crate::ir::ModuleOp;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::Operation;
use crate::ir::Region;
use crate::parser::scanner::Scanner;
use crate::parser::token::Token;
use crate::parser::token::TokenKind;
use anyhow::Result;
use std::sync::Arc;
use std::sync::RwLock;

/// Interface to add custom operations to the parser.
///
/// Downstream crates can implement this trait to support custom parsing.  The
/// default implementation can only know about operations defined in this crate.
/// This gives the Rust compiler more insight into the dispatches compared to to
/// using a hashmap registry.
pub trait ParserDispatch {
    fn dispatch_parse(
        name: String,
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>>
    where
        Self: Sized;
    fn parse_op(
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>>
    where
        Self: Sized,
    {
        let name = if parser.peek_n(1).kind == TokenKind::Equal {
            // Ignore result name and '=' (e.g., `%0 = <op name>`).
            parser.peek_n(2)
        } else {
            // Ignore nothing (e.g., `<op name> %0, %1`).
            parser.peek()
        };
        let name = name.lexeme.clone();
        Self::dispatch_parse(name, parser, parent)
    }
}

/// Default operation parser.
///
/// This parser knows about all operations defined in this crate.  For
/// operations in external dialects, define another parser dispatcher and use
/// it.
pub struct DefaultParserDispatch;

impl ParserDispatch for DefaultParserDispatch {
    fn dispatch_parse(
        name: String,
        parser: &mut Parser<Self>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        match name.as_str() {
            "return" => <func::ReturnOp as Parse>::op(parser, parent),
            "arith.addi" => <arith::AddiOp as Parse>::op(parser, parent),
            "arith.constant" => <arith::ConstantOp as Parse>::op(parser, parent),
            "func.func" => <func::FuncOp as Parse>::op(parser, parent),
            "llvm.func" => <llvmir::FuncOp as Parse>::op(parser, parent),
            "llvm.mlir.constant" => <llvmir::ConstantOp as Parse>::op(parser, parent),
            "llvm.return" => <llvmir::ReturnOp as Parse>::op(parser, parent),
            "llvm.add" => <llvmir::AddOp as Parse>::op(parser, parent),
            "llvm.mlir.global" => <llvmir::GlobalOp as Parse>::op(parser, parent),
            "module" => <ModuleOp as Parse>::op(parser, parent),
            _ => Err(anyhow::anyhow!("Unknown operation: {}", name)),
        }
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
    pub fn peek_n(&self, n: usize) -> &Token {
        self.tokens.get(self.current + n).unwrap()
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
            "Expected {:?}, but got \"{}\" of kind {:?}",
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
        let mut blocks = vec![];
        let block = self.block(Some(region.clone()))?;
        blocks.push(block);
        region.write().unwrap().set_blocks(blocks);
        self.advance();
        Ok(region)
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
            let mut region = Region::default();
            region.set_parent(Some(op.clone()));
            let region = Arc::new(RwLock::new(region));
            let ops = Arc::new(RwLock::new(vec![op.clone()]));
            let arguments = Arc::new(vec![]);
            let block = Block::new(None, arguments, ops, Some(region.clone()));
            let block = Arc::new(RwLock::new(block));
            {
                let func_op = op.try_read().unwrap();
                let mut func_operation = func_op.operation().try_write().unwrap();
                func_operation.set_parent(Some(block.clone()));
            }
            region.write().unwrap().blocks_mut().push(block.clone());
            let mut module_operation = Operation::default();
            module_operation.set_name(ModuleOp::operation_name());
            module_operation.set_region(Some(region.clone()));
            let module_operation = Arc::new(RwLock::new(module_operation));
            let module_op = ModuleOp::from_operation(module_operation);
            Arc::new(RwLock::new(module_op.unwrap()))
        };
        Ok(op)
    }
}

impl<T: ParserDispatch> Parser<T> {
    // Parse %0.
    fn operand(&mut self, parent: Arc<RwLock<Block>>) -> Result<Arc<RwLock<OpOperand>>> {
        let identifier = self.expect(TokenKind::PercentIdentifier)?;
        let name = identifier.lexeme.clone();
        let block = parent.read().unwrap();
        let assignment = block.assignment(&name);
        let assignment = match assignment {
            Some(assignment) => assignment,
            None => {
                let msg = "Expected assignment before use.";
                let msg = self.error(&identifier, msg);
                return Err(anyhow::anyhow!(msg));
            }
        };
        let operand = OpOperand::new(assignment);
        Ok(Arc::new(RwLock::new(operand)))
    }
    /// Parse %0, %1.
    pub fn operands(&mut self, parent: Arc<RwLock<Block>>) -> Result<operation::Operands> {
        let mut arguments = vec![];
        while self.check(TokenKind::PercentIdentifier) {
            let operand = self.operand(parent.clone())?;
            arguments.push(operand);
            if self.check(TokenKind::Comma) {
                let _comma = self.advance();
            }
        }
        Ok(Arc::new(RwLock::new(arguments)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Op;

    #[test]
    fn parse_global() {
        // From test/Target/LLVMIR/llvmir.mlir
        let src = "llvm.mlir.global internal @i32_global(42 : i32) : i32";
        let module_op = Parser::<DefaultParserDispatch>::parse(src).unwrap();
        let module_op = module_op.try_read().unwrap();
        let module_op = module_op.as_any().downcast_ref::<ModuleOp>().unwrap();
        assert_eq!(
            module_op.operation().read().unwrap().name().to_string(),
            "module"
        );
        // let body = module_op.operation().get_body_region();
        // assert_eq!(body.blocks().len(), 1);
        module_op.first_op().unwrap();

        let repr = format!("{:#}", module_op);
        let lines: Vec<&str> = repr.split('\n').collect();
        println!("-- After:\n{}\n", repr);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "module {");
        assert_eq!(lines[1], "  llvm.mlir.global internal @i32_global(42)");
        assert_eq!(lines[2], "}");
    }
}
