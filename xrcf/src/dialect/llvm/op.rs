use crate::dialect::func;
use crate::dialect::func::Call;
use crate::dialect::func::Func;
use crate::dialect::llvm::attribute::LinkageAttr;
use crate::ir::AnyAttr;
use crate::ir::Attribute;
use crate::ir::Attributes;
use crate::ir::Block;
use crate::ir::GuardedOpOperand;
use crate::ir::GuardedOperation;
use crate::ir::Op;
use crate::ir::OpOperand;
use crate::ir::OpOperands;
use crate::ir::Operation;
use crate::ir::OperationName;
use crate::ir::StringAttr;
use crate::ir::Type;
use crate::parser::Parse;
use crate::parser::Parser;
use crate::parser::ParserDispatch;
use crate::parser::TokenKind;
use anyhow::Result;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;

const TOKEN_KIND: TokenKind = TokenKind::PercentIdentifier;

/// `llvm.add`
pub struct AddOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for AddOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.add".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        AddOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{}", self.operation().read().unwrap())
    }
}

impl Parse for AddOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let op = Parser::<T>::parse_add::<AddOp>(parser, parent)?;
        Ok(op)
    }
}

/// `llvm.alloca`
///
/// ```ebnf
/// `llvm.alloca` `inalloca`? $array_size `x` type attribute-dict? `:` type `->` type
/// ```
pub struct AllocaOp {
    operation: Arc<RwLock<Operation>>,
    element_type: Option<String>,
}

impl AllocaOp {
    pub fn element_type(&self) -> Option<String> {
        self.element_type.clone()
    }
    pub fn set_element_type(&mut self, element_type: String) {
        self.element_type = Some(element_type);
    }
    pub fn array_size(&self) -> Arc<RwLock<OpOperand>> {
        self.operation().operand(0).expect("no operand")
    }
}

impl Op for AllocaOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.alloca".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        AllocaOp {
            operation,
            element_type: None,
        }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().read().unwrap();
        write!(f, "{} = ", operation.results())?;
        write!(f, "{} ", operation.name())?;
        let array_size = self.array_size();
        let array_size = array_size.try_read().unwrap();
        write!(f, "{}", array_size)?;
        write!(f, " x ")?;
        write!(f, "{}", self.element_type().expect("no element type"))?;
        write!(f, " : ")?;
        let array_size_type = array_size.typ().unwrap();
        let array_size_type = array_size_type.try_read().unwrap();
        write!(f, "({})", array_size_type)?;
        write!(f, " -> ")?;
        let result_type = operation.result_type(0).unwrap();
        let result_type = result_type.try_read().unwrap();
        write!(f, "{result_type}")?;
        Ok(())
    }
}

impl Parse for AllocaOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        let results = parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Equal)?;
        parser.parse_operation_name_into::<AllocaOp>(&mut operation)?;
        operation.set_parent(parent.clone());

        let array_size =
            parser.parse_op_operand_into(parent.unwrap(), TOKEN_KIND, &mut operation)?;
        parser.parse_keyword("x")?;
        let element_type = T::parse_type(parser)?;
        let element_type = element_type.try_read().unwrap();
        parser.expect(TokenKind::Colon)?;
        parser.expect(TokenKind::LParen)?;

        let array_size_type = T::parse_type(parser)?;
        parser.verify_type(array_size, array_size_type)?;

        parser.expect(TokenKind::RParen)?;
        parser.expect(TokenKind::Arrow)?;
        let result_type = T::parse_type(parser)?;
        operation.set_result_type(0, result_type)?;

        let op = AllocaOp {
            operation: Arc::new(RwLock::new(operation)),
            element_type: Some(element_type.to_string()),
        };
        let op = Arc::new(RwLock::new(op));
        results.set_defining_op(op.clone());
        Ok(op)
    }
}

/// `llvm.br`
pub struct BranchOp {
    operation: Arc<RwLock<Operation>>,
}

impl BranchOp {
    pub fn dest(&self) -> Option<Arc<RwLock<OpOperand>>> {
        self.operation().operand(0)
    }
    pub fn set_dest(&mut self, dest: Arc<RwLock<OpOperand>>) {
        self.operation().set_operand(0, dest);
    }
}

impl Op for BranchOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.br".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        BranchOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", self.operation.name())?;
        let dest = self.dest();
        let dest = dest.unwrap();
        let dest = dest.try_read().unwrap();
        write!(f, "{}", dest)?;
        let operands = self.operation().operands();
        let operands = operands.vec();
        let operands = operands.try_read().unwrap();
        let operands = operands.iter().skip(1);
        if 0 < operands.len() {
            write!(f, "(")?;
            for operand in operands {
                let operand = operand.try_read().unwrap();
                operand.display_with_type(f)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

impl Parse for BranchOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<BranchOp>(&mut operation)?;

        let operation = Arc::new(RwLock::new(operation));
        let dest = parser.parse_block_dest()?;
        let dest = Arc::new(RwLock::new(dest));
        operation.set_operand(0, dest);
        if parser.check(TokenKind::LParen) {
            parser.expect(TokenKind::LParen)?;
            let operands = operation.operands().vec();
            let mut operands = operands.try_write().unwrap();
            loop {
                let operand = parser.parse_op_operand(parent.clone().unwrap(), TOKEN_KIND)?;
                operands.push(operand);
                parser.expect(TokenKind::Colon)?;
                let _typ = parser.advance();
                if !parser.check(TokenKind::Comma) {
                    break;
                }
            }
            parser.expect(TokenKind::RParen)?;
        }
        let op = BranchOp { operation };
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

impl Display for BranchOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

/// `llvm.call`
///
/// ```ebnf
/// `llvm.call` $callee `(` $operands `)` attr-dict `:` `(` type($operands) `)` -> type($results)
/// ```
pub struct CallOp {
    operation: Arc<RwLock<Operation>>,
    identifier: Option<String>,
    varargs: Option<Arc<RwLock<dyn Type>>>,
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
        OperationName::new("llvm.call".to_string())
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
        let allow_varargs = true;
        let op = CallOp::parse_call_op::<T, CallOp>(parser, parent, allow_varargs)?;
        Ok(op)
    }
}

/// `llvm.cond_br`
pub struct CondBranchOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for CondBranchOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.cond_br".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        CondBranchOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
}

impl Parse for CondBranchOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent.clone());
        parser.parse_operation_name_into::<CondBranchOp>(&mut operation)?;
        parser.parse_op_operands_into(parent.clone().unwrap(), TOKEN_KIND, &mut operation)?;

        let operation = Arc::new(RwLock::new(operation));
        let op = CondBranchOp { operation };
        let op = Arc::new(RwLock::new(op));
        Ok(op)
    }
}

pub struct ConstantOp {
    operation: Arc<RwLock<Operation>>,
}

impl ConstantOp {
    pub fn value(&self) -> Arc<dyn Attribute> {
        self.operation().attributes().get("value").unwrap().clone()
    }
    pub fn set_value(&self, value: Arc<dyn Attribute>) {
        self.operation().attributes().insert("value", value);
    }
}

impl Op for ConstantOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.mlir.constant".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        ConstantOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn is_const(&self) -> bool {
        true
    }
    fn is_pure(&self) -> bool {
        true
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let results = self.operation().results();
        let results = results.vec();
        let results = results.try_read().unwrap();
        let result = results.get(0).expect("no result");
        let result = result.try_read().unwrap();
        write!(f, "{} = {}", result, Self::operation_name())?;
        write!(f, "(")?;
        let value = self.value();
        write!(f, "{}", value)?;
        write!(f, ")")?;

        let typ = self.operation().result_type(0).expect("no result type");
        let typ = typ.try_read().unwrap();
        write!(f, " : {typ}")?;

        Ok(())
    }
}

impl Display for ConstantOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.display(f, 0)
    }
}

impl Parse for ConstantOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        operation.set_parent(parent);
        let results = parser.parse_op_results_into(TOKEN_KIND, &mut operation)?;
        parser.expect(TokenKind::Equal)?;
        parser.parse_operation_name_into::<ConstantOp>(&mut operation)?;

        parser.expect(TokenKind::LParen)?;
        let value: Arc<dyn Attribute> = if parser.check(TokenKind::Integer) {
            Arc::new(parser.parse_integer()?)
        } else if parser.is_boolean() {
            parser.parse_boolean()?
        } else {
            Arc::new(parser.parse_string()?)
        };
        parser.expect(TokenKind::RParen)?;

        operation.attributes().insert("value", value);

        parser.expect(TokenKind::Colon)?;
        let typ = T::parse_type(parser)?;
        operation.set_result_type(0, typ)?;

        let operation = Arc::new(RwLock::new(operation));
        let op = ConstantOp {
            operation: operation.clone(),
        };
        let op = Arc::new(RwLock::new(op));
        results.set_defining_op(op.clone());

        Ok(op)
    }
}
pub struct GlobalOp {
    operation: Arc<RwLock<Operation>>,
}

impl Op for GlobalOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.mlir.global".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        GlobalOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        write!(f, "{} ", Self::operation_name().name())?;
        let attributes = self.operation().attributes().map();
        let attributes = attributes.read().unwrap();
        if let Some(attribute) = attributes.get("linkage") {
            write!(f, "{} ", attribute)?;
        }
        if let Some(attribute) = attributes.get("symbol_name") {
            write!(f, "{}(", attribute)?;
        }
        if let Some(attribute) = attributes.get("value") {
            write!(f, "{}", attribute)?;
        }
        write!(f, ")")?;
        Ok(())
    }
}

pub struct FuncOp {
    identifier: Option<String>,
    operation: Arc<RwLock<Operation>>,
}

impl FuncOp {
    pub fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    pub fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.func".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        FuncOp {
            identifier: None,
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
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        let identifier = self.identifier.clone();
        func::FuncOp::display_func(self, identifier.unwrap(), f, indent)
    }
}

impl Func for FuncOp {
    fn identifier(&self) -> Option<String> {
        self.identifier.clone()
    }
    fn set_identifier(&mut self, identifier: String) {
        self.identifier = Some(identifier);
    }
}

impl Parse for FuncOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let op = Parser::<T>::parse_func::<FuncOp>(parser, parent)?;
        if parser.check(TokenKind::BareIdentifier) {
            let text = parser.peek();
            let text = text.lexeme.clone();
            if text == "attributes" {
                parser.advance();
                let attributes = parser.parse_attributes()?;
                let op = op.try_read().unwrap();
                op.operation().set_attributes(attributes);
            }
        }
        Ok(op)
    }
}

impl Parse for GlobalOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let _operation_name = parser.advance();
        let attributes = Attributes::new();
        if parser.check(TokenKind::BareIdentifier) {
            if let Some(attribute) = LinkageAttr::parse(parser) {
                attributes
                    .map()
                    .write()
                    .unwrap()
                    .insert("linkage".to_string(), Arc::new(attribute));
            }
        }
        let symbol_name = parser.peek();
        if symbol_name.kind != TokenKind::AtIdentifier {
            return Err(anyhow::anyhow!(
                "Expected @identifier, got {:?}",
                symbol_name
            ));
        }
        if let Some(attribute) = StringAttr::parse(parser) {
            attributes
                .map()
                .write()
                .unwrap()
                .insert("symbol_name".to_string(), Arc::new(attribute));
        }
        if parser.check(TokenKind::LParen) {
            parser.advance();
            if let Some(attribute) = AnyAttr::parse(parser) {
                attributes
                    .map()
                    .write()
                    .unwrap()
                    .insert("value".to_string(), Arc::new(attribute));
            }
        }
        let mut operation = Operation::default();
        operation.set_name(GlobalOp::operation_name());
        operation.set_attributes(attributes);
        operation.set_parent(parent);
        let op = GlobalOp::from_operation(operation);
        Ok(Arc::new(RwLock::new(op)))
    }
}

pub struct ReturnOp {
    operation: Arc<RwLock<Operation>>,
}

impl ReturnOp {
    pub fn operand(&self) -> Option<Arc<RwLock<OpOperand>>> {
        self.operation().operand(0)
    }
}

impl Op for ReturnOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.return".to_string())
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
    fn display(&self, f: &mut Formatter<'_>, indent: i32) -> std::fmt::Result {
        func::ReturnOp::display_return(self, f, indent)
    }
}

/// `llvm.store`
///
/// ```ebnf
/// `llvm.store` (`volatile` $volatile_^)? $value `,` $addr
///   (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
///   attr-dict `:` type($value) `,` qualified(type($addr))
/// ```
pub struct StoreOp {
    operation: Arc<RwLock<Operation>>,
}

impl StoreOp {
    pub fn value(&self) -> Arc<RwLock<OpOperand>> {
        self.operation().operand(0).expect("no value set")
    }
    pub fn set_value(&mut self, value: Arc<RwLock<OpOperand>>) {
        self.operation().set_operand(0, value);
    }
    pub fn addr(&self) -> Arc<RwLock<OpOperand>> {
        let operation = self.operation.try_read().unwrap();
        let operand = operation.operand(1).expect("no address set");
        operand
    }
    pub fn set_addr(&mut self, addr: Arc<RwLock<OpOperand>>) {
        let operation = self.operation.try_read().unwrap();
        let mut operands = operation.operands();
        operands.set_operand(1, addr);
    }
}

impl Op for StoreOp {
    fn operation_name() -> OperationName {
        OperationName::new("llvm.store".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        StoreOp { operation }
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn operation(&self) -> &Arc<RwLock<Operation>> {
        &self.operation
    }
    fn display(&self, f: &mut Formatter<'_>, _indent: i32) -> std::fmt::Result {
        let operation = self.operation().read().unwrap();
        write!(f, "{}", operation.name())?;
        write!(f, " {}", operation.operands())?;
        write!(f, " : ")?;
        let value = self.value();
        write!(f, "{}", value.typ().unwrap().try_read().unwrap())?;
        write!(f, ", ")?;
        let addr = self.addr();
        write!(f, "{}", addr.typ().unwrap().try_read().unwrap())?;
        Ok(())
    }
}

impl Parse for StoreOp {
    fn op<T: ParserDispatch>(
        parser: &mut Parser<T>,
        parent: Option<Arc<RwLock<Block>>>,
    ) -> Result<Arc<RwLock<dyn Op>>> {
        let mut operation = Operation::default();
        let name = parser.expect(TokenKind::BareIdentifier)?;
        assert!(name.lexeme == StoreOp::operation_name().to_string());
        operation.set_name(StoreOp::operation_name());
        operation.set_parent(parent.clone());

        let mut operands = vec![];
        let value = parser.parse_op_operand(parent.clone().unwrap(), TOKEN_KIND)?;
        operands.push(value.clone());

        parser.expect(TokenKind::Comma)?;
        let addr = parser.parse_op_operand(parent.clone().unwrap(), TOKEN_KIND)?;
        operands.push(addr.clone());
        operation.set_operands(OpOperands::from_vec(operands));
        parser.expect(TokenKind::Colon)?;
        let value_type = T::parse_type(parser)?;
        parser.verify_type(value, value_type)?;
        parser.expect(TokenKind::Comma)?;
        let addr_type = T::parse_type(parser)?;
        parser.verify_type(addr, addr_type)?;

        let op = StoreOp {
            operation: Arc::new(RwLock::new(operation)),
        };
        Ok(Arc::new(RwLock::new(op)))
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
