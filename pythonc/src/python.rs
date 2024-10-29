use anyhow::Result;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;
use xrcf::ir::Attribute;
use xrcf::ir::Block;
use xrcf::ir::IntegerAttr;
use xrcf::ir::Op;
use xrcf::ir::Operation;
use xrcf::ir::OperationName;
use xrcf::ir::PlaceholderType;
use xrcf::parser::Parse;
use xrcf::parser::Parser;
use xrcf::parser::ParserDispatch;

pub struct FuncOp {
    operation: Arc<RwLock<Operation>>,
}

impl FuncOp {}

impl Op for FuncOp {
    fn operation_name() -> OperationName {
        OperationName::new("def".to_string())
    }
    fn new(operation: Arc<RwLock<Operation>>) -> Self {
        FuncOp { operation }
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
        let operation = self.operation.try_read().unwrap();
        operation.display_results(f)?;
        write!(f, "{}", operation.name())?;
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
        let results = parser.parse_op_results_into(&mut operation)?;

        parser.parse_operation_name_into::<FuncOp>(&mut operation)?;

        let integer = parser.parse_integer()?;
        let typ = integer
            .as_any()
            .downcast_ref::<IntegerAttr>()
            .unwrap()
            .typ();
        let hack = typ.to_string();
        let typ = PlaceholderType::new(&hack);
        let typ = Arc::new(RwLock::new(typ));
        operation.set_result_type(typ)?;

        let attributes = operation.attributes();
        attributes.insert("value", Arc::new(integer));
        let operation = Arc::new(RwLock::new(operation));
        let op = FuncOp {
            operation: operation.clone(),
        };
        let op = Arc::new(RwLock::new(op));
        results.set_defining_op(op.clone());
        Ok(op)
    }
}
