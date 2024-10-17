extern crate xrcf;
mod tester;

use crate::tester::Test;
use indoc::indoc;
use std::panic::Location;
use xrcf::dialect::func::FuncOp;
use xrcf::ir::Op;

#[test]
fn parse_module() {
    let src = indoc! {"
    module {
      func.func @main() -> i64 {
        %0 = arith.constant 2 : i64
        return %0 : i64
      }
    }
  "};
    Test::init_subscriber(tracing::Level::DEBUG);
    let caller = Location::caller();
    Test::parse(src, src, caller);
}

#[test]
fn parse_addi() {
    let src = indoc! {"
    func.func @test_addi(%arg0 : i64) -> i64 {
      %0 = arith.constant 1 : i64
      %1 = arith.constant 2 : i64
      %2 = arith.addi %0, %1 : i64
      %3 = arith.addi %arg0, %2 : i64
      return %3 : i64
    }
    "};
    let expected = indoc! {"
    module {
      func.func @test_addi(%arg0 : i64) -> i64 {
        %0 = arith.constant 1 : i64
        %1 = arith.constant 2 : i64
        %2 = arith.addi %0, %1 : i64
        %3 = arith.addi %arg0, %2 : i64
        return %3 : i64
      }
    }
    "};
    let caller = Location::caller();
    Test::init_subscriber(tracing::Level::DEBUG);
    let module = Test::parse(src, expected, caller);

    let module = module.try_read().unwrap();
    let module_operation = module.operation().try_read().unwrap();
    let module_parent = module_operation.parent();
    assert!(module_parent.is_none());

    let ops = module.ops();
    assert_eq!(ops.len(), 1);
    let func_op = ops[0].try_read().unwrap();
    let func_operation = func_op.operation().try_read().unwrap();
    assert_eq!(func_operation.name(), FuncOp::operation_name());
    assert!(func_operation.parent().is_some());
}
