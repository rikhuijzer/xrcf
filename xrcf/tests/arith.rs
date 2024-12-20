extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::dialect::func::FuncOp;
use xrcf::ir::Op;
use xrcf::shared::SharedExt;
use xrcf::tester::Tester;

#[test]
fn parse_module() {
    Tester::init_tracing();
    let src = indoc! {"
    module {
      func.func @main() -> i64 {
        %0 = arith.constant 2 : i64
        return %0 : i64
      }
    }
    "};
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, &src, Location::caller());
}

#[test]
fn parse_addi() {
    Tester::init_tracing();
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
    let (module, actual) = Tester::parse(src);
    Tester::verify(module.clone());

    let module = module.rd();
    let module_operation = module.operation().rd();
    let module_parent = module_operation.parent();
    assert!(module_parent.is_none());

    let ops = module.ops();
    assert_eq!(ops.len(), 1);
    let func_op = ops[0].rd();
    let func_operation = func_op.operation().rd();
    assert_eq!(func_operation.name(), FuncOp::operation_name());
    let func_parent = func_operation.parent();
    assert!(func_parent.is_some());

    let func_op = ops[0].clone();
    let func_op = func_op.rd();
    let func_parent = func_op.parent_op();
    assert!(func_parent.is_some());
    let func_parent = func_parent.unwrap();
    let func_parent = func_parent.rd();
    assert_eq!(func_parent.name().to_string(), "module");

    Tester::check_lines_contain(&actual, expected, caller);
}
