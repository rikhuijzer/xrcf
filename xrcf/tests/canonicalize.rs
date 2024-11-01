extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::dialect::arith;
use xrcf::ir;
use xrcf::ir::Op;
use xrcf::parser::DefaultParserDispatch;
use xrcf::parser::Parser;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--canonicalize"]
}

#[test]
fn determine_users() {
    let src = indoc! {"
    func.func @test_determine_users(%arg0 : i64) -> i64 {
        %0 = arith.constant 1 : i64
        %1 = arith.constant 2 : i64
        return %1 : i64
    }
    "};

    let module = Parser::<DefaultParserDispatch>::parse(src).unwrap();
    Tester::verify(module.clone());
    let module = module.try_read().unwrap();

    let ops = module.ops();
    assert_eq!(ops.len(), 1);
    let func_op = ops[0].try_read().unwrap();
    let ops = func_op.ops();
    assert_eq!(ops.len(), 3);

    let op0 = ops[0].try_read().unwrap();
    let op0 = op0.as_any().downcast_ref::<arith::ConstantOp>().unwrap();
    let operation = op0.operation().try_read().unwrap();
    let users = operation.users();
    assert_eq!(users.len(), 0);

    let op1 = ops[1].try_read().unwrap();
    let op1 = op1.as_any().downcast_ref::<arith::ConstantOp>().unwrap();
    let operation = op1.operation().try_read().unwrap();
    let users = operation.users();
    assert_eq!(users.len(), 1);
}

#[test]
fn canonicalize_addi() {
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
        %c3_i64 = arith.constant 3 : i64
        %3 = arith.addi %arg0, %c3_i64 : i64
        return %3 : i64
      }
    }
    "};
    Tester::init_tracing();
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module.clone());
    let module = module.try_read().unwrap();
    Tester::check_lines_exact(&actual, expected, Location::caller());
    assert!(module.as_any().is::<ir::ModuleOp>());
}
