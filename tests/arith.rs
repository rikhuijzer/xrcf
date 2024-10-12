extern crate rrcf;

use rrcf::dialect::func::FuncOp;
use rrcf::ir::Op;
use rrcf::parser::BuiltinParse;
use rrcf::Parser;

#[test]
fn parse_addi() {
    use indoc::indoc;

    let src = indoc! {"
    func.func @test_addi(%arg0 : i64) -> i64 {
        %0 = arith.constant 1 : i64
        %1 = arith.constant 2 : i64
        %2 = arith.addi %0, %1 : i64
        %3 = arith.addi %arg0, %2 : i64
        return %3 : i64
    }
    "};
    println!("-- Before:\n{}", src);
    let module = Parser::<BuiltinParse>::parse(src).unwrap();
    println!("-- After:\n{}\n", module);
    let repr = format!("{}", module);

    let module_operation = module.operation().try_read().unwrap();
    let module_parent = module_operation.parent();
    assert!(module_parent.is_none());

    let ops = module.ops();
    assert_eq!(ops.len(), 1);
    let func_op = ops[0].try_read().unwrap();
    let func_operation = func_op.operation().try_read().unwrap();
    assert_eq!(func_operation.name(), FuncOp::operation_name());
    assert!(func_operation.parent().is_some());

    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "module {");
    assert_eq!(lines[1], "  func.func @test_addi(%arg0 : i64) -> i64 {");
    assert_eq!(lines[2], "    %0 = arith.constant 1 : i64");
    assert_eq!(lines[3], "    %1 = arith.constant 2 : i64");
    assert_eq!(lines[4], "    %2 = arith.addi %0, %1 : i64");
    assert_eq!(lines[5], "    %3 = arith.addi %arg0, %2 : i64");
    assert_eq!(lines[6], "    return %3 : i64");
    assert_eq!(lines[7], "  }");
    assert_eq!(lines[8], "}");
}
