extern crate rrcf;

use rrcf::parser::BuiltinParse;
use rrcf::Parser;

#[test]
fn parse_addi() {
    let src = "
    func.func @test_addi(%arg0 : i64) -> i64 {
        %0 = arith.constant 1 : i64
        %1 = arith.addi %arg0, %0 : i64
        return %1 : i64
    }
    ";
    let module = Parser::<BuiltinParse>::parse(src).unwrap();
    let op = module.first_op().unwrap();
    let repr = format!("{}", op);
    println!("repr:\n{}\n", repr);
    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "func.func @test_addi(%arg0 : i64) -> i64 {");
    assert_eq!(lines[1], "  %0 = arith.constant 1 : i64");
    assert_eq!(lines[2], "  %1 = arith.addi %arg0, %0 : i64");
    assert_eq!(lines[2], "  return %0 : i64");
    assert_eq!(lines[3], "}");
}
