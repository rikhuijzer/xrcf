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
    println!("\n{}\n", module);
    let repr = format!("{}", module);
    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "module {");
    assert_eq!(lines[1], "  func.func @test_addi(%arg0 : i64) -> i64 {");
    assert_eq!(lines[2], "    %0 = arith.constant 1 : i64");
    assert_eq!(lines[3], "    %1 = arith.addi %arg0, %0 : i64");
    assert_eq!(lines[4], "    return %0 : i64");
    assert_eq!(lines[5], "  }");
    assert_eq!(lines[6], "}");
}
