extern crate rrcf;

use rrcf::canonicalize;
use rrcf::parser::BuiltinParse;
use rrcf::Parser;

#[test]
fn canonicalize_addi() {
    let src = "
    func.func @test_addi(%arg0 : i64) -> i64 {
        %0 = arith.constant 1 : i64
        %1 = arith.constant 2 : i64
        %2 = arith.addi %0, %1 : i64
        %3 = arith.addi %arg0, %2 : i64
        return %3 : i64
    }
    ";
    let mut module = Parser::<BuiltinParse>::parse(src).unwrap();
    canonicalize(&mut module);
    println!("\n{}\n", module);
    let repr = format!("{}", module);
    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "module {");
    assert_eq!(lines[1], "  func.func @test_addi(%arg0 : i64) -> i64 {");
    assert_eq!(lines[2], "    %c3_i64 = arith.constant 3 : i64");
    assert_eq!(lines[3], "    %0 = arith.addi %arg0, %c3_i64 : i64");
    assert_eq!(lines[4], "    return %0 : i64");
    assert_eq!(lines[5], "  }");
    assert_eq!(lines[6], "}");
}
