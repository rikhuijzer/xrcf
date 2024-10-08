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
    println!("\nBefore canonicalize:{src}");
    canonicalize(&mut module);
    println!("\nAfter canonicalize:\n{module}\n");
    let repr = format!("{}", module);
    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "module {");
    assert_eq!(lines[1], "  func.func @test_addi(%arg0 : i64) -> i64 {");
    assert_eq!(lines[2], "    %0 = arith.constant 1 : i64");
    assert_eq!(lines[3], "    %1 = arith.constant 2 : i64");
    assert_eq!(lines[4], "    %2 = arith.constant 3 : i64");
    assert_eq!(lines[5], "    %3 = arith.addi %arg0, %2 : i64");
    assert_eq!(lines[6], "    return %3 : i64");
    assert_eq!(lines[7], "  }");
    assert_eq!(lines[8], "}");
}
