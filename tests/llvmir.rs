#![allow(dead_code)]
#![allow(unused)]
extern crate rrcf;

use rrcf::opt;
use rrcf::parser::BuiltinParse;
use rrcf::OptOptions;
use rrcf::Parser;

#[test]
fn test_translate() {
    let mlir = "llvm.mlir.global internal @i32_global(42: i32) : i32";
    let llvmir = "i32_global = internal global i32 42";

    // assert_eq!(translate(mlir), llvmir);
}

// #[test]
fn test_constant_func() {
    use indoc::indoc;

    let src = indoc! {"
      func.func @test_ret_1() -> i64 {
        %0 = arith.constant 1 : i64
        return %0 : i64
      }
    "};

    let mut module = Parser::<BuiltinParse>::parse(src).unwrap();
    println!("\n-- Before convert-func-to-llvm:\n{src}");
    // mlir-opt --convert-func-to-llvm tmp.mlir
    let mut options = OptOptions::default();
    options.set_convert_func_to_llvm(true);
    opt(&mut module, options).unwrap();
    println!("\n-- After convert-func-to-llvm:\n{module}\n");
    let repr = format!("{}", module);
    let lines = repr.lines().collect::<Vec<&str>>();
    assert_eq!(lines[0], "module {");
    assert_eq!(lines[1], "  llvm.func @test_ret_1() -> i64 {");
    assert_eq!(lines[2], "    llvm.mlir.constant(1 : i64) : i64");
    assert_eq!(lines[3], "    llvm.return %0 : i64");
    assert_eq!(lines[4], "  }");
    assert_eq!(lines[5], "}");
}
