extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-mlir-to-llvmir"]
}

#[test]
fn test_if() {
    Tester::init_tracing();
    let src = indoc! {r#"
    func.func @main() -> i32 {
      %false = arith.constant false
      %result = scf.if %false -> (i32) {
        %c1_i32 = arith.constant 3 : i32
        scf.yield %c1_i32 : i32
      } else {
        %c2_i32 = arith.constant 4 : i32
        scf.yield %c2_i32 : i32
      }
      return %result : i32
    }
    "#};
    let expected = indoc! {r#"
    func.func @main() -> i32 {
        %false = arith.constant false
        cf.cond_br %false, ^bb1, ^bb2
    ^bb1:
        %c3_i32 = arith.constant 3 : i32
        cf.br ^bb3(%c3_i32 : i32)
    ^bb2:
        %c4_i32 = arith.constant 4 : i32
        cf.br ^bb3(%c4_i32 : i32)
    ^bb3(%0: i32):
        cf.br ^bb4
    ^bb4:
        return %0 : i32
    }
    "#};
    let (_module, actual) = Tester::parse(src);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module.clone());
    Tester::check_lines_contain(&actual, expected, Location::caller());
}
