extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-scf-to-cf"]
}

#[test]
fn test_if() {
    Tester::init_tracing();
    let src = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %false = arith.constant false
        %result = scf.if %false -> (i32) {
          %c3_i32 = arith.constant 3 : i32
          scf.yield %c3_i32 : i32
        } else {
          %c4_i32 = arith.constant 4 : i32
          scf.yield %c4_i32 : i32
        }
        return %result : i32
      }
    }
    "#};
    let expected = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %false = arith.constant false
        cf.cond_br %false, ^bb1, ^bb2
      ^bb1:
        %c3_i32 = arith.constant 3 : i32
        cf.br ^bb3(%c3_i32 : i32)
      ^bb2:
        %c4_i32 = arith.constant 4 : i32
        cf.br ^bb3(%c4_i32 : i32)
      ^bb3(%result : i32):
        cf.br ^bb4
      ^bb4:
        return %result : i32
      }
    }
    "#};
    let (module, actual) = Tester::parse(src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_exact(&actual, expected, Location::caller());
}

#[test]
fn test_if_without_yield() {
    Tester::init_tracing();
    let src = indoc! {r#"
    func.func @main() -> i64 {
      %x = arith.constant false
      scf.if %x {
        %0 = arith.constant 0 : i64
      } else {
        %1 = arith.constant 1 : i64
      }
      %2 = arith.constant 0 : i64
      return %2 : i64
    }
    "#};
    let expected = indoc! {r#"
    func.func @main() -> i64 {
      %x = arith.constant false
      cf.cond_br %x, ^bb1, ^bb2
    ^bb1: 
      %0 = arith.constant 0 : i64
      cf.br ^bb3
    ^bb2:
      %1 = arith.constant 1 : i64
      cf.br ^bb3
    ^bb3:
      %2 = arith.constant 0 : i64
      return %2 : i64
    }
    "#};
    let (module, actual) = Tester::parse(src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_contain(&actual, expected, Location::caller());
}
