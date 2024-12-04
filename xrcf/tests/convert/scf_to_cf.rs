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
      ^bb3(%0: i32):
        cf.br ^bb4
      ^bb4:
        return %0 : i32
      }
    }
    "#};
    // let (module, actual) = Tester::parse(src);
    // Tester::verify(module);
    // Tester::check_lines_contain(&actual, &src, Location::caller());
    let (module, actual) = Tester::transform(flags(), src);
    Tester::verify(module);
    Tester::check_lines_exact(&actual, expected, Location::caller());
}
