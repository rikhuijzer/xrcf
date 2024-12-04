extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use xrcf::tester::Tester;

fn flags() -> Vec<&'static str> {
    vec!["--convert-cf-to-llvm"]
}

#[test]
fn test_if_else() {
    Tester::init_tracing();
    let src = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %false = arith.constant false
        cf.cond_br %false, ^then, ^else
      ^then:
        %c3_i32 = arith.constant 3 : i32
        cf.br ^merge(%c3_i32 : i32)
      ^else:
        %c4_i32 = arith.constant 4 : i32
        cf.br ^merge(%c4_i32 : i32)
      ^merge(%result : i32):
        cf.br ^exit
      ^exit:
        return %result : i32
      }
    }
    "#};
    let expected = indoc! {r#"
    module {
      func.func @main() -> i32 {
        %false = arith.constant false
        llvm.cond_br %false, ^then, ^else
      ^then:
        %c3_i32 = arith.constant 3 : i32
        llvm.br ^merge(%c3_i32 : i32)
      ^else:
        %c4_i32 = arith.constant 4 : i32
        llvm.br ^merge(%c4_i32 : i32)
      ^merge(%result : i32):
        llvm.br ^exit
      ^exit:
        return %result : i32
      }
    }
    "#};
    let (module, actual) = Tester::parse(src);
    Tester::check_lines_exact(&actual, &src, Location::caller());
    Tester::verify(module);
    let (module, actual) = Tester::transform(flags(), src);
    Tester::check_lines_exact(&actual, expected, Location::caller());
    Tester::verify(module);
}
