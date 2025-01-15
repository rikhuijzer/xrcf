extern crate xrcf;

use indoc::indoc;
use std::panic::Location;
use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;
use xrcf::tester::DefaultTester;

fn flags() -> Vec<&'static str> {
    vec!["--tester-no-print", "--convert-func-to-llvm"]
}

fn benchmark_rewrite() {
    DefaultTester::init_tracing();
    let src = indoc! {"
    func.func @add_one(%arg0 : i32) -> i32 {
        %0 = arith.constant 1 : i32
        %1 = arith.addi %0, %arg0 : i32
        return %1 : i32
    }
    "};
    let src = &src.repeat(100);
    let expected = indoc! {"
    llvm.func @add_one(%arg0 : i32) -> i32 {
        %0 = llvm.mlir.constant(1 : i32) : i32
        %1 = llvm.add %0, %arg0 : i32
        llvm.return %1 : i32
    }
    "};
    let (module, actual) = DefaultTester::transform(flags(), src);
    DefaultTester::verify(module);
    DefaultTester::check_lines_contain(&actual, expected, Location::caller());
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("rewrite");
    group.sample_size(10);
    group.bench_function("benchmark_rewrite", |b| b.iter(benchmark_rewrite));
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
