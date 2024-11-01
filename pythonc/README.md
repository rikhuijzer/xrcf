# pythonc

This directory contains an example Python compiler that can compile a small subset of Python code to LLVM IR, and can be executed via `lli`.

To build the compiler, we use `cargo run`.
To do so, we can step inside this directory and run the following:

```sh
$ cargo run -- --help
```

Which prints:

```text
An example Python compiler that can compile a small subset of Python

Usage: pythonc [OPTIONS] [INPUT]

Arguments:
  [INPUT]  The input file (defaults to "-", which is stdin) [default: -]

Options:
      --convert-unstable-to-mlir  Convert unstable operations to MLIR
      --convert-func-to-llvm      Convert function operations to LLVM IR
      --convert-mlir-to-llvmir    Convert MLIR to LLVM IR
      --convert-python-to-mlir    Convert Python operations to MLIR
      --compile                   Compile the code
  -h, --help                      Print help
  -V, --version                   Print version
```

If this succeeds, then that shows that the compiler is correctly built.

To compile Python, let's create a file called `tmp.py` with the following content:

```python
def hello():
    print("Hello, World!")

hello()
```

Before we run this, let's see what the compiler does with the `--convert-python-to-mlir` pass:

```sh
$ cargo run -- --convert-python-to-mlir tmp.py
```

This prints:

```mlir
module {
  func.func @hello() {
    unstable.printf("Hello, World!")
    return
  }
  func.func @main() -> i32 {
    %0 = arith.constant 0 : i32
    func.call @hello() : () -> ()
    return %0 : i32
  }
}
```

What this shows is that the compiler has wrapped the `hello` call into a `main` function.
This way, once we lowered the code to LLVM, LLVM will execute `main` and return a 0 status code if the code didn't crash.
This is similar to Python, which by default also will return a 0 status code.

To convert our code to LLVM IR, let's run all the required passes in order:

```sh
$ cargo run -- --convert-python-to-mlir --convert-unstable-to-mlir --convert-func-to-llvm --convert-mlir-to-llvmir tmp.py
```

This prints:

```llvm
foo
```

Remembering these passes and in which order is cumbersome, so let's use the `--compile` flag.
This is just a wrapper around the above command:

```sh
$ cargo run -- --compile tmp.py
```

So it returns the same as the above command.

To run this, we can use the `lli` command.
`lli` executes programs written in the LLVM bitcode format.
This executable is part of the LLVM project, so it can usually be installed via the package manager.
For example, on MacOS, `brew install llvm`.

So let's run our compiled code:

```sh
$ cargo run -- --compile tmp.py | lli
```

This should print:

```text
Hello, World!
```
