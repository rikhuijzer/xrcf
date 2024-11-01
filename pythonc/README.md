# pythonc

This directory contains an example Python compiler that can compile a small subset of Python code to LLVM IR, and can be executed via `lli`.
Please do not expect this to be a fully featured Python compiler.
It is a proof of concept that shows how to build your own compiler using the `xrcf` crate.

To run the compiler, we install the `pythonc` binary via

```sh
$ cargo install --path pythonc
```

This places the `pythonc` binary in `$HOME/.cargo/bin`.
(It can be uninstalled later with `cargo uninstall pythonc`.)

Next, let's see whether the installation was successful:

```sh
$ pythonc --help
```

This should print:

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
You can also decide to use 

To compile Python, let's create a file called `tmp.py` with the following content:

```python
def hello():
    print("Hello, World!")

hello()
```

Before we run this, let's see what the compiler does with the `--convert-python-to-mlir` pass:

```sh
$ pythonc --convert-python-to-mlir tmp.py
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
$ pythonc --convert-python-to-mlir --convert-unstable-to-mlir --convert-func-to-llvm --convert-mlir-to-llvmir tmp.py
```

This prints:

```llvm
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i32 @printf(ptr)

define i32 @main() {
    %2 = alloca i8, i64 14, align 1
    store [14 x i8] c"hello, world\0A\00", ptr %2, align 1
    %3 = call i32 @printf(ptr %2)
    ret i32 0
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

Remembering these passes and in the order in which to run them is cumbersome, so let's use the `--compile` flag, which is a wrapper around the above command:

```sh
$ pythonc --compile tmp.py
```

It returns the same LLVM IR as above.

To run our compiled code, we can use the `lli` command.
`lli` executes programs written in the LLVM bitcode format.
This executable is part of the LLVM project, so it can usually be installed via the package manager.
For example, on MacOS, `brew install llvm`.

So let's run our compiled code:

```sh
$ pythonc --compile tmp.py | lli
```

This should print:

```text
Hello, World!
```

To learn how to build your own compiler like this, see the files inside this `pythonc` directory.
It is split into three parts:

1. `src/main.rs` contains the command line interface of the compiler.
1. `src/python.rs` specifies how to parse the Python code (convert the text to data structures).
1. `src/python_to_mlir.rs` contains the `--convert-python-to-mlir` pass, which converts the Python code to MLIR.

All other passes such as `--convert-func-to-llvm` are implemented in the `xrcf` crate.

To get inspiration for your own compiler, the following projects are built on MLIR:

- [jax](https://github.com/jax-ml/jax): A Python library for accelerator-oriented computing
- [triton](https://github.com/triton-lang/triton): A Python library for high-performance computation on GPUs by OpenAI.
- [torch-mlir](https://github.com/llvm/torch-mlir): Compiles PyTorch to MLIR.
- [mlir-hlo](https://github.com/llvm/mlir-hlo): A set of transformations from TensorFlow HLO to MLIR.
- [circt](https://github.com/llvm/circt): A compiler for hardware design.
- [mojo](https://www.modular.com/mojo): A new programming language for AI by Modular.