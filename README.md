# xrcf

Tools to build your own compiler.

This project is very similar to MLIR.
For people unfamiliar with MLIR, here is a high-level overview:

Say you want to write a compiler for a new programming language.
The compiler should take the source code in your language and convert it to platform that can execute it such as a CPU or GPU.
Now you could do this via some string manipulation.
You just read a string like `print("Hello")`, but then you realize that converting this to say LLVM is not that easy.
In LLVM, you would need to create an array to hold the string, to then allocate this in memory, and next pass this pointer to a call to `printf`.
Doing this via string manipulation is very hard.
That's why compilers first scan and parse the source code into some intermediate representation (IR).
Unlike strings, the IR provides common methods to interact with the code.
For example, this project defines a `insert_before` method for operations.
That means that if you want to insert an operation before the `print` call, you can just call `print_op.insert_before(new_op)`.

Next you decide that you want to compile this code to another platform such as a GPU.
Typically GPUs don't have print operations, but let's assume for now that they do.
Then you would need to convert the `print` operation this `gpu.print` operation.
But how to manage when to convert the `print` to `printf` and when to `gpu.print`?
This is where passes come in.
A pass is a group of transformations that are applied to the IR.
For example, to compile to CPU via LLVM, you would use the passes `--convert-func-to-llvm` and `--convert-llvm-to-llvm-ir`.
And to compile to GPU, you would use the passes `--convert-func-to-gpu`.

This project gives you these building blocks.
It already includes some default IR and default passes, but you can also add your own.
This means that if you want to write your own compiler for your language, you only have to convert your code into the default IR that is inside this project, and then you can choose which passes you want to use in which situation.

Note that this is different than LLVM.
Where this project allows you to write your own passes, LLVM expects you to generate LLVM IR.
This means that you first have to write your own compiler and then hand the generated LLVM IR over to LLVM.

In a more big picture, the long-term goal of MLIR is to provide a common IR that developers can lower to, and next there should be various platforms, such as GPUs, CPUs, and TPUs, available.

How this project differs from MLIR is that it brings all the benefits of Rust.
In MLIR, if one team builds pass to lower to say an AMD GPU, and another team builds passes to lower to NVIDIA GPUs, then the two teams now have separate C++ codebases.
Combining the base MLIR codebase with the two other codebases is not easy.
In Rust, this would be easier because the backends can be separate crates.
In other words, Rust provides much better tooling around managing dependencies.

Another benefit of this project is that the code is easier to read than MLIR.
For example, this project does not use code generators in the codebase like TableGen.
TableGen is great for generating code, but it also gets in the way of much of the tooling.
For example, finding where a certain operation is defined is hard because TableGen generates code that is not always visible in the IDE.

Finally, another benefit of this project is that Rust provides much better tooling for writing tests.
MLIR tests work via the `lit` tool, which is a binary build by the LLVM project to test their compiler.
`lit` is executed on `lit` files and compare the code before and after transformations.
This makes sense in C++ because there isn't much tooling around testing.
But in Rust, the default testing framework can be used.
With this, it is possible to not only compare the code before and after transformations, but also to easily access the IR and verify the data structures more explicitly.

## Notes

### LLVM's codebase

This project does not include any LLVM code.
The LLVM codebase is great, but also big.
Including the C files would require lots of compilation.
Calling out to an installed LLVM version is better but is hard with package versions and such.
Let's for now just generate LLVM IR, then the LLVM binary can be used to compile that.
At a later point, the plan is to include the LLVM backend in a separate crate.

### `Operation`

MLIR has a `Op` trait where each `struct` that implements it contains a `Operation` field.
This means that `Operation` is very generic and the various `Op` implementations
all access the real data through the `Operation` field.

A downside of the `Operation` field is that it may contain fields that are not necessary.
For example, `arith.constant` does not take any operands,
but the `Operation` field will still contain an empty `operands` field.

The benefit is that transformations do not require the fields to be copied.
They can just call it a different type while pointing to the same `Operation` struct.
This is probably why MLIR uses it.
Otherwise, transforming all ops from one dialect to another would require copying all op fields.

But wait, it's not that much copying.
Many fields are just pointers to other data.
Instead focus on the fact that any `Op` transformation takes ownership of the data.
Then, it's will be harder to mess up the underlying data.
There will be less state to keep track of.

In summary: Do not prematurely optimize!
