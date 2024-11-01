//! XRCF is a set of tools to build your own compiler.
//!
//! Below is a a high-level overview of xrcf.
//! To instead see the code in action, see the example Python compiler in the
//! [`pythonc` directory](https://github.com/rikhuijzer/xrcf/tree/main/pythonc).
//!
//! This project is very similar to MLIR, namely it provides tools to build your own compiler.
//! This works as follows:
//!
//! Say you want to write a compiler for a new programming language.
//! The compiler should take the source code in your language and convert it to platform that can execute it such as a CPU or GPU.
//! Now you could do this via some string manipulation.
//! You just read a string like `print("Hello")`, but then you realize that converting this to say LLVM is not that easy.
//! In LLVM, you would need to create an array to hold the string, to then allocate this in memory, and next pass this pointer to a call to `printf`.
//! Doing this via string manipulation is very hard.
//! That's why compilers first scan and parse the source code into some intermediate representation (IR).
//! Unlike strings, the IR provides common methods to interact with the code.
//! For example, this project defines a `insert_before` method for operations.
//! That means that if you want to insert an operation before the `print` call, you can just call `print_op.insert_before(new_op)`.
//!
//! Next you decide that you want to compile this code to another platform such as a GPU.
//! Typically GPUs don't have print operations, but let's assume for now that they do.
//! Then you would need to convert the `print` operation this `gpu.print` operation.
//! But how to manage when to convert the `print` to `printf` and when to `gpu.print`?
//! This is where passes come in.
//! A pass is a group of transformations that are applied to the IR.
//! For example, to compile to CPU via LLVM, you would use the passes `--convert-func-to-llvm` and `--convert-llvm-to-llvm-ir`.
//! And to compile to GPU, you would use the passes `--convert-func-to-gpu`.
//!
//! This project gives you these building blocks.
//! It already includes some default IR and default passes, but you can also add your own.
//! This means that if you want to write your own compiler for your language, you only have to convert your code into the default IR that is inside this project, and then you can choose which passes you want to use in which situation.
//!
//! Note that this is different than LLVM.
//! Where this project allows you to write your own passes, LLVM expects you to generate LLVM IR.
//! This means that you first have to write your own compiler and then hand the generated LLVM IR over to LLVM.
//!
//! In a more big picture, the long-term goal of MLIR is to provide a common IR that developers can lower to, and next there should be various platforms, such as GPUs, CPUs, and TPUs, available.
//!
//! How this project differs from MLIR is that it brings all the benefits of Rust.
//! In MLIR, if one team builds pass to lower to say an AMD GPU, and another team builds passes to lower to NVIDIA GPUs, then the two teams now have separate C++ codebases.
//! Combining the base MLIR codebase with the two other codebases is not easy.
//! In Rust, this would be easier because the backends can be separate crates.
//! In other words, Rust provides much better tooling around managing dependencies.
//!
//! Another benefit of this project is that the code is easier to read than MLIR.
//! For example, this project does not use code generators in the codebase like TableGen.
//! TableGen is great for generating code, but it also gets in the way of much of the tooling.
//! For example, finding where a certain operation is defined is hard because TableGen generates code that is not always visible in the IDE.
//!
//! Finally, another benefit of this project is that Rust provides much better tooling for writing tests.
//! MLIR tests work via the `lit` tool, which is a binary build by the LLVM project to test their compiler.
//! `lit` is executed on `lit` files and compare the code before and after transformations.
//! This makes sense in C++ because there isn't much tooling around testing.
//! But in Rust, the default testing framework can be used.
//! With this, it is possible to not only compare the code before and after transformations, but also to easily access the IR and verify the data structures more explicitly.

mod canonicalize;
pub mod convert;
pub mod dialect;
pub mod ir;
pub mod parser;
pub mod targ3t;
#[cfg(feature = "test-utils")]
pub mod tester;
mod transform;

pub use transform::default_passes;
pub use transform::init_subscriber;
pub use transform::transform;
pub use transform::DefaultTransformDispatch;
pub use transform::Passes;
pub use transform::SinglePass;
pub use transform::TransformDispatch;

/// Dialects can define new operations, attributes, and types.
/// Each dialect is given an unique namespace that is prefixed.
///
/// Dialects can co-exist and can be produced and consumed by different passes.
pub trait Dialect {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
}
