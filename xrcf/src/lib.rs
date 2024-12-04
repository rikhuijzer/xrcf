//! XRCF is a set of tools to build your own compiler.
//!
//! Below is a a high-level overview of xrcf.
//! To instead see the code in action, see the example compiler in the
//! [`arnoldc` directory](https://github.com/rikhuijzer/xrcf/tree/main/arnoldc).
//!
//! What follows is some background on compilers and how this project can help you:
//!
//! Say you want to write a compiler for a new programming language.
//! The compiler should take the source code in your language and convert it to platform that can execute it such as a CPU or GPU.
//! Now you could do this via string manipulation.
//! To sum two matrices, you read a string like `x = a + b`, but then you realize that converting this to say LLVM via string manipulation is not that easy.
//! That's why compilers first scan and parse the source code into an intermediate representation (IR).
//! Unlike strings, the IR provides common methods to interact with the code.
//! For example, this project defines a `insert_before` method for operations.
//! Whereas with strings you would need to manually search for the line where to insert the new operation, with the IR you can just call `add_op.insert_before(new_op)`.
//!
//! Next you decide that you want to compile this code to another platform such as a GPU.
//! Then you would need to convert some of the operations to GPU-specific operations.
//! This is where passes come in.
//! A pass is a group of transformations that are applied to the IR.
//! For example, to compile to CPU via LLVM, you would use the passes `--convert-func-to-llvm` and `--convert-llvm-to-llvm-ir`.
//! And to compile to GPU, you would use `--convert-func-to-gpu`.
//!
//! This project gives you these building blocks.
//! It contains some default IR and default passes, but more importantly you can also add your own.
//! This means that if you want to write your own compiler for your language, you only have to convert your code into the default IR that is inside this project, and then you can choose which passes you want to use in which situation.
//!
//! ## Long-term Goal
//!
//! It is unclear where computations will be done in the future.
//! Will it be on CPUs, GPUs, TPUs, or something else?
//! But this is not what this project should focus on.
//! This project focuses on what does not change: transformations.
//! It's very likely that we still need to transform code in the future.
//! There will probably always be a gap between code that is easy to read for humans and code that can be efficiently executed by the hardware.
//!
//! So, the long-term goal of this project is to provide an easy-to-use set of tools that can be used to build your own compiler.
//! In other words, it should be easy to build a compiler that can transform your favorite language to this project's core IR, and then it should be easy to transform this to various platforms such as GPUs, CPUs, and TPUs.

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
