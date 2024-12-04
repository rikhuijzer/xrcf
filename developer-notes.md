# Developer Notes

## Verbosity vs. Complexity

In case of doubt, prefer verbosity over complexity.
For example, prefer some duplication over things like macros, declarative code, or DSLs.
The main aim is to keep things easy to understand; for both humans as well as tooling.
I prefer "boilerplate" code over complex things like macros.
Especially with LLMs built into the editor, it's still reasonably easy to refactor boilerplate code.
This can't be said for macros; it's usually very hard to refactor them.

For example, in my experience, declarative code is a beautiful idea, but in practice it's often hard to learn and understand.

## `Arc<RwLock<T>>`

This project uses `Arc<RwLock<T>>` for pretty much everything.
The reason is that the intermediate representation (IR) is essentially a tree with nodes.
This tree is traversed in multiple directions (up, down, and sideways), so we need to store pointers all over the place.
Also, this project uses `Arc` instead of `Rc` because at some point the compiler will be multi-threaded.

To simplify the usage, there are some helpers functions available via `GuardedOp`, `GuardedBlock`, etc.

In some cases, this can simplify code such as
```rust
let x = x.read().unwrap();
let parent = x.parent();
```
to
```rust
let x = x.parent();
```

## LLVM's codebase

This project does not include any LLVM code.
The LLVM codebase is great, but also big.
Including the C files would require lots of compilation.
Calling out to an installed LLVM version is better but is hard with package versions and such.
Let's for now just generate LLVM IR, then the LLVM binary can be used to compile that.
At a later point, the plan is to include the LLVM backend in a separate crate.

## Comments

Comments should explain *why* something is done, not *what* something does.
The problem with "what" comments is that they often get out of date and then become confusing.
If the "what" is not obvious from the code, then try to first explain it via descriptive variable and function names.

## Op Docstrings

Explaining *what* an op does is a good idea for docstrings.
For example, for the `scf.yield` op, the docstring could be:

    `scf.yield`
 
    ```ebnf
    `scf.yield` $operands `:` type($operands)
    ```
    For example,
    ```mlir
    scf.yield %0 : i32
    ```

This uses the Extended Backus-Naur Form (EBNF) to describe the syntax of the op.

The benefit of using EBNF and the example is that it can be convenient to find this information in the documentation.
Even more importantly, hopefully LLMs will learn from these examples how the parsing and printing code should look like.