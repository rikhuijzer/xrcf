# RRCF Code Style Guide

Some notes on the code style used in this project.

## Verbosity vs. Complexity

In case of doubt, prefer verbosity over complexity.
For example, prefer some duplication over things like macros, declarative code, or DSLs.
The main aim is to keep things easy to understand; for both humans as well as tooling.

On declarative code.
In my experience, declarative code is a beautiful idea, but in practice it often is hard to learn and understand.
There is now even data to back this up: in benchmarks, LLMs score higher on imperative code than declarative code.