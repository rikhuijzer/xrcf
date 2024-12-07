# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2024-12-07

### Added

- Allow lowering of `scf.if` to `cf` ([#25](https://github.com/rikhuijzer/xrcf/pull/25)).
- Support `scf.if` without `yield` ([#28](https://github.com/rikhuijzer/xrcf/pull/28)).
- Add missing passes to the ArnoldC compiler ([#29](https://github.com/rikhuijzer/xrcf/pull/29)).
- Add `--print-ir-before-all` command line argument ([#31](https://github.com/rikhuijzer/xrcf/pull/31)).

### Changed

- Multiple changes to IR logic ([#25](https://github.com/rikhuijzer/xrcf/pull/25)).
- Make `Tester` more strict ([#27](https://github.com/rikhuijzer/xrcf/pull/27)).

## [0.4.0] - 2024-12-02

### Changed

- Changed many core methods to allow lowering of if-else from LLVM dialect to LLVMIR ([#23](https://github.com/rikhuijzer/xrcf/pull/23)).
