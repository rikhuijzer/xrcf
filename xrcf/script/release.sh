#!/usr/bin/env bash

#
# Trigger a release
#

set -e

# We have to run this locally because tags created from workflows do not
# trigger new workflows.
# "This prevents you from accidentally creating recursive workflow runs."

echo "CREATING A RELEASE WITH:"

METADATA="$(cargo metadata --format-version=1 --no-deps)"
VERSION="$(echo $METADATA | jq -r '.packages[0].version')"
echo "VERSION $VERSION"
TAGNAME="v$VERSION"
echo "TAGNAME $TAGNAME"

echo ""
echo "ENSURE 'CHANGELOG.md' IS UP-TO-DATE"
echo ""
echo "ENSURE YOU ARE ON THE MAIN BRANCH"
echo ""
echo "ENSURE 'cargo publish --dry-run' SUCCEEDED"
echo ""
echo "ENSURE 'cargo publish' SUCCEEDED"
echo ""

NOTES="\`xrcf-bin\` is a compiler that can compile basic MLIR programs to LLVM IR, and can be used for testing the xrcf package. This binary contains all the default passes such as \`--convert-func-to-llvm\`.

\`arnoldc\` is a compiler that can compile basic ArnoldC programs to LLVM IR. Next to the default passes, this binary contains the pass \`--convert-arnold-to-mlir\` which can lower ArnoldC programs to MLIR. From there, the default passes such as \`--convert-func-to-llvm\` can be used to lower ArnoldC to LLVM IR.

See [CHANGELOG.md](https://github.com/rikhuijzer/xrcf/blob/main/CHANGELOG.md) for more information about changes since the last release."

echo "Ready to create a new tag, which WILL TRIGGER A RELEASE with the following release notes:"
echo "\"$NOTES\""
echo ""
read -p "Are you sure? Type YES to continue. " REPLY

if [[ $REPLY == "YES" ]]; then
    echo ""
    git tag -a $TAGNAME -m "$NOTES"
    git push origin $TAGNAME
    exit 0
else
    echo ""
    echo "Did not receive YES, aborting"
    exit 1
fi
