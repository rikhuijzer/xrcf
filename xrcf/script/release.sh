#!/usr/bin/env bash

#
# Trigger a release
#

set -e -u -o pipefail

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
echo "STEPS:"
echo ""
echo "- UPDATE 'CHANGELOG.md'"
echo ""
echo "- ENSURE YOU ARE ON THE MAIN BRANCH"
echo ""
echo "- PUSH A NEW COMMIT WITH MESSAGE 'Release $VERSION'"
echo ""
echo "- STEP INTO THE 'xrcf' DIR"
echo ""
echo "- RUN 'cargo publish --dry-run'"
echo ""
echo "- RUN 'cargo publish'"
echo ""
echo "- CREATE A NEW TAG, SEE BELOW"
echo ""

NOTES="See [CHANGELOG.md](https://github.com/rikhuijzer/xrcf/blob/main/CHANGELOG.md) for more information about changes since the last release."

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
