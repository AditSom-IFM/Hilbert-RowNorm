#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 REPO EXPECTED_COMMIT" >&2
  exit 2
fi

repo=$1
expected_commit=$2

if [[ ${repo} != /* ]]; then
  echo "launch checkout must be an absolute path: ${repo}" >&2
  exit 2
fi
if [[ ! ${expected_commit} =~ ^[0-9a-f]{40}$ ]]; then
  echo "EXPECTED_COMMIT must be a full lowercase Git SHA-1: ${expected_commit}" >&2
  exit 2
fi
if [[ ! -d ${repo} ]]; then
  echo "launch checkout does not exist: ${repo}" >&2
  exit 1
fi

canonical_repo=$(cd "${repo}" && pwd -P)
if [[ ${canonical_repo} != "${repo}" ]]; then
  echo "launch checkout must be a canonical absolute path: ${repo}" >&2
  exit 2
fi
repo_root=$(git -C "${repo}" rev-parse --show-toplevel)
if [[ ${repo_root} != "${repo}" ]]; then
  echo "launch checkout is not the Git worktree root: ${repo}" >&2
  exit 1
fi
actual_commit=$(git -C "${repo}" rev-parse HEAD)
if [[ ${actual_commit} != "${expected_commit}" ]]; then
  echo "expected commit ${expected_commit}; found ${actual_commit}" >&2
  exit 1
fi
if [[ -n $(git -C "${repo}" status --porcelain --untracked-files=all) ]]; then
  echo "launch checkout is not clean: ${repo}" >&2
  exit 1
fi
if [[ ! -x ${repo}/scripts/run.sh || ! -d ${repo}/src/hilbert_rownorm ]]; then
  echo "launch checkout is incomplete: ${repo}" >&2
  exit 1
fi

printf '%s\n' "${actual_commit}"
