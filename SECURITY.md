# Security policy

## Supported versions

Fixes go into the latest release on PyPI. Please check that a problem is present there
before reporting it.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/dundysm/gaitkeeper/security/advisories/new),
or email dundysm@gmail.com. You will get an answer within a week.

Things that are in scope:

* Code execution or file writes outside the intended directories when gaitkeeper reads a
  config, an ONNX file, a trace or an adapter it was pointed at.
* `gaitkeeper fetch` downloading or trusting something other than the pinned files.

gaitkeeper compiles and runs teleop-walking-benchmark adapters (`policy.cpp`) on purpose,
and evaluates legged_gym config classes from source without importing them. Only point it at
code you trust; running an untrusted adapter is not a vulnerability in gaitkeeper.
