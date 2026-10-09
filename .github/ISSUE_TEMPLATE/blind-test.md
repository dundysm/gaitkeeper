---
name: Blind test submission
about: Seal the labels for harness logs you want gaitkeeper scored on, before sending them
title: "Blind test: <your harness>"
labels: blind-test
---

Protocol: [docs/BLIND_TEST.md](../../docs/BLIND_TEST.md). Do not attach `labels.json` here
until the outputs hash has been posted.

**Labels commitment** (from `python tools/blind.py seal labels.json`):

```
labels sha256 
```

**Harness:** <name, simulator and version, physics step, policy and robot>

**Number of logs:** <n>, of which <k> you believe are clean

**How you will send the folder:** <link to a release, a dataset, or "please contact me">
