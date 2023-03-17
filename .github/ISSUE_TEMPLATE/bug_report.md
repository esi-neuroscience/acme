---
name: Report a Bug
about: Ran into a problem? Please let us know!
title: ''
labels: 'bug'
assignees: ''
---

Thank you for taking the time to report an issue! We know that writing
detailed error reports takes effort, and we appreciate you taking the time.

**Describe the problem**
Please provide a short description of the problem.

**Steps To Reproduce**
If possible, please provide a short stub of the function you are wrapping
with `ParallelMap`, e.g.,

```python
def myfunc(x, y, z):
    ...
    return something
```

Please re-run the erroneous `ParallelMap` call with full verbosity and
ACME's output recorded in a logfile, i.e.,

```python
with ParallelMap(myfunc,...,verbose=True, logfile=True) as pmap:
    pmap.compute()
```

Please attach the generated logfile to this issue.

**System Profile:**
Please tell us a little bit about your computational working environment:

- Are you using ACME locally or on HPC cluster hardware?
  If your code is running on a cluster:
  - which scheduling manager are you using (SLURM, PBS, etc.)?
  - what architecture/OS version is your code running on (x86, ARM, etc.)?
    On Linux, please simply paste the output of the command `uname -a` here:

**Additional Information**
Anything else you feel might be important.

**Thank you again for your time!**
