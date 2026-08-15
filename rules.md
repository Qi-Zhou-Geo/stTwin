```sh
# Last modified: 2026-08-15T12:55:12
# Author: Qi Zhou
```

## **Readme**
Working together efficiently and happily 🤝
---

### Please follow these rules:
These steps may increase your workload initially, <br>
but they will make your workflow more *traceable*, *reproducible*, and *easier to share* in the long run.

1. Code docstrings <br>
Reading and understanding code without clear docsting can be extremely painful. <br>
Please add docsting once you are happy with your model/function/class. <br>
You may refer to the
[Google-style Python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
format.

1. Variable naming <br>
Same for the variable. You may refer to the
[Snakecase](https://curc.readthedocs.io/en/latest/programming/coding-best-practices.html) style.


1. Commit changes <br>
Use a single squashed commit with a version number in the format v_major.minor.patch <br>
Example: **v0.1.3** <br>
v0.1 is the latest tagged release already published on Zenodo. <br>
The patch number (.3) indicates the third set of changes made on top of tag v0.1. <br>


1. Push changes <br>
Resolve any merge conflicts. <br>
Ensure the branch is clean and your commit history is squashed into one commit.<br>


1. Update Env. Dependencies <br>
Whenever you install a new package, update both the package name and its version in [environment.yml](config/environment.yml).
