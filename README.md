# Conservative Active Subspace Method (CASM)

This repository contains a basic implementation of the conservative active subspace method (CASM), i.e. the active subspace method with conservative surrogates (with high-probability). The goal is to enforce more strictly constraints in optimization problems using the ASM as a dimensionality reduction technique, so that the satifaction of surrogate constraints implies with high-probability the satisfaction of the exact contraints.

The first notebook contains a simple synthetic example to study the feasible regions of the exact problem and of the approximate problem. The second notebook contains an application of the method to a thermal design problem.

This code is related to the paper **Conservative Surrogate Models for Optimization using the Active Subspace Method** (preprint out soon).
