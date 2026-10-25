# Sampling and surrogate optimization

## What these methods do

Sampling plans choose a set of parameter combinations to evaluate. Full factorial, random, stratified, and Latin-hypercube-style designs are different ways to spread those evaluations through a search space. A surrogate is a cheaper model fitted to the expensive samples; a Gaussian process can also estimate uncertainty about its predictions.

## Everyday analogy

Imagine tasting a few recipes and learning a rough map of how ingredient amounts affect flavor, instead of cooking every possible recipe. Bayesian optimization uses that map to choose the next recipe that may improve the result or reduce uncertainty.

## Strengths and limits

Sampling and surrogates help when each true evaluation is expensive and the parameter space is small enough to explore carefully. A surrogate can be inaccurate between samples, and Bayesian optimization can spend evaluations in the wrong region if its model is poorly calibrated.

## RouteArena status

This is a planned Unit 3 feature and is not implemented in the current app. The future objective is to tune routing-solver settings using average tour cost over several seeds, compare prediction error on held-out samples, and report Bayesian optimization against random search.
