# SEARCH & NATURE-INSPIRED ALGORITHMS

A comprehensive benchmarking framework for comparing nature-inspired optimization algorithms.

## Overview

This project implements and evaluates search algorithms from two main families:

- **Classical Search** - Traditional algorithmic approaches (placeholder)
- **Nature-Inspired Algorithms** - Bio-inspired and physics-inspired optimizers (ABC, Firefly, Cuckoo, Simulated Annealing)

Current focus: Benchmarking on continuous optimization problems (Sphere function, 5D)

## Learning Objectives

- Understand how nature-inspired optimization algorithms work
- Design experiments to compare multiple algorithms
- Analyze performance using multiple metrics (quality, speed, robustness, scalability)
- Perform parameter sensitivity analysis
- Apply statistical hypothesis testing for algorithm comparison

## Project Structure

```
Source/
├── Search/Nature_Inspired/
│   ├── Biology-Based/       (ABC, Firefly, Cuckoo)
│   └── Physics-Based/       (Simulated Annealing)
└── Problems/Continuous/
    ├── Benchmarking tools & output plots
    └── Parameter sensitivity analysis
```

## Quick Start

Compare algorithms:

```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Firefly
```

Optimize parameters:

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity ABC
```

## Documentation

- [TEST_BIOINSPIRED_SPHERE.md](Source/Problems/Continuous/TEST_BIOINSPIRED_SPHERE.md) - Benchmarking guide
- [TEST_PARAMETER_SENSITIVITY.md](Source/Problems/Continuous/TEST_PARAMETER_SENSITIVITY.md) - Parameter tuning guide

## Requirements

```
Python 3.7+
numpy, scipy, matplotlib
```

**Status**: Work in progress - Core framework complete, detailed content to be added
