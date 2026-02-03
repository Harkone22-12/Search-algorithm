# SEARCH & NATURE-INSPIRED ALGORITHMS

A comprehensive framework for implementing, experimenting with, and comparing **classical search algorithms** and **nature-inspired optimization algorithms** as required in the course project.

This project is developed as a **group assignment** and focuses on both **algorithm implementation** and **experimental comparison** across multiple problem domains.

---

## Project Objectives

- Implement search and optimization algorithms from multiple categories
- Understand differences between **exact search** and **metaheuristic optimization**
- Apply algorithms to **continuous** and **discrete** problems
- Compare algorithms using solution quality, convergence, and computational cost
- Design a reusable and extensible software architecture

---

## Implemented Algorithm Categories

### 1️⃣ Evolution-Based Algorithms
- **Differential Evolution (DE)** ✅
- **Genetic Algorithm (GA)** ✅

---

### 2️⃣ Physics-Based Algorithms
- **Simulated Annealing (SA)** ✅

SA is implemented and used both as:
- A classical local search method
- A metaheuristic optimization algorithm

---

### 3️⃣ Biology-Based Algorithms
- **Particle Swarm Optimization (PSO)** ✅
- **Artificial Bee Colony (ABC)** ✅
- **Firefly Algorithm (FA)** ✅
- **Cuckoo Search (CS)** ✅
- **Ant Colony Optimization (ACO)** ✅

This category is fully satisfied with multiple algorithms implemented.

---

### 4️⃣ Human-Based Algorithms (Optional / Bonus)
- **Teaching–Learning-Based Optimization (TLBO)** ⏳ (not required)

---

### 5️⃣ Traditional (Classical) Search Algorithms
At least four traditional algorithms are required.

Implemented:
- **Breadth-First Search (BFS)** ✅
- **Depth-First Search (DFS)** ✅
- **A\* Search** ✅
- **Hill Climbing** ✅
- **Simulated Annealing** ✅

✔ Requirement satisfied with more than four algorithms.

---

## Supported Problems

### 🔹 Continuous Optimization
- Sphere Function (n-dimensional)
- Parameter sensitivity analysis

### 🔹 Discrete & Combinatorial Problems
- Traveling Salesman Problem (TSP)
- Knapsack Problem (KP)
- Graph Coloring (GC)
- Shortest Path Problem

Each problem is designed to be compatible with:
- Classical search algorithms (when applicable)
- Nature-inspired optimization algorithms using cost minimization

---

## Project Structure

```text
Source/
├── Search/
│   ├── Search.py                  # Base SearchAlgorithm abstraction
│   ├── traditional/               # BFS, DFS, A*, Hill Climbing
│   └── Nature_Inspired/
│       ├── Physics-Based/         # Simulated Annealing
│       ├── Biology-Based/         # PSO, ABC, FA, CS
│       └── Evolution-Based/       # DE, GA
│
├── Problems/
│   ├── problem.py                 # Base SearchProblem abstraction
│   ├── Continuous/                # Sphere
│   └── Discrete/                  # TSP, KP, GC, Shortest Path
│
└── Experiments/
    ├── Benchmark scripts
    └── Result plots & statistics

```

## Design Principles

Algorithms are problem-agnostic

- Problems are algorithm-independent

- All optimization problems follow a minimization convention

- Constraints are handled via penalty functions

- Continuous algorithms applied to discrete problems use encoding/decoding

- Each algorithm reports:

    - Best solution

    - Best cost

    - Iteration history

    - Expanded/evaluated states

## Example Usage

### Continuous Optimization
```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Firefly
```

### Parameter Sensitivity Analysis
```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity ABC
```

### Discrete Optimization
```python
problem = TSPProblem(cities)
result = sa.search(problem)
print(result["best_state"], result["cost"])
```

## Requirements
```text
Python 3.7+
numpy
scipy
matplotlib
```
## Status


- Core framework: ✅ complete

- Required algorithms: ✅ implemented

- Discrete & continuous problems: ✅ implemented

- Benchmarking & comparison: ⚙️ in progress

- Optional algorithms (GA, ACO, TLBO): ⏳ future work