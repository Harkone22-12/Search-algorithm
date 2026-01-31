# Test Bioinspired Algorithms on Sphere Function

## 📋 Mô Tả

So sánh 4 thuật toán tối ưu trên hàm Sphere: **ABC, Firefly, Cuckoo, SA**

| Thuật Toán            | Viết tắt |
| --------------------- | -------- |
| Artificial Bee Colony | ABC      |
| Firefly Algorithm     | FA       |
| Cuckoo Search         | CS       |
| Simulated Annealing   | SA       |

---

## 🔌 Input (Cách Sử Dụng)

### Chạy Tất Cả 4 Thuật Toán

```bash
cd c:\Users\ASUS\Documents\GitHub\Search-algorithm
python -m Source.Problems.Continuous.test_bioinspired_sphere
```

### Chạy 1 Thuật Toán

```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC
python -m Source.Problems.Continuous.test_bioinspired_sphere Firefly
python -m Source.Problems.Continuous.test_bioinspired_sphere Cuckoo
python -m Source.Problems.Continuous.test_bioinspired_sphere SA
```

### Chạy Nhiều Thuật Toán

```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Firefly
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Firefly Cuckoo
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Cuckoo SA
```

**Thời gian chạy ước tính:**

- 1 thuật toán: 1-2 phút
- 2 thuật toán: 2-4 phút
- 3 thuật toán: 4-6 phút
- 4 thuật toán: 5-10 phút

---

## 📊 Output Kỳ Vọng

### Console Output

```
════════════════════════════════════════════════════════════════════════════════
BENCHMARKING BIO-INSPIRED ALGORITHMS ON SPHERE FUNCTION
════════════════════════════════════════════════════════════════════════════════

Configuration:
  Problem: Sphere Function (f(x) = sum(x_i^2))
  Dimensions: 5D
  Max Iterations: 100
  Number of Runs: 10
  Search Space: [-5.12, 5.12]
  Optimal Value: 0.0

[1/1] Running Artificial Bee Colony...
ABC Run 1/10: Best=0.000034, Time=0.1525s
ABC Run 2/10: Best=0.000034, Time=0.1218s
...
ABC Run 10/10: Best=0.000034, Time=0.1185s

════════════════════════════════════════════════════════════════════════════════
DETAILED METRICS
════════════════════════════════════════════════════════════════════════════════

ALGORITHM: ABC
1. CONVERGENCE SPEED: 2.0 ± 0.0
2. SOLUTION QUALITY: Best=0.000034, Mean=0.000034
3. ROBUSTNESS: Std=0.000000
4. COMPUTATIONAL COMPLEXITY: Time=0.1260s
5. EXPLORATION vs EXPLOITATION: Score=10.00

════════════════════════════════════════════════════════════════════════════════
SCALABILITY ANALYSIS
════════════════════════════════════════════════════════════════════════════════

Testing with 5 dimensions...
  ABC (5D): Best=0.000034, Mean=0.000034, Time=0.1260s

Testing with 10 dimensions...
  ABC (10D): Best=0.045757, Mean=0.045757, Time=0.1777s

Testing with 20 dimensions...
  ABC (20D): Best=6.116693, Mean=6.116693, Time=0.2785s

Testing with 30 dimensions...
  ABC (30D): Best=21.004935, Mean=21.004935, Time=0.3586s

════════════════════════════════════════════════════════════════════════════════
BENCHMARKING COMPLETE
════════════════════════════════════════════════════════════════════════════════

Tested Algorithms: ABC
Results saved to: Source/Problems/Continuous/benchmark_plots/
```

### Files Sinh Ra

**1 thuật toán (ABC):**

```
Source/Problems/Continuous/benchmark_plots/
├── 01_solution_quality.png         (Histogram + Trend line)
├── 02_execution_time.png           (Time distribution + Bar)
├── 03_convergence_curves.png       (10 runs + Mean + Best/Worst)
├── 04_robustness_boxplot.png       (Box plot + Violin)
└── 05_scalability_analysis.png     (1x1 grid: ABC 5D→30D)
```

**2 thuật toán (ABC + Firefly):**

```
Source/Problems/Continuous/benchmark_plots/
├── 01_solution_quality.png         (So sánh ABC vs Firefly)
├── 02_execution_time.png           (So sánh thời gian)
├── 03_convergence_curves.png       (So sánh đường hội tụ)
├── 04_robustness_boxplot.png       (So sánh độ ổn định)
└── 05_scalability_analysis.png     (1x2 grid: ABC & Firefly)
```

**3 thuật toán (ABC + Firefly + Cuckoo):**

```
Source/Problems/Continuous/benchmark_plots/
├── 01_solution_quality.png         (So sánh 3 thuật toán)
├── 02_execution_time.png
├── 03_convergence_curves.png
├── 04_robustness_boxplot.png
└── 05_scalability_analysis.png     (2x2 grid: 3 subplots + 1 ẩn)
```

**4 thuật toán (Tất cả):**

```
Source/Problems/Continuous/benchmark_plots/
├── 01_solution_quality.png         (So sánh 4 thuật toán)
├── 02_execution_time.png
├── 03_convergence_curves.png
├── 04_robustness_boxplot.png
└── 05_scalability_analysis.png     (2x2 grid: 4 subplots)
```

**Grid tự động:**

- 1 algo → 1x1 grid
- 2 algo → 1x2 grid
- 3 algo → 2x2 grid (1 ẩn)
- 4 algo → 2x2 grid (đầy đủ)

---

## 📝 Ví Dụ Chạy

### Ví dụ 1: Chạy ABC tự mình

```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC
```

**Kết quả:**

- ✅ 5 files PNG (01-05)
- ✅ Scalability plot: 1 subplot (1x1)
- ✅ Thời gian: ~1-2 phút
- ✅ Output: Metricscho ABC + Scalability 5D→30D

### Ví dụ 2: So sánh ABC vs Firefly

```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Firefly
```

**Kết quả:**

- ✅ 5 files PNG (01-05)
- ✅ Plots 01-04: So sánh ABC vs Firefly
- ✅ Scalability plot: 2 subplots (1x2)
- ✅ Thời gian: ~3-4 phút
- ✅ Output: Hypothesis testing + Scalability

### Ví dụ 3: So sánh 3 thuật toán

```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere ABC Firefly Cuckoo
```

**Kết quả:**

- ✅ 5 files PNG (01-05)
- ✅ Plots 01-04: So sánh 3 thuật toán
- ✅ Scalability plot: 3 subplots (2x2, 1 ẩn)
- ✅ Thời gian: ~5-6 phút
- ✅ Output: Pairwise hypothesis testing (3 tests)

### Ví dụ 4: Chạy tất cả (không args)

```bash
python -m Source.Problems.Continuous.test_bioinspired_sphere
```

**Kết quả:**

- ✅ 5 files PNG (01-05)
- ✅ Plots 01-04: So sánh 4 thuật toán
- ✅ Scalability plot: 4 subplots (2x2)
- ✅ Thời gian: ~8-10 phút
- ✅ Output: Pairwise hypothesis testing (6 tests)

---

**Cập Nhật**: 31 Tháng 1, 2026
