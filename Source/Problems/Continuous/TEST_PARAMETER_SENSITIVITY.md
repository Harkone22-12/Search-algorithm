# Test Parameter Sensitivity

## 📋 Mô Tả

Phân tích nhạy cảm tham số của 4 thuật toán: **ABC, Firefly, Cuckoo, SA**

---

## 🎯 Tham Số Mỗi Thuật Toán

**ABC (Artificial Bee Colony)**:

- `colony_size`: [10, 20, 30, 40, 50] (Mặc định: 30)
- `limit`: [50, 100, 150, 200] (Mặc định: 100)

**Firefly Algorithm**:

- `population_size`: [15, 25, 30, 40, 50] (Mặc định: 30)
- `alpha`: [0.1, 0.3, 0.5, 0.7, 0.9] (Mặc định: 0.5)
- `gamma`: [0.001, 0.01, 0.05, 0.1, 0.2] (Mặc định: 0.01)

**Cuckoo Search**:

- `population_size`: [15, 20, 25, 35, 50] (Mặc định: 25)
- `pa`: [0.1, 0.2, 0.25, 0.3, 0.4] (Mặc định: 0.25)

**Simulated Annealing**:

- `initial_temperature`: [10, 50, 100, 500, 1000] (Mặc định: 100)
- `cooling_rate`: [0.85, 0.90, 0.95, 0.98, 0.99] (Mặc định: 0.95)
- `min_temperature`: [0.001, 0.01, 0.05, 0.1] (Mặc định: 0.01)

---

## 🔌 Input (Cách Sử Dụng)

### Phân Tích ABC

```bash
cd c:\Users\ASUS\Documents\GitHub\Search-algorithm
python -m Source.Problems.Continuous.test_parameter_sensitivity ABC
```

### Phân Tích Firefly

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity Firefly
```

### Phân Tích Cuckoo

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity Cuckoo
```

### Phân Tích SA

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity SA
```

**Thời gian chạy ước tính:**

- ABC: 3-5 phút
- Firefly: 8-12 phút
- Cuckoo: 5-8 phút
- SA: 3-5 phút

---

## 📊 Output Kỳ Vọng

### Console Output

```
════════════════════════════════════════════════════════════════════════════════
PARAMETER SENSITIVITY ANALYSIS: colony_size
════════════════════════════════════════════════════════════════════════════════

Testing values: [10, 20, 30, 40, 50]

Parameter Value: 10
  Run 1/5: Best=0.000647, Time=0.2314s
  Run 2/5: Best=0.000521, Time=0.2145s
  Run 3/5: Best=0.000832, Time=0.2267s
  Run 4/5: Best=0.000705, Time=0.2198s
  Run 5/5: Best=0.000614, Time=0.2211s
  Best: 0.000521, Mean: 0.000664, Std: 0.000121, Time: 0.2227s

Parameter Value: 20
  ...

Parameter Value: 30
  ...

Parameter Value: 40
  ...

Parameter Value: 50
  Best: 0.000006, Mean: 0.000008, Std: 0.000001, Time: 0.3015s ⭐

════════════════════════════════════════════════════════════════════════════════
PARAMETER SENSITIVITY ANALYSIS: limit
════════════════════════════════════════════════════════════════════════════════

Testing values: [50, 100, 150, 200]

...
```

### Files Sinh Ra

```
Source/Problems/Continuous/sensitivity_plots/ABC/
├── sensitivity_colony_size.png     (Fitness & Time vs colony_size)
└── sensitivity_limit.png           (Fitness & Time vs limit)

Source/Problems/Continuous/sensitivity_plots/Firefly/
├── sensitivity_population_size.png (Fitness & Time vs population_size)
├── sensitivity_alpha.png           (Fitness & Time vs alpha)
└── sensitivity_gamma.png           (Fitness & Time vs gamma)

Source/Problems/Continuous/sensitivity_plots/Cuckoo/
├── sensitivity_population_size.png (Fitness & Time vs population_size)
└── sensitivity_pa.png              (Fitness & Time vs pa)

Source/Problems/Continuous/sensitivity_plots/SA/
├── sensitivity_initial_temperature.png  (Fitness & Time vs initial_temperature)
├── sensitivity_cooling_rate.png         (Fitness & Time vs cooling_rate)
└── sensitivity_min_temperature.png      (Fitness & Time vs min_temperature)
```

**Mỗi plot gồm 2 biểu đồ:**

- Trái: Fitness (Best & Mean) vs Tham số (log scale)
- Phải: Thời gian chạy vs Tham số (bar chart)

---

## 📝 Ví Dụ Chạy

### Ví dụ 1: Phân Tích ABC

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity ABC
```

**Kết quả:**

- ✅ 2 files PNG trong `sensitivity_plots/ABC/`
- ✅ Phân tích `colony_size` (5 giá trị) → 5 runs mỗi giá trị
- ✅ Phân tích `limit` (4 giá trị) → 4 runs mỗi giá trị
- ✅ Tìm được giá trị tối ưu cho mỗi tham số
- ✅ Thời gian: 3-5 phút

**Ví dụ Kết quả**:

```
ABC Parameter Sensitivity Results:

colony_size:
  10: Best=0.000647, Mean=0.000664, Time=0.2227s (Kém)
  20: Best=0.000089, Mean=0.000145, Time=0.2512s (Bình thường)
  30: Best=0.000034, Mean=0.000056, Time=0.1506s (Tốt)
  40: Best=0.000008, Mean=0.000015, Time=0.2845s (Rất tốt)
  50: Best=0.000006, Mean=0.000008, Time=0.3015s (Tốt nhất) ⭐

limit:
  50:  Best=0.000034, Time=0.1506s (Giống)
  100: Best=0.000034, Time=0.1506s (Giống) → limit không quan trọng
  150: Best=0.000034, Time=0.1506s (Giống)
  200: Best=0.000034, Time=0.1506s (Giống)
```

### Ví dụ 2: Phân Tích Firefly

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity Firefly
```

**Kết quả:**

- ✅ 3 files PNG trong `sensitivity_plots/Firefly/`
- ✅ Phân tích `population_size`, `alpha`, `gamma`
- ✅ Mỗi tham số: 5 giá trị × 5 runs
- ✅ Thời gian: 8-12 phút

**Ví dụ Kết quả**:

```
Firefly Parameter Sensitivity Results:

population_size:
  15: Best=0.000234 (Tệ)
  25: Best=0.000056 (Bình thường)
  30: Best=0.000012 (Tốt)
  40: Best=0.000008 (Rất tốt) ⭐
  50: Best=0.000009 (Rất tốt)

alpha:
  0.1: Best=0.000145 (Kém)
  0.3: Best=0.000078 (Bình thường)
  0.5: Best=0.000012 (Tốt) ⭐ ← Mặc định tốt
  0.7: Best=0.000025 (Tốt)
  0.9: Best=0.000089 (Kém)

gamma:
  0.001: Best=0.000234 (Tệ)
  0.01:  Best=0.000012 (Tốt) ⭐ ← Mặc định tốt
  0.05:  Best=0.000045 (Bình thường)
  0.1:   Best=0.000078 (Kém)
  0.2:   Best=0.000145 (Kém)
```

### Ví dụ 3: Phân Tích Cuckoo

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity Cuckoo
```

**Kết quả:**

- ✅ 2 files PNG trong `sensitivity_plots/Cuckoo/`
- ✅ Phân tích `population_size`, `pa`
- ✅ Thời gian: 5-8 phút

### Ví dụ 4: Phân Tích SA

```bash
python -m Source.Problems.Continuous.test_parameter_sensitivity SA
```

**Kết quả:**

- ✅ 3 files PNG trong `sensitivity_plots/SA/`
- ✅ Phân tích `initial_temperature`, `cooling_rate`, `min_temperature`
- ✅ Thời gian: 3-5 phút

---

## 📈 Cách Đọc Kết Quả

**Biểu đồ trái (Fitness vs Tham số):**

- Trục X: Giá trị tham số
- Trục Y: Fitness (log scale)
- Đường xanh (Mean): Giá trị trung bình
- Đường xanh lá (Best): Giá trị tốt nhất
- Dải sai số: Độ biến thiên

**Biểu đồ phải (Thời gian vs Tham số):**

- Trục X: Giá trị tham số
- Trục Y: Thời gian chạy (giây)
- Bar chart: Thời gian mỗi giá trị

**Giải thích:**

- Đường Fitness nằm ngang → Tham số không quan trọng
- Đường Fitness có đỉnh/đáy → Tham số quan trọng
- Tìm điểm có Fitness thấp nhất = Giá trị tối ưu

---

**Cập Nhật**: 31 Tháng 1, 2026
