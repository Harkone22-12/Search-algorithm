# Thuật Toán Tìm Kiếm và Tối Ưu Hóa

## Giới Thiệu

Dự án này triển khai và so sánh các thuật toán tìm kiếm từ hai nhóm chính:

- **Tìm Kiếm Cổ Điển**: Các thuật toán truyền thống (BFS, DFS, A\*)
- **Thuật Toán Lấy Cảm Hứng Từ Tự Nhiên**:
  - Dựa trên sinh học (ABC, ACO, Firefly, Cuckoo Search, PSO)
  - Dựa trên tiến hóa (Genetic Algorithm, Differential Evolution)
  - Dựa trên vật lý (Hill Climbing, Simulated Annealing)

Dự án cung cấp khung công tác để đánh giá hiệu suất của các thuật toán trên các bài toán tối ưu hóa liên tục (Sphere, Rastrigin) và rời rạc (Knapsack, TSP).

## Cấu Trúc Dự Án

```
Source/
├── Search/                          # Các thuật toán tìm kiếm
│   ├── Classical/                   # BFS, DFS, A*
│   └── Nature_Inspired/             # Thuật toán lấy cảm hứng từ tự nhiên
│       ├── Biology_Based/           # ABC, ACO, FA, PSO, Cuckoo
│       ├── Evolution_Based/         # GA, DE
│       └── Physics_Based/           # Hill Climbing, Simulated Annealing
│
└── Problems/                        # Các bài toán tối ưu hóa
    ├── Continuous/                  # Bài toán tối ưu hóa liên tục
    │   ├── Functions/               # Sphere, Rastrigin, 3D Visualizer
    │   └── Tests/                   # Lab 1, Lab 2
    │
    └── Discrete/                    # Bài toán tối ưu hóa rời rạc
        ├── Functions/               # Knapsack, TSP
        └── Tests/                   # Lab 3
```

## Yêu Cầu Hệ Thống

- Python 3.8 trở lên
- pip (trình quản lý gói Python)

## Cài Đặt

### 1. Clone hoặc tải xuống dự án

```bash
cd Search-algorithm
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## Hướng Dẫn Chạy Các Lab

### Lab 1: Đánh Giá Hiệu Suất Các Thuật Toán trên Bài Toán Sphere và Rastrigin

**Mục tiêu**: So sánh hiệu suất của 8 thuật toán khác nhau trên hai hàm mục tiêu kinh điển.

**Cách chạy**:

```bash
python -m Source.Problems.Continuous.Tests.Lab_1
```

**Kết quả**:

- Bảng so sánh hiệu suất (chất lượng giải pháp, thời gian chạy, độ ổn định)
- Biểu đồ so sánh (convergence plots, box plots)
- Kết quả được lưu trong thư mục: `Source/Problems/Continuous/Tests/Lab1_Results/`

---

### Lab 2: Phân Tích Độ Nhạy Tham Số của PSO

**Mục tiêu**: Phân tích ảnh hưởng của các siêu tham số (cognition weight, social weight, inertia) đối với hiệu suất của thuật toán PSO.

**Cách chạy**:

```bash
python -m Source.Problems.Continuous.Tests.Lab_2
```

**Kết quả**:

- Biểu đồ độ nhạy tham số (1D: parameter vs performance)
- Heatmap 2D để trực quan hóa tương tác giữa các tham số
- Khuyến nghị tham số tối ưu
- Kết quả được lưu trong thư mục: `Source/Problems/Continuous/Tests/Lab2_Results/`

---

### Lab 3: Đánh Giá Thuật Toán trên Bài Toán Rời Rạc (Knapsack & TSP)

**Mục tiêu**: So sánh 8 thuật toán trên các bài toán tối ưu hóa rời rạc khác nhau.

**Cách chạy**:

```bash
python -m Source.Problems.Discrete.Tests.Lab_3
```

**Kết quả**:

- Bảng so sánh hiệu suất trên bài toán Knapsack và TSP
- Biểu đồ phân tích chất lượng và thời gian chạy
- Thống kê chi tiết về sự hội tụ
- Kết quả được lưu trong thư mục: `Source/Problems/Discrete/Tests/Lab3_results/`

---

## Giải Thích Các Hàm Mục Tiêu

### Bài Toán Liên Tục

- **Sphere Function**: Hàm đơn giản, mục tiêu là tìm cực tiểu tại điểm gốc (0, 0, ..., 0)
- **Rastrigin Function**: Hàm phức tạp với nhiều cực tiểu địa phương, thử thách khả năng thoát khỏi bẫy của các thuật toán

### Bài Toán Rời Rạc

- **Knapsack Problem**: Chọn các mục không vượt quá giới hạn trọng lượng để tối đa hóa giá trị
- **TSP (Traveling Salesman Problem)**: Tìm đường đi ngắn nhất ghé thăm tất cả thành phố

## Các Thuật Toán Được Triển Khai

| Nhóm            | Tên           | Mô Tả                                    |
| --------------- | ------------- | ---------------------------------------- |
| Classical       | BFS           | Tìm kiếm theo chiều rộng                 |
| Classical       | DFS           | Tìm kiếm theo chiều sâu                  |
| Classical       | A\*           | Tìm kiếm heuristic với chi phí           |
| Biology-Based   | ABC           | Artificial Bee Colony - Ong nhân tạo     |
| Biology-Based   | ACO           | Ant Colony Optimization - Bầy kiến       |
| Biology-Based   | FA            | Firefly Algorithm - Thuật toán Đom đóm   |
| Biology-Based   | PSO           | Particle Swarm Optimization - Bầy chim   |
| Biology-Based   | Cuckoo        | Cuckoo Search - Chim cúc cu              |
| Evolution-Based | GA            | Genetic Algorithm - Thuật toán di truyền |
| Evolution-Based | DE            | Differential Evolution - Tiến hóa vi sai |
| Physics-Based   | Hill Climbing | Leo đồi                                  |
| Physics-Based   | SA            | Simulated Annealing - Luyện kim mô phỏng |

## Thư Viện Sử Dụng

- **numpy**: Tính toán số học
- **pandas**: Xử lý và phân tích dữ liệu
- **matplotlib**: Vẽ biểu đồ
- **scipy**: Tính toán khoa học nâng cao và kiểm định thống kê

## Hướng Dẫn Phát Triển Thêm

Để thêm một thuật toán mới, bạn cần:

1. Tạo file Python trong thư mục phù hợp (Biology_Based, Evolution_Based, hoặc Physics_Based)
2. Kế thừa từ lớp cơ sở `SearchAlgorithm` hoặc `OptimizationBase`
3. Triển khai phương thức `search()` hoặc `optimize()`
4. Import và thêm thuật toán vào file Lab tương ứng

## Lưu Ý Khi Sử Dụng

- Các Lab có thể mất từ vài phút đến vài giờ tùy thuộc vào cấu hình máy tính
- Kết quả có tính ngẫu nhiên, do đó nên chạy nhiều lần để có kết quả đáng tin cậy
- Các biểu đồ và dữ liệu chi tiết được lưu tự động vào thư mục Results

## Liên Hệ và Hỗ Trợ

Nếu gặp vấn đề khi chạy dự án, vui lòng kiểm tra:

- Python version: `python --version` (phải ≥ 3.8)
- Tất cả thư viện đã được cài đặt: `pip list`
- Đường dẫn thư mục đúng khi chạy Lab
