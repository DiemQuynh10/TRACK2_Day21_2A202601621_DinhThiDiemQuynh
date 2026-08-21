# Báo cáo Lab Day 21 - CI/CD cho AI Systems

**Sinh viên:** Đinh Thị Diễm Quỳnh - 2A202601621

## 1. Bộ siêu tham số đã chọn (Bước 1)

Sau khi thử nghiệm 16 lần chạy với các tổ hợp `model_type` (random_forest, gradient_boosting,
logistic_regression), `n_estimators` (100-800), `max_depth` (3-20, None), `min_samples_split`,
kết quả tốt nhất trên `train_phase1.csv` (2998 mẫu) / `eval.csv` (500 mẫu) là:

| Tham số | Giá trị |
|---|---|
| model_type | random_forest |
| n_estimators | 800 |
| max_depth | None (không giới hạn) |
| min_samples_split | 2 |
| **accuracy** | **0.684** |
| **f1_score** | **0.6827** |

**Lý do chọn:** RandomForest với cây không giới hạn độ sâu và số cây lớn cho kết quả ổn định nhất
trong các thuật toán thử nghiệm. GradientBoosting cho kết quả tương đương ở một vài cấu hình
(0.674-0.68) nhưng nhạy với overfitting hơn khi tăng max_depth. LogisticRegression cho kết quả
thấp nhất (0.536) vì bài toán phân loại 3 lớp từ đặc trưng hóa học không tuyến tính. Việc tăng
n_estimators quá 800 (thử tới 2000) hoặc dùng ensemble voting không cải thiện thêm - đây là giới
hạn thực sự của tập dữ liệu 2998 mẫu, không phải do thiếu tinh chỉnh.

## 2. Khó khăn gặp phải và cách giải quyết

**a) Accuracy không đạt ngưỡng 0.70 ở lần chạy CI/CD đầu tiên (Bước 2).**
Với `train_phase1.csv` (2998 mẫu), accuracy tối đa đạt được chỉ ~0.68 dù đã thử nhiều thuật toán/
tham số khác nhau. Job Eval chặn Deploy đúng như thiết kế. Sau khi thực hiện Bước 3 (bổ sung
`train_phase2.csv`, nâng tổng dữ liệu lên 5996 mẫu), accuracy tăng lên ~0.75, vượt ngưỡng và
pipeline chạy đủ 4 job xanh. Điều này cho thấy bài lab được thiết kế có chủ đích để minh chứng cả
hai hành vi: gate chặn deploy khi model chưa đủ tốt, và continuous training cải thiện chất lượng
model khi có thêm dữ liệu.

**b) Lỗi xác thực GCS trong CI (`Invalid Credentials, 401`) khi chạy `dvc pull`.**
Nguyên nhân: file `.dvc/config` (đã commit lên git) có `credentialpath` trỏ tới `sa-key.json` -
file này không tồn tại trong môi trường CI (đã gitignore, không commit). DVC ưu tiên
`credentialpath` hơn biến môi trường `GOOGLE_APPLICATION_CREDENTIALS`, nên khi file không tồn tại
DVC rơi về chế độ anonymous và bị từ chối. Cách khắc phục: chuyển `credentialpath` sang
`.dvc/config.local` (không commit, chỉ dùng khi chạy cục bộ) bằng lệnh
`dvc remote modify --local myremote credentialpath sa-key.json`, để CI hoàn toàn dựa vào
`GOOGLE_APPLICATION_CREDENTIALS` được set từ secret `CLOUD_CREDENTIALS`.

**c) Job Deploy thất bại dù VM service chạy bình thường.**
`serve.py` mất khoảng 15 giây để tải model từ GCS và unpickle (do model RandomForest 800 cây khá
lớn, ~172MB), nhưng script deploy chỉ `sleep 5` trước khi health-check, dẫn tới curl thất bại vì
service chưa sẵn sàng. Khắc phục bằng cách đổi sang vòng lặp retry (10 lần, mỗi lần cách 3 giây)
thay vì chờ cố định.

**d) GitHub Actions không tự chạy dù đã push code lần đầu.**
Repo được tạo bằng cách fork nên GitHub tự động tắt Actions cho tới khi xác nhận thủ công
("Workflows aren't being run on this forked repository"). Chỉ cần bấm "I understand my workflows,
go ahead and enable them" một lần trong tab Actions.

## 3. Thách thức nâng cao đã thực hiện (Bonus)

- **Multiple algorithms:** `params.yaml` có `model_type` (random_forest / gradient_boosting /
  logistic_regression), `src/train.py` chọn thuật toán tương ứng. Đã chạy và so sánh cả 3 loại
  trong MLflow UI.
- **Data drift/skew warning:** `src/train.py` tính tỉ lệ từng lớp trong tập huấn luyện, in cảnh
  báo nếu lớp nào chiếm dưới 10%, và ghi `label_distribution` vào `outputs/metrics.json`.
