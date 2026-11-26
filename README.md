# TabPFN v2 — Reproduce & cải tiến: **TabPFN-GBX-PHE** (đồ án môn Deep Learning)

* **Paper gốc:** Hollmann et al., *Accurate predictions on small data with a tabular foundation
  model*, **Nature 637, 2025** (TabPFN v2).
* **Phần A – Reproduce:** TabPFN v2 dùng ngay (không tune) tốt hơn các mô hình cổ điển cấu hình
  mặc định trên dữ liệu bảng nhỏ (OpenML-CC18).
* **Phần B – Cải tiến:** **TabPFN-GBX-PHE** = (1) TabPFN-GBX: làm giàu ngữ cảnh của TabPFN v2 bằng
  thông tin GBDT được cross-fit, nhắm vào điểm yếu với dữ liệu nhiều biến phân loại; (2) portfolio
  cấu hình tiền xử lý của cùng checkpoint TabPFN v2; (3) post-hoc ensemble selection trên toàn bộ
  portfolio và CatBoost. Không fine-tune, không pretrain lại, chạy trên Kaggle T4.

---

## 1. Điểm yếu của TabPFN v2 và ý tưởng

Kết quả Phần A (33 dataset OpenML-CC18) cho thấy TabPFN v2 đứng đầu tổng thể, nhưng **thua mô
hình cây trên các dataset có nhiều biến phân loại** (ví dụ cylinder-bands với 17 biến phân loại,
credit-g, credit-approval, dresses-sales). Lý do:

1. **Biến phân loại** được đưa vào TabPFN dưới dạng mã số nguyên tùy ý. Prior giả lập dùng để
   pretrain TabPFN chủ yếu sinh từ cơ chế liên tục, nên với vài trăm dòng mô hình khó học được
   thống kê riêng cho từng giá trị của biến.
2. **Thiên kiến quy nạp khác nhau:** TabPFN học từ prior trơn; GBDT tách theo trục và dùng target
   statistics (CatBoost), hai thiên kiến này bổ sung cho nhau.

**TabPFN-GBX** giữ nguyên TabPFN v2 đã pretrain và chỉ thêm cột vào bảng đầu vào, tính bằng
cross-fitting K-fold trên tập train:

| Cột thêm | Nội dung | Đánh vào điểm yếu |
|---|---|---|
| **OOF-TE** | target encoding out-of-fold (có làm mịn) cho từng biến phân loại | biến phân loại |
| **OOF-GBDT** | log-xác suất out-of-fold của CatBoost (C cột, 1 cột logit nếu nhị phân) | thiên kiến cây |

* Dòng train nhận giá trị **out-of-fold**, nên ngữ cảnh không chứa rò rỉ nhãn.
  `tests/test_smoke.py` kiểm tra điều này bằng nhãn ngẫu nhiên (AUC phải ở mức ngẫu nhiên).
* Dòng test nhận **trung bình dự đoán của K mô hình fold**, cùng phân phối với giá trị OOF.
* TabPFN tự quyết định trong ngữ cảnh khi nào tin GBDT, khi nào tin đặc trưng gốc.
* **Portfolio cấu hình TabPFN v2** (cùng checkpoint, chỉ đổi suy luận): one-hot hoặc ordinal cho
  biến phân loại, biến đổi KDI, thêm 50 đặc trưng đa thức, nhiệt độ softmax 0.75, trung bình logits,
  32 estimator; và TabPFN-GBX với one-hot. Paper Nature cũng báo cáo post-hoc ensembling (PHE) các
  cấu hình TabPFN v2 giúp cải thiện so với mặc định.
* **TabPFN-GBX-PHE (bản chính):** ensemble selection (Caruana 2004) trên toàn bộ portfolio +
  TabPFN-GBX + CatBoost, trọng số chọn trên validation nội bộ (80/20 trong fold train, không dùng
  tập test). So với **TabPFN v2-PHE** (chỉ portfolio TabPFN) để tách riêng đóng góp của GBX.

Chi phí: K lần fit CatBoost trên CPU và khoảng 10 lượt suy luận TabPFN (chậm hơn bản gốc vài lần);
không cần gradient.

## 2. Thiết kế thực nghiệm

* **Dữ liệu:** OpenML-CC18 (suite 99), lọc ≤ 3000 dòng, ≤ 100 đặc trưng, ≤ 10 lớp, mỗi lớp
  ≥ 10 mẫu: 33 dataset.
* **Giao thức:** stratified 3-fold CV, cùng fold cho mọi phương pháp.
* **So sánh:**

| Nhóm | Phương pháp |
|---|---|
| Bản gốc | TabPFN v2 (checkpoint của paper Nature, 8 estimator) |
| Baseline cổ điển | Logistic Regression, kNN, Random Forest, XGBoost, LightGBM, CatBoost |
| Cải tiến đã có | TabPFN v2 + fine-tune (module chính thức `tabpfn.finetuning`, Rubachev et al. 2025); Ens(TabPFN v2, CatBoost); Ens(TabPFN v2-FT, CatBoost); TabPFN v2-PHE (ensemble cấu hình) |
| **Của ta** | **TabPFN-GBX-PHE** (bản chính), +FT; TabPFN-GBX, TabPFN-GBX-Ens; ablation: chỉ OOF-TE, chỉ OOF-GBDT, từng cấu hình portfolio |

* **Metric:** ROC-AUC (chính), accuracy, log-loss, mean rank, win/tie/loss, Wilcoxon signed-rank.
* **Phân tích:** mức cải thiện theo nhóm dataset (TabPFN v2 đang thua / đang thắng; có / không có
  biến phân loại) và tương quan Spearman giữa mức cải thiện và độ yếu của TabPFN v2.

## 3. Chạy trên Kaggle

1. Đẩy repo lên GitHub (public).
2. Kaggle: **Accelerator = GPU T4 x2**, **Internet = On**.
3. **Session 1:** `notebooks/01_reproduce_tabpfn_v2.ipynb`. Sửa `REPO_URL`, chạy hết, tải
   `session1_results.zip`.
4. **Session 2:** `notebooks/02_improve_tabpfn_gbx_phe.ipynb`. Upload `session1_results.zip` thành
   Kaggle Dataset, gắn vào notebook, sửa `REPO_URL`, chạy hết, tải `final_results.zip`.

Mọi script chạy tiếp được: chạy lại cell sẽ bỏ qua phần đã xong.

| Bước | Thời gian ước tính trên Kaggle |
|---|---|
| Notebook 1 (Phần A) | ~1–1,5 giờ |
| Notebook 2: GBX + portfolio (GPU 0) song song fine-tune baseline (GPU 1) | ~3–4 giờ |
| Ensemble + báo cáo | vài phút |

Giảm thời gian: `FT_TIME = 30` hoặc `GBX_FOLDS = 4` trong Notebook 2, hoặc bỏ `pfn_wide` khỏi danh sách.

## 4. Cấu trúc repo
```
tabpfn-dl-project/
├── notebooks/01_reproduce_tabpfn_v2.ipynb   # Session 1
├── notebooks/02_improve_tabpfn_gbx_phe.ipynb  # Session 2
├── scripts/run_benchmark.py   # chạy mọi phương pháp, lưu metric + dự đoán, resumable
├── scripts/run_ensembles.py   # ensemble selection từ dự đoán đã lưu
├── scripts/make_report.py     # bảng Markdown/CSV + hình PNG
├── src/tabdl/augment.py       # TabPFN-GBX: OOF target encoding, OOF CatBoost logits
├── src/tabdl/models.py        # baselines, TabPFN v2, TabPFN v2-FT, portfolio cấu hình, biến thể GBX
├── src/tabdl/data.py          # OpenML-CC18 + fallback sklearn, chia fold
├── src/tabdl/ensemble.py, metrics.py, utils.py
└── tests/test_smoke.py        # kiểm thử nhanh, gồm kiểm tra không rò rỉ nhãn
```

Chạy local: `pip install -r requirements.txt && python tests/test_smoke.py`, sau đó
`python scripts/run_benchmark.py --source openml --val_preds --methods ...`
(`--source sklearn` để thử offline).

## 5. Đầu ra (`results/report/`)
* `table_A_reproduction.csv`, `table_A_weakness_gap.csv`, `fig_A_rank.png`
* `table_B_improvement.csv`, `table_B_deltas.csv`, `table_B_subgroups.csv`, `table_B_portfolio_members.csv`,
  `table_B_ensemble_weights.csv`, `fig_B_delta_auc.png`, `fig_B_delta_logloss.png`,
  `fig_B_gain_vs_weakness.png`
* `report.md` tổng hợp toàn bộ

## 6. Giới hạn cần nêu trong báo cáo
* Baseline dùng tham số mặc định (paper còn so với baseline tune 4 giờ); 3 fold thay vì 10 lần lặp;
  dataset ≤ 3000 dòng để vừa ngân sách T4.
* TabPFN-GBX-PHE tốn thêm thời gian (K lần fit CatBoost + ~10 lượt suy luận TabPFN, mỗi mô hình
  fit hai lần cho validation nội bộ) so với TabPFN v2 gốc; báo cáo có cột thời gian để so sánh.
* Kết luận chỉ dựa trên các con số chạy được ở Notebook 2. Báo cáo đúng kết quả, kể cả khi một
  thành phần (ví dụ OOF-TE) không giúp.

## 7. Tài liệu tham khảo
* Hollmann et al. *Accurate predictions on small data with a tabular foundation model.* Nature 637, 2025.
* Prokhorenkova et al. *CatBoost: unbiased boosting with categorical features.* NeurIPS 2018.
* Rubachev et al. *On Finetuning Tabular Foundation Models.* 2025.
* Caruana et al. *Ensemble Selection from Libraries of Models.* ICML 2004.
* Wolpert. *Stacked Generalization.* Neural Networks, 1992.
* Bischl et al. *OpenML Benchmarking Suites.* NeurIPS Datasets & Benchmarks 2021.
