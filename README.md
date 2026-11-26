# TabPFN v2 — Reproduce & cải tiến (đồ án môn Deep Learning)

Đồ án gồm hai phần:

* **Phần A – Reproduce:** kiểm chứng kết luận chính của paper
  *Accurate predictions on small data with a tabular foundation model*
  (Hollmann et al., **Nature 637, 2025**): TabPFN v2 dùng ngay (không tune) cho kết quả tốt hơn
  các mô hình cổ điển cấu hình mặc định trên dữ liệu bảng nhỏ.
* **Phần B – Cải tiến**, theo hai hướng nhẹ, chạy được trên Kaggle T4:
  * **B-1 (mức ứng dụng):** làm TabPFN v2 mạnh hơn trên từng dataset bằng fine-tune chính thức
    và **ensemble selection** hậu kỳ với CatBoost.
  * **B-2 (mức kiến trúc, phần "deep learning" chính):** pretrain lại một foundation model dạng
    bảng thu nhỏ (**nanoTabPFN**, cùng kiến trúc TabPFN v2) và kiểm tra ba ý tưởng mới của
    2025: **Gated Attention** (NeurIPS 2025 best paper), **Dynamic Tanh** (CVPR 2025) và
    optimizer **Muon**, với ablation đầy đủ, nhiều seed và kiểm định thống kê.

> Không thể pretrain lại TabPFN v2 thật trên T4: bản gốc được pretrain trên 8 GPU RTX 2080 Ti
> trong 2 tuần. Vì vậy phần B-2 so sánh các biến thể nanoTabPFN với nhau ở **cùng chi phí tính
> toán và cùng dữ liệu pretrain**, không so trực tiếp với checkpoint TabPFN v2.

---

## 1. Cấu trúc repo

```
tabpfn-dl-project/
├── README.md
├── requirements.txt
├── notebooks/kaggle_run_all.ipynb   # notebook chạy 2 session trên Kaggle
├── scripts/
│   ├── run_benchmark.py             # Phần A + B-1: TabPFN v2 / fine-tune / baselines
│   ├── run_ensembles.py             # Phần B-1: ensemble selection từ dự đoán đã lưu
│   ├── run_nano_pretrain.py         # Phần B-2: pretrain các biến thể nanoTabPFN
│   ├── run_nano_eval.py             # Phần B-2: đánh giá trên dữ liệu thật
│   └── make_report.py               # gom kết quả -> bảng Markdown/CSV + hình PNG
├── src/tabdl/
│   ├── data.py        # OpenML-CC18 (suite 99) + fallback sklearn, chia fold, tiền xử lý
│   ├── models.py      # LR, kNN, RF, XGBoost, LightGBM, CatBoost, TabPFN v2, TabPFN v2-FT
│   ├── ensemble.py    # Caruana ensemble selection
│   ├── metrics.py     # ROC-AUC (OvR), accuracy, log-loss
│   ├── utils.py
│   └── nano/
│       ├── prior.py     # prior sinh dữ liệu bảng giả lập (MLP-SCM + cây + tuyến tính)
│       ├── model.py     # nanoTabPFN (attention theo đặc trưng + theo dòng), DyT, Gated Attn
│       ├── muon.py      # optimizer Muon (Newton–Schulz) + AdamW phụ
│       ├── train.py     # vòng lặp pretrain, 6 biến thể
│       └── evaluate.py  # dự đoán in-context + ensemble hoán vị lớp/đặc trưng
├── tests/test_smoke.py              # kiểm thử nhanh (CPU, < 1 phút)
```

## 2. Phương pháp

### Phần A — Reproduce
* **Dữ liệu:** bộ benchmark OpenML-CC18 (suite 99, cùng họ benchmark với paper), lọc các
  dataset có ≤ 3000 dòng, ≤ 100 đặc trưng, ≤ 10 lớp. Mỗi lớp cần ít nhất 10 mẫu.
* **Giao thức:** stratified 3-fold CV, cùng fold cho mọi phương pháp.
* **Phương pháp:** TabPFN v2 (checkpoint gốc của paper Nature `tabpfn-v2-classifier.ckpt`,
  8 estimator) so với Logistic Regression, kNN, Random Forest, XGBoost, LightGBM, CatBoost,
  tất cả dùng tham số mặc định.
* **Metric:** ROC-AUC (OvR macro, metric chính của paper), accuracy, log-loss, thời gian;
  mean rank, win/tie/loss và kiểm định Wilcoxon theo cặp.

### Phần B-1 — Cải tiến TabPFN v2 trên từng dataset
1. **Fine-tune toàn bộ** TabPFN v2 bằng module chính thức `tabpfn.finetuning`, giới hạn thời gian
   mỗi lần fit, early stopping trên validation nội bộ.
2. **Ensemble selection** (Caruana 2004): chia fold train thành 80/20, fit các mô hình trên 80%,
   chọn trọng số tối thiểu log-loss trên 20%, sau đó áp trọng số cho mô hình fit lại trên 100%
   fold train. Tập test không được dùng khi chọn trọng số.
   Ba tổ hợp: `TabPFN v2 + CatBoost`, `TabPFN v2-FT + CatBoost`, `TabPFN v2 + v2-FT + CatBoost`.

### Phần B-2 — Cải tiến kiến trúc foundation model (nanoTabPFN)
* **Kiến trúc** (theo TabPFN v2): mỗi ô của bảng là một token; mỗi layer gồm attention giữa
  các đặc trưng trong một dòng, attention giữa các dòng trong một cột (dòng test chỉ được nhìn
  dòng train), MLP. Định danh cột bằng embedding ngẫu nhiên; nhãn đọc ra từ ô cột target.
* **Prior pretrain:** hỗn hợp MLP-SCM ngẫu nhiên (kích hoạt ngẫu nhiên, nhiễu, đồ thị thưa),
  tổng các cây quyết định ngẫu nhiên, và hàm tuyến tính thưa; target được rời rạc hóa thành
  2–10 lớp; một phần đặc trưng được rời rạc hóa giống biến phân loại.
* **Biến thể** (cùng số bước, cùng batch, cùng chuỗi dữ liệu giả lập):

| Biến thể | Norm | Attention | Optimizer |
|---|---|---|---|
| `base` | LayerNorm | thường | AdamW |
| `dyt` | **DyT** | thường | AdamW |
| `gate` | LayerNorm | **gated** | AdamW |
| `muon` | LayerNorm | thường | **Muon** |
| `gate_muon` | LayerNorm | **gated** | **Muon** |
| `full` | **DyT** | **gated** | **Muon** |

* **Gated attention:** `Y = SDPA(Q,K,V) ⊙ σ(X W_g)`, cổng riêng cho từng head và từng phần tử,
  đặt ngay sau SDPA (vị trí G1 tốt nhất trong paper của Qiu et al.).
* **DyT:** `DyT(x) = γ · tanh(αx) + β` thay mọi LayerNorm, α khởi tạo 0.5.
* **Muon:** trực giao hóa update của các ma trận ẩn bằng 5 vòng Newton–Schulz; embedding, head,
  tham số norm và bias dùng AdamW.
* **Đánh giá:** cùng các dataset OpenML-CC18; fold train được lấy mẫu con 200 dòng (đúng cỡ
  ngữ cảnh lúc pretrain), tối đa 20 đặc trưng; so với các baseline và TabPFN v2 thật **trên đúng
  cùng ngữ cảnh 200 dòng**; 3 seed pretrain, báo cáo mean ± std và Wilcoxon so với `base`.

## 3. Chạy trên Kaggle (khuyến nghị)

1. Đưa repo này lên GitHub (public).
2. Tạo notebook Kaggle, import `notebooks/kaggle_run_all.ipynb`.
3. Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
4. Sửa `REPO_URL` ở cell đầu, rồi chạy:
   * **Session 1:** phần "0. Cài đặt chung" và "SESSION 1" (Phần A + B-1). Tải `session1_results.zip`.
   * **Session 2:** phần "0. Cài đặt chung" và "SESSION 2" (Phần B-2 + báo cáo cuối).
     Muốn báo cáo cuối gồm cả Session 1: upload `session1_results.zip` thành Kaggle Dataset và
     gắn vào notebook. Tải `final_results.zip`.

Mọi script đều resumable: chạy lại sẽ bỏ qua phần đã xong.

### Thời gian dự kiến trên T4 (ước tính, chưa đo trên Kaggle)
| Bước | Thời gian |
|---|---|
| Phần A (≈30 dataset × 3 fold × 7 phương pháp) | ~1 giờ |
| Phần B-1 fine-tune (`FT_TIME=60` s/lần fit) | ~2–4 giờ |
| Phần B-2 pretrain 18 run × 3000 bước (2 GPU song song) | ~2–3 giờ |
| Phần B-2 đánh giá | ~20–40 phút |

Notebook có cell đo tốc độ trước khi pretrain và in ra tổng thời gian dự kiến; giảm `STEPS`
hoặc `FT_TIME` nếu cần.

## 4. Chạy local (tùy chọn)
```bash
pip install -r requirements.txt
python tests/test_smoke.py
python scripts/run_benchmark.py --source openml --val_preds \
       --methods logreg,knn,rf,xgb,lgbm,catboost,tabpfn_v2
python scripts/run_benchmark.py --source openml --val_preds --methods tabpfn_v2_ft
python scripts/run_ensembles.py
python scripts/run_nano_pretrain.py --seeds 0,1,2 --steps 3000
python scripts/run_nano_eval.py --source openml --with_tabpfn
python scripts/make_report.py
```
`--source sklearn` chạy được offline với vài dataset của scikit-learn (để thử nhanh).

## 5. Kết quả đầu ra
`results/report/report.md` cùng các file:
* `table_A_reproduction.csv`, `table_A_per_dataset_auc.csv`, `fig_A_rank.png`, `fig_A_scatter.png`
* `table_B1_improvement.csv`, `table_B1_ensemble_weights.csv`, `fig_B1_logloss.png`
* `table_B2_nano.csv`, `table_B2_vs_base.csv`, `fig_B2_curves.png`, `fig_B2_auc.png`

## 6. Trạng thái kiểm thử
* `tests/test_smoke.py` đã chạy qua trên CPU: prior, cả 6 biến thể (forward, backward, một bước
  optimizer), kiểm tra dòng test không ảnh hưởng lẫn nhau, dự đoán in-context, metric, ensemble.
* `run_benchmark.py`, `run_ensembles.py`, `run_nano_eval.py`, `make_report.py` đã chạy thử với
  dữ liệu sklearn (offline). TabPFN v2 cần tải weights từ HuggingFace nên chỉ chạy được khi có
  Internet (Kaggle).
* Mọi con số trong báo cáo phải lấy từ lần chạy trên Kaggle (`results/report/`).

## 7. Giới hạn cần nêu trong báo cáo
* Phần A dùng tham số **mặc định** cho baseline (paper còn so với baseline đã tune 4 giờ), 3 fold
  thay vì 10 lần lặp, và giới hạn ≤ 3000 dòng để vừa ngân sách T4.
* Phần B-1 là kỹ thuật đã biết (fine-tune, ensemble), đóng góp nằm ở việc đo trên cùng giao thức.
* Phần B-2 dùng mô hình và prior thu nhỏ; kết luận là về **hướng cải tiến ở cùng chi phí**, không
  khẳng định vượt TabPFN v2 gốc. Nếu một thành phần (ví dụ DyT) làm kết quả xấu đi, hãy báo cáo
  đúng như vậy; ablation được thiết kế để thấy rõ từng thành phần.

## 8. Tài liệu tham khảo
* Hollmann et al. *Accurate predictions on small data with a tabular foundation model.* Nature 637, 2025.
* Qiu et al. *Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.* NeurIPS 2025.
* Zhu, Chen, He, LeCun, Liu. *Transformers without Normalization.* CVPR 2025.
* Jordan et al. *Muon: An optimizer for hidden layers in neural networks.* 2024; Liu et al. *Muon is Scalable for LLM Training.* 2025.
* Rubachev et al. *On Finetuning Tabular Foundation Models.* 2025.
* *nanoTabPFN: A Lightweight and Educational Reimplementation of TabPFN.* arXiv:2511.03634, 2025 (github.com/automl/nanoTabPFN).
* Caruana et al. *Ensemble Selection from Libraries of Models.* ICML 2004.
* Bischl et al. *OpenML Benchmarking Suites.* NeurIPS Datasets & Benchmarks 2021.
