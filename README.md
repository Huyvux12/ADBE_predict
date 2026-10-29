# REL301m Assignment 2 — Adobe (ADBE), CEM policy search

Học tăng cường để tối đa hóa tổng tài sản khi giao dịch cổ phiếu ADBE, vốn đầu 100.000 USD.

| Tập | Khoảng | Tổng tài sản | Kết thúc |
|---|---|---:|---|
| Train 8 năm | [08/09/1990, 08/09/1998) | 1.286.607,16 USD | thắng |
| Test 2 năm (load/predict, không train lại) | [08/09/1998, 08/09/2000) | 1.017.252,01 USD | thắng |

Báo cáo: `Bao_cao_REL301m_ADBE.html` (nguồn `.md`, sinh tự động). Điền họ tên, MSSV, lớp trước khi nộp.

## Chạy

```bash
pip install -r requirements.txt
python main.py all        # evaluate -> replay -> verify -> report, dùng model đã lưu
python -m pytest          # unit test luật giao dịch
python main.py train      # huấn luyện lại từ đầu, ghi đè outputs/*/models (~10 phút CPU)
```

Từng bước: `python main.py {train|evaluate|replay|verify|report} [--track consecutive|gap|all] [--quick] [--output DIR]`.

Notebook:

- `ADBE_REL301m_end_to_end.ipynb`: toàn bộ quy trình dữ liệu → môi trường → huấn luyện CEM → đối chiếu model đã nộp → đánh giá → test load/predict → replay → kiểm chứng. Kết quả ghi vào `outputs/end_to_end/`; khoảng 3–4 phút, hoặc đặt `QUICK = True` để chạy thử.
- `ADBE_REL301m.ipynb`: bản ngắn, chỉ load/predict trên test và kiểm chứng.

Mở bằng Jupyter/VS Code (cần `pip install ipykernel`), hoặc chạy không cần Jupyter và lưu output vào notebook: `python main.py notebook` (bản end-to-end); bản ngắn được chạy lại bởi `python main.py report`.

## Cấu trúc

```text
config.json              Luật giao dịch ADBE, chia train/test, ngân sách CEM, cửa sổ replay
main.py                  CLI
adbe_rl/
  config.py              Đọc config, đường dẫn
  market.py              Dữ liệu OHLC, 12 đặc trưng nhân quả, chia train/test
  broker.py              Làm tròn lệnh, phí, phạt, thắng/thua (vector hóa, dùng chung)
  env.py                 TradingEnv: observation 18 chiều, 7 action, reward log-return
  policies.py            CEM chỉ báo, CEM theo thời gian, baseline, lưu/nạp model
  simulate.py            Mô phỏng cả population song song
  cem.py                 Cross-Entropy Method
  training.py            Pipeline huấn luyện một track
  evaluation.py          Đánh giá model đã lưu, replay các giai đoạn khác
  verify.py              Kiểm chứng log, kế toán, chia dữ liệu, tính nhân quả, replay
  report.py              Hình, báo cáo Markdown/HTML
  notebook.py            Chạy notebook không cần Jupyter (lưu text, bảng, hình)
  report_template.md     Nội dung báo cáo
tests/                   Unit test (pytest)
data/                    ADBE daily 1986–2026 (Yahoo Finance, giá điều chỉnh split)
outputs/
  consecutive/           Bản chính: models/, histories/, learning/, summary.csv, splits.json
  gap/                   Đối chiếu: train 1986–1994, test 1998–2000
  replay/                Model chính trên các cửa sổ 2 năm 2020–2024
  figures/               Hình trong báo cáo
  end_to_end/            Kết quả của notebook end-to-end
  verification.json      Kết quả kiểm chứng
```

## Quy ước

- Giá mua = High phiên trước. Giá bán = High phiên hiện tại (kịch bản `sell_high`) hoặc Close (`sell_close`). Định giá theo High.
- Số cổ phiếu nguyên, luôn trong BUY 35–73% tiền mặt và SELL 13–82% cổ phiếu; lệnh không hợp lệ bị mask và không reset bộ đếm phạt.
- Phí 4% hai chiều. Phạt 4.376 USD sau 21 phiên liên tiếp không giao dịch.
- Thua khi tổng tài sản < 1.000 USD hoặc tiền mặt ≤ −5.000 USD; thắng khi tổng tài sản ≥ 1.000.000 USD. Episode dừng ngay.
- Mỗi episode test bắt đầu lại từ 100.000 USD; observation chỉ dùng giá đến phiên trước.
