# Chạy notebook end-to-end trên Kaggle

Notebook `ADBE_REL301m_end_to_end.ipynb` cần một dataset Kaggle chứa code `adbe_rl/`, `config.json`, dữ liệu ADBE và 4 model đã nộp. Notebook tự tìm code trong `/kaggle/input` và ghi kết quả vào `/kaggle/working/outputs`. Chỉ cần CPU, không cần GPU hay Internet; thời gian chạy khoảng 5 phút.

## 1. Đóng gói

```bash
python kaggle/stage.py
```

Lệnh này tạo hai thư mục:

- `kaggle/build/dataset/`: code, dữ liệu, model và `dataset-metadata.json`.
- `kaggle/build/kernel/`: notebook đã xóa output và `kernel-metadata.json`.

Đổi `huyv12345` trong hai file metadata nếu dùng tài khoản khác.

## 2. Tải lên (mặc định private)

PowerShell, chạy tại thư mục project:

```powershell
kaggle datasets create -p "$PWD\kaggle\build\dataset" -r zip               # lần đầu
kaggle datasets version -p "$PWD\kaggle\build\dataset" -r zip -m "update"   # các lần sau
kaggle kernels push -p "$PWD\kaggle\build\kernel" -t 3600
kaggle kernels status huyv12345/adbe-rel301m-end-to-end
kaggle kernels output huyv12345/adbe-rel301m-end-to-end -p "$PWD\kaggle\run"
```

`-r zip` giữ nguyên cấu trúc thư mục (`adbe_rl/`, `data/`, `outputs/`) trong dataset.

## 3. Chạy bằng giao diện web

1. Tạo dataset mới, kéo thả toàn bộ nội dung `kaggle/build/dataset/`.
2. Tạo notebook mới, chọn **File → Import Notebook** và chọn `ADBE_REL301m_end_to_end.ipynb`.
3. **Add Input** → dataset vừa tạo. Accelerator: None (CPU).
4. **Run All**, hoặc **Save Version → Save & Run All** để lưu lại output.

## Lần chạy đã kiểm chứng

- Dataset (private): https://www.kaggle.com/datasets/huyv12345/adbe-rel301m-cem. Đã tải lại và đối chiếu SHA-256: đủ 24 file, không thiếu, không thừa.
- Notebook (private, CPU): https://www.kaggle.com/code/huyv12345/adbe-rel301m-end-to-end, version 2.
  - Huấn luyện mất 289 giây.
  - Tham số 4 model trùng hoàn toàn với model đã nộp.
  - Test đạt 1,017,252.01 USD, thắng ngày 2000-07-11.
  - Kiểm chứng: PASS.
- Output và log của lần chạy này nằm trong `kaggle/run/`.
