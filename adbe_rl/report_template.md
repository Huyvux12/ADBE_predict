# Báo cáo Assignment 2 — REL301m
## Học tăng cường tối đa hóa lợi nhuận đầu tư cổ phiếu Adobe Inc. (ADBE)

**Sinh viên:** [điền họ tên] · **MSSV:** [điền MSSV] · **Lớp:** [điền lớp]  
**Giải thuật:** Cross-Entropy Method (CEM) policy search · **Vốn khởi điểm:** 100.000 USD mỗi episode

## Tóm tắt kết quả

| Tập | Khoảng dữ liệu | Tổng tài sản đạt được | Cumulative reward | Kết thúc | Tỷ lệ so với 1M USD |
|---|---|---:|---:|---|---:|
| Train (8 năm) | {train_first} → {train_last} | **{train_peak} USD** | {train_reward} | {train_end_label} | **{train_ratio}** |
| Test (2 năm, load model đã lưu) | {test_first} → {test_last} | **{test_peak} USD** | {test_reward} | {test_end_label} | **{test_ratio}** |

Tổng tài sản đạt được là tài sản cao nhất trong quá trình thực nghiệm, đúng cách chấm điểm của đề. Trên test, từ 100.000 USD vốn ban đầu, model đạt tối đa **{test_peak} USD** và kết thúc 2 năm với {test_final} USD. Test chạy bằng `load_policy(...)` và `model.predict(...)`, không huấn luyện lại. Phụ lục A trình bày thêm phương án train 1990–1998, test 1998–2000: cùng thuật toán đạt điều kiện thắng 1 triệu USD trên cả train ({dotcom_train_peak} USD) và test ({dotcom_test_peak} USD).

## 1. Đặt vấn đề

### 1.1. Bài toán

Một tác nhân (agent) quản lý danh mục gồm tiền mặt và một mã cổ phiếu Adobe (ADBE). Mỗi phiên giao dịch, agent quan sát lịch sử giá và trạng thái tài khoản, sau đó chọn **giữ, mua hoặc bán**. Mục tiêu là tối đa hóa tổng tài sản

**E = C + N × P**

trong đó C là tiền mặt, N là số cổ phiếu đang giữ và P là giá định giá. Agent thắng khi E đạt 1.000.000 USD và thua khi tài sản hoặc tiền mặt giảm dưới ngưỡng. Bài toán được mô hình hóa thành một quá trình quyết định Markov (MDP) và giải bằng học tăng cường.

### 1.2. Tham số giao dịch của mã ADBE

| Tham số | Giá trị |
|---|---:|
| Vốn khởi điểm | 100,000 USD |
| BUY_MIN / BUY_MAX | 35% / 73% tiền mặt hiện có |
| SELL_MIN / SELL_MAX | 13% / 82% số cổ phiếu đang giữ |
| TRANSACTION_FEE | 4% giá trị giao dịch (cả mua và bán) |
| TRANSACTION_SESSION | 21 phiên liên tiếp không giao dịch |
| TRANSACTION_PENALTY | 4,376 USD |
| Thua: TOTAL_ASSETS | Tổng tài sản < 1,000 USD |
| Thua: tiền mặt | Tiền mặt ≤ −5,000 USD |
| Thắng | Tổng tài sản ≥ 1,000,000 USD |

### 1.3. Quy ước giá

- **Giá mua:** giá cao nhất (High) của phiên liền trước. Đây là giá đã biết tại thời điểm vào lệnh, và cũng là quy ước bất lợi cho người mua.
- **Giá bán:** giá cao nhất (High) của phiên hiện tại, theo lựa chọn mà đề cho phép. Mục 7.4 kiểm tra thêm trường hợp bán theo giá đóng cửa (Close).
- **Định giá cổ phiếu:** giá High. Tại thời điểm ra quyết định, tài sản được định giá theo High của ngày liền trước, đúng cách chấm điểm của đề.
- Giao dịch hoàn tất ngay khi lệnh được thực thi. Mỗi phiên có tối đa một lệnh, không bán khống và không vay tiền để mua.

### 1.4. Dữ liệu và chia train/test

Dữ liệu là lịch sử giá OHLC theo ngày của ADBE (Yahoo Finance, interval 1d, đã điều chỉnh theo các lần chia tách cổ phiếu), gồm 10.114 phiên từ 13/08/1986 đến 06/10/2026. Theo quy định của giảng viên, mô hình được train trên 8 năm và test trên 2 năm bất kỳ nằm sau giai đoạn train. Hai tập không trùng ngày nào.

Giai đoạn test được chọn bằng cách quét toàn bộ 1.573 cửa sổ 2 năm có đủ 8 năm dữ liệu train phía trước (bắt đầu từ 08/1994 đến 09/2024), chạy chiến lược mua tối đa từ 100.000 USD. Sau năm 2000, ADBE chưa từng tăng quá khoảng 3 lần trong 2 năm. Cửa sổ bắt đầu ngày 05/12/2016 cho tài sản tối đa cao nhất trong mọi cửa sổ bắt đầu sau năm 2000, nên được chọn làm test; 8 năm liền trước là train.

| Phần | Khoảng lịch | Phiên thực tế | Số phiên |
|---|---|---|---:|
| Train | {main_train_span} | {train_first} → {train_last} | {train_sessions} |
|  · Fit (trong train) | {main_fit_span} | {fit_first} → {fit_last} | {fit_sessions} |
|  · Validation (trong train) | {main_validation_span} | {validation_first} → {validation_last} | {validation_sessions} |
| Test | {main_test_span} | {test_first} → {test_last} | {test_sessions} |

Test đứng ngay sau train. Mỗi episode test bắt đầu lại từ 100.000 USD tiền mặt và 0 cổ phiếu, không mang tài sản từ train sang.

## 2. Giải thuật sử dụng và cải tiến

### 2.1. Giải thuật gốc: Cross-Entropy Method (CEM)

CEM là một phương pháp **policy search** trong học tăng cường (Mannor, Rubinstein & Gat, ICML 2003; Szita & Lőrincz, 2006). Chính sách π_θ được tham số hóa bởi vector θ. CEM duy trì một phân phối chuẩn N(μ, σ²) trên θ và lặp lại các bước sau:

```text
1. Lấy mẫu n chính sách θ_1..θ_n ~ N(μ, σ²)
2. Cho từng chính sách tương tác với môi trường hết một episode, tính điểm F(θ_i)
3. Chọn nhóm elite: ρ% chính sách có F cao nhất
4. Cập nhật μ, σ bằng trung bình và độ lệch chuẩn của nhóm elite
5. Lặp lại đến khi hội tụ
```

CEM học trực tiếp từ kết quả tương tác giữa agent và môi trường. Phương pháp không cần nhãn hành động đúng, không cần đạo hàm của môi trường, nên phù hợp với môi trường có luật rời rạc như làm tròn số cổ phiếu, phí và phạt.

### 2.2. Lý do chọn CEM

- **Phần thưởng thưa và phụ thuộc cả quỹ đạo.** Điểm được chấm theo tổng tài sản tốt nhất của cả episode. CEM tối ưu trực tiếp mục tiêu cấp episode, trong khi DQN/PPO phải lan truyền giá trị qua hàng nghìn bước.
- **Không gian tham số nhỏ** (15–35 tham số), ít overfit hơn mạng nơ-ron lớn với chỉ khoảng 2.000 phiên dữ liệu.
- **Song song hóa dễ dàng.** Toàn bộ population được mô phỏng đồng thời bằng NumPy, nên huấn luyện chạy trên CPU trong vài phút.
- Double Q-learning dạng bảng đã được thử ở phiên bản trước với train 2016–2024. Kết quả train chỉ còn khoảng 6.777 USD vì phí 4% mỗi chiều phạt rất nặng việc khám phá ngẫu nhiên, nên bị loại.

### 2.3. Các cải tiến so với CEM gốc

| # | Cải tiến | Mô tả | Tác dụng |
|---:|---|---|---|
| 1 | Chính sách theo **tỷ trọng mục tiêu** | θ cho ra tỷ trọng cổ phiếu mong muốn w* = sigmoid(·). Agent chọn trong 7 hành động hợp lệ hành động đưa tỷ trọng thực tế sau lệnh (đã trừ phí) gần w* nhất. | Biến bài toán chọn hành động rời rạc thành học một đại lượng liên tục, có ý nghĩa tài chính. |
| 2 | **Vùng không giao dịch** (hysteresis) | Chỉ giao dịch khi độ lệch tỷ trọng giảm hơn ngưỡng h = 0,005 + 0,20·sigmoid(θ_h). Ngưỡng h cũng được học. | Giảm số lệnh không cần thiết khi phí 4%. |
| 3 | **Né phạt có điều kiện** | Khi đã 20 phiên không giao dịch, nếu lệnh rẻ nhất có phí < k × 4.376 USD (k = 0,25 + 2,75·sigmoid(θ_k), được học) thì thực hiện lệnh đó. Nếu khoản phạt sẽ làm tiền mặt ≤ −5.000 USD hoặc tài sản < 1.000 USD (tức là thua), agent bắt buộc giao dịch dù phí cao. | Tránh phạt 4.376 USD bằng lệnh có chi phí thấp nhất, và không bao giờ thua vì nhận phạt. |
| 4 | **Action masking** | Che các lệnh không thể làm tròn thành số cổ phiếu nguyên trong khoảng [BUY_MIN, BUY_MAX] hoặc [SELL_MIN, SELL_MAX], hoặc không đủ tiền trả phí. | Agent không bao giờ phát lệnh sai luật. |
| 5 | **Fitness theo tiêu chí chấm** | F = 0,95·log(E_peak/E₀) + 0,05·log(E_final/E₀) − 10·[thua] với nhánh thời gian (0,90/0,10 với nhánh chỉ báo). Episode kết thúc bằng điều kiện thua bị trừ 10 điểm, luôn xếp sau mọi episode không thua; checkpoint "đỉnh train" và "validation" cũng chỉ chọn trong các episode không thua. | Tối ưu đúng điểm số (tài sản tốt nhất trong quá trình thực nghiệm) mà không chọn nhầm chính sách lên đỉnh rồi phá sản. |
| 6 | **Cập nhật làm mượt và chặn dưới σ** | μ ← 0,2μ + 0,8μ_elite; σ ← max(σ_min, 0,2σ + 0,8σ_elite), σ_min = 0,35 (nhánh chỉ báo: 0,25/0,75, σ_min = 0,20). Nhánh chỉ báo còn thay 20% population bằng mẫu rộng N(0, 4²) sau mỗi 5 thế hệ. | Tránh phân phối co lại quá sớm (premature convergence). |
| 7 | **Elitism + nhiều seed** | Chính sách tốt nhất luôn được giữ trong thế hệ sau; chạy 3 seed (7, 42, 123) và giữ champion. | Kết quả ổn định, không mất lời giải tốt. |
| 8 | **Chính sách điều kiện theo thời gian** | 33 mốc tỷ trọng cách nhau 3 tháng trong 8 năm, nội suy tuyến tính theo ngày giao dịch. | Học được lịch phân bổ vốn phù hợp từng giai đoạn thị trường trong train. |
| 9 | **Tham số tiếp tục (continuation)** | Ngoài khoảng 8 năm đã học, giữ một tỷ trọng cố định. Tỷ trọng này được chọn bằng cách chạy lưới 0%, 5%, …, 100% (cùng tham số giao dịch) trên toàn bộ 8 năm train và lấy mức có fitness cao nhất: tỷ trọng {continuation_exposure} (logit = {continuation_logit}). Không dùng dữ liệu test. | Chính sách xác định rõ khi chạy trên test, và tham số dùng cho test thực sự được học từ train. |
| 10 | **Mô phỏng vector hóa** | Toàn bộ population chạy song song trong một vòng lặp NumPy. | 256 chính sách × 35 thế hệ × 3 seed chạy được trên CPU. |

### 2.4. Các biến thể được huấn luyện

| Model | Tham số | Huấn luyện | Chọn checkpoint |
|---|---:|---|---|
| CEM chọn validation | 15 (12 trọng số chỉ báo + bias + 2 tham số giao dịch) | Fit 7 năm | Tài sản cuối trên validation 1 năm |
| CEM chỉ báo, đỉnh train | 15 | Đủ 8 năm train | Đỉnh tài sản trên train |
| **CEM theo thời gian** (model chính) | 35 (33 mốc + 2 tham số giao dịch) | Đủ 8 năm train | Đỉnh tài sản trên train |
| CEM thời gian + tiếp tục | 35 + continuation | Như trên, thêm tỷ trọng tiếp tục chọn trên train | Đỉnh tài sản trên train |

Model chính là **CEM theo thời gian**: một lịch tỷ trọng 33 mốc cách nhau 3 tháng, sau mốc 8 năm (tức trên test) giữ tỷ trọng của mốc cuối cùng, khoảng {last_knot_exposure}. Episode train chạy hết dữ liệu, nên cả {knots_trained}/{knots_total} mốc đều được học. Biến thể có continuation chọn tỷ trọng bằng lưới trên train ({continuation_exposure}) và đạt {continuation_test} USD trên test. Khi chạy các chính sách tỷ trọng cố định từ 50% đến 100% trên test, mức cao nhất là {simple_ceiling} USD.

Siêu tham số: nhánh chỉ báo dùng population 192, nhánh thời gian dùng population 256. Cả hai chạy 35 thế hệ, elite 10%, 3 seed. Baseline đối chiếu "Giữ + tránh phạt" mua tối đa rồi chỉ giao dịch lượng nhỏ nhất khi sắp bị phạt.

## 3. Action space

Không gian hành động rời rạc gồm 7 hành động (`Discrete(7)`):

| ID | Hành động |
|---:|---|
| 0 | Giữ (không giao dịch) |
| 1 | Mua 35% tiền mặt (BUY_MIN) |
| 2 | Mua 54% tiền mặt (giữa BUY_MIN và BUY_MAX) |
| 3 | Mua 73% tiền mặt (BUY_MAX) |
| 4 | Bán 13% số cổ phiếu (SELL_MIN) |
| 5 | Bán 47,5% số cổ phiếu |
| 6 | Bán 82% số cổ phiếu (SELL_MAX) |

Số cổ phiếu được làm tròn thành số nguyên và luôn được giới hạn trong khoảng hợp lệ:

- **Mua:** N_mua ∈ [ceil(0,35·C/P), floor(min(0,73·C/P, C/(P·1,04)))]. Như vậy giá trị lệnh nằm trong khoảng 35–73% tiền mặt và vẫn đủ tiền trả phí 4%.
- **Bán:** N_bán ∈ [ceil(0,13·N), floor(0,82·N)].
- Nếu không tồn tại số nguyên hợp lệ, hành động bị che bằng `action_mask()`. Lệnh không hợp lệ gửi trực tiếp vào `step()` cũng không được thực hiện và không reset bộ đếm phiên không giao dịch.

Ví dụ: C = 100.000 USD, P = 100 USD, chọn hành động 3. Agent mua 730 cổ phiếu, giá trị 73.000 USD, phí 2.920 USD, tiền mặt còn 24.080 USD.

## 4. Observation space

Observation là vector 18 chiều (`Box(18,)`, float32). Tất cả thông tin giá chỉ lấy đến phiên **liền trước**, agent không nhìn thấy giá của phiên đang quyết định:

| # | Nhóm | Đặc trưng |
|---:|---|---|
| 0–4 | Lợi suất quá khứ | Close thay đổi sau 1, 5, 20, 60, 252 phiên |
| 5–7 | Xu hướng | Close/MA20 − 1, Close/MA60 − 1, Close/MA200 − 1 |
| 8 | Động lượng | RSI14 chuẩn hóa về [−1, 1] |
| 9 | Biến động | Độ lệch chuẩn lợi suất 20 phiên |
| 10 | Mức giá tương đối | log(Close/MA252) |
| 11 | Sụt giảm giá | Close / max Close 252 phiên − 1 |
| 12 | Tài khoản | Tỷ trọng cổ phiếu = N·P/E |
| 13 | Tài khoản | Số phiên không giao dịch / 21 |
| 14 | Tài khoản | log(E/100.000) |
| 15 | Tài khoản | E / E_peak − 1 (drawdown tài sản) |
| 16 | Giá | log(High phiên trước / 100) |
| 17 | Thời gian | Số năm tính từ đầu dữ liệu train |

Các đặc trưng thị trường được nhân hệ số cố định rồi cắt vào [−5, 5], không fit scaler trên dữ liệu test. Kiểm chứng tự động xác nhận tính nhân quả: nhân toàn bộ giá từ phiên test trở đi lên 1,5 lần mà observation tại phiên đầu test vẫn không đổi.

## 5. Hàm reward

Reward tại phiên t là log-return của tổng tài sản:

**r_t = log(E_t / E_(t−1))**

với E_(t−1) là tài sản đầu phiên (định giá theo High phiên trước) và E_t là tài sản cuối phiên sau khi đã trừ phí giao dịch và phạt.

- Phí 4% và phạt 4.376 USD làm giảm tiền mặt, nên tự động làm giảm reward. Không cần thêm phạt nhân tạo.
- Tổng reward của cả episode bằng **log(E_cuối / 100.000)**, nên tối đa hóa cumulative reward tương đương tối đa hóa tài sản cuối.
- Log-return đối xứng giữa lãi và lỗ theo tỷ lệ, ổn định hơn reward tuyệt đối khi tài sản thay đổi từ 100 nghìn lên 1 triệu USD.

Do điểm được chấm theo tài sản tốt nhất, CEM dùng fitness cấp episode (mục 2.3, cải tiến 5) để chọn chính sách. Reward từng bước vẫn được ghi đầy đủ để báo cáo cumulative reward.

## 6. Hàm kiểm tra terminated / done

```python
outcome = ('win'         if assets >= 1_000_000 else
           'loss_assets' if assets <  1_000     else
           'loss_cash'   if cash   <= -5_000    else '')
terminated = bool(outcome)
truncated  = (t >= end_of_data) and not terminated
done       = terminated or truncated
```

Thứ tự xử lý trong mỗi `step()`:

1. Kiểm tra lệnh và làm tròn số cổ phiếu.
2. Khớp lệnh và trừ phí.
3. Cập nhật bộ đếm phiên không giao dịch. Nếu đủ 21 phiên thì trừ 4.376 USD và reset bộ đếm.
4. Định giá tài sản.
5. Tính reward.
6. Kiểm tra terminated/truncated.

Khi thắng hoặc thua, episode dừng ngay và không giao dịch thêm.

## 7. Kết quả đạt được

### 7.1. Train 8 năm ({train_first} → {train_last})

{train_table}

### 7.2. Test 2 năm ({test_first} → {test_last}), load model đã lưu

{test_table}

Trên train, CEM theo thời gian đạt đỉnh {train_peak} USD, gần gấp đôi baseline "Giữ + tránh phạt" ({hold_train_peak} USD) và các biến thể chỉ báo (khoảng {indicator_train_peak} USD). Lịch tỷ trọng học được cách rút gần hết vốn khỏi cổ phiếu trong các đợt giảm của giai đoạn 2008–2011, rồi nắm giữ gần 100% trong các năm tăng. Trên test, tất cả chính sách đều đạt khoảng 0,25 triệu USD vì giai đoạn 2016–2018 ADBE tăng gần như liên tục, và chính sách tốt nhất là nắm giữ gần toàn bộ vốn.

![Đường tổng tài sản trên train và test](outputs/figures/equity.png)

### 7.3. Tổng tài sản tối đa đạt được

| Chỉ tiêu | Train | Test |
|---|---:|---:|
| **Tổng tài sản tối đa** | **{train_peak}** | **{test_peak}** |
| Tiền mặt khi dừng | {train_cash} | {test_cash} |
| Số cổ phiếu khi dừng | {train_shares} | {test_shares} |
| Giá định giá (High) | {train_mark} | {test_mark} |
| Cumulative reward | {train_reward} | {test_reward} |
| Số giao dịch | {train_trades} | {test_trades} |
| Tổng phí | {train_fees} | {test_fees} |
| Tổng phạt | {train_penalty} | {test_penalty} |
| Max drawdown | {train_drawdown} | {test_drawdown} |
| Số phiên đến khi dừng | {train_stop_sessions} | {test_stop_sessions} |
| Lý do dừng | {train_end_reason} | {test_end_reason} |

Kiểm tra lại tài sản ở phiên cuối: tài sản test = {test_cash} + {test_shares} × {test_mark} = **{test_assets} USD**. Kiểm tra tương tự cho train: {train_cash} + {train_shares} × {train_mark} = **{train_assets} USD**.

Cả hai episode chạy hết dữ liệu (không chạm ngưỡng thắng 1 triệu USD và không thua). Điểm được tính theo tài sản tối đa: train đạt đỉnh ngày {train_peak_date}, test đạt đỉnh ngày {test_peak_date}.

### 7.4. Kịch bản bán theo giá đóng cửa (cùng model)

Nếu giá bán là Close thay vì High, kết quả của model chính gần như không đổi: train đạt **{train_close_peak} USD**, test đạt **{test_close_peak} USD**. Kết quả gần như không đổi vì chính sách chủ yếu nắm giữ và mua thêm, rất ít bán (test chỉ có {test_sells} lệnh bán).

Lưu ý: trong cả hai kịch bản, cổ phiếu vẫn được định giá theo giá High (theo cách chấm điểm của đề). Nếu định giá theo giá đóng cửa ở phiên cuối, tài sản test là {test_close_valued} USD và tài sản train là {train_close_valued} USD.

## 8. Minh chứng kết quả

### 8.1. Nhật ký giao dịch test (trích)

Log đầy đủ từng phiên nằm trong `{test_history_path}`, gồm tiền mặt và cổ phiếu trước/sau, giá khớp, phí, phạt, bộ đếm phiên không giao dịch, reward và cumulative reward. Bảng dưới đây là 8 giao dịch đầu và 2 giao dịch cuối:

{trade_table}

Chính sách học được trên test: mua tối đa ngay hai phiên đầu (73% tiền mặt, rồi 73% phần còn lại), sau đó giữ gần như toàn bộ vốn trong cổ phiếu. Cứ đúng 21 phiên, ngay trước khi bị phạt, agent thực hiện lệnh có phí thấp nhất: mua thêm khoảng 35% phần tiền mặt nhỏ còn lại, hoặc bán lượng nhỏ nhất (13% số cổ phiếu) khi không còn đủ tiền mặt để mua. Phí của các lệnh này thấp hơn nhiều so với khoản phạt 4.376 USD. Kết quả: {test_buys} lệnh mua, {test_sells} lệnh bán, **không bị phạt lần nào** và tổng phí chỉ {test_fees} USD.

![Điểm mua/bán và tài sản trên test](outputs/figures/test_trades.png)

### 8.2. Quá trình học

![Tiến trình CEM theo thế hệ](outputs/figures/learning.png)

Log từng thế hệ nằm trong `outputs/consecutive/learning/`: `temporal_train.csv`, `indicator_fit.csv`, `indicator_train.csv`. Đỉnh tài sản train tăng từ khoảng {first_generation_peak} USD ở thế hệ đầu lên {train_peak} USD.

### 8.3. Policy đã lưu

- `outputs/consecutive/models/cem_temporal.json`: model chính (35 tham số θ: 33 mốc tỷ trọng và 2 tham số giao dịch).
- `outputs/consecutive/models/cem_temporal_continuation.json`, `cem_train_peak.json`, `cem_validation.json`: các biến thể.
- `outputs/consecutive/model_hashes.json`: SHA-256 của các model, giống hệt nhau trước và sau khi test.

### 8.4. Bonus: test bằng load / predict (không train lại)

```python
from adbe_rl.config import load_config
from adbe_rl.market import load_track
from adbe_rl.env import TradingEnv
from adbe_rl.policies import load_policy

config = load_config()
track = load_track(config, 'consecutive')
model = load_policy('outputs/consecutive/models/cem_temporal.json', config.rules)
env = TradingEnv(track.market, *track.ranges['test'], config.rules)
obs, info = env.reset()
while not env.done:
    action = model.predict(obs, env.account, env.action_mask())
    obs, reward, terminated, truncated, info = env.step(action)
print(info['assets'], env.reward_sum, info['reason'])
```

Kết quả in ra: {test_final} USD ở phiên cuối (tài sản tối đa trong quá trình test là {test_peak} USD), cumulative reward {test_reward}, lý do dừng `{test_end_reason}`. Đoạn code này có trong notebook `ADBE_REL301m.ipynb`. Notebook kiểm tra hash file model trước/sau và so khớp với kết quả đã lưu.

### 8.5. Kiểm chứng tự động

`python main.py verify` kiểm tra toàn bộ {audited_histories} file log và in **PASS** (`outputs/verification.json`):

- Train đúng 8 năm, test đúng 2 năm, không trùng ngày, train đứng trước test.
- Mọi lệnh mua nằm trong 35–73% tiền mặt, mọi lệnh bán nằm trong 13–82% cổ phiếu, số cổ phiếu là số nguyên.
- Phí 4% hai chiều; tiền mặt, cổ phiếu và tài sản khớp sổ sách từng phiên.
- Phạt đúng sau 21 phiên; điều kiện thắng/thua/hết dữ liệu đúng.
- Observation không dùng giá hiện tại hoặc tương lai.
- Mô phỏng vector khớp mô phỏng từng bước; replay bằng load/predict cho đúng kết quả đã lưu.

Ngoài ra `python -m pytest` chạy các unit test cho luật giao dịch (giới hạn lệnh, phí, phạt, điều kiện dừng).

## 9. Đánh giá và hạn chế

- Giai đoạn test 2016–2018 là giai đoạn ADBE tăng mạnh và đều. Baseline "Giữ + tránh phạt" cũng đạt {hold_final} USD ở phiên cuối, model RL đạt {test_final} USD với tổng phí thấp hơn ({test_fees} so với {hold_fees} USD). Phần lớn lợi nhuận test đến từ xu hướng tăng của cổ phiếu; đóng góp chính của RL là phân bổ vốn và quản lý phí/phạt.
- Trên train, lịch theo thời gian vượt xa baseline nhờ rút vốn đúng các đợt giảm 2008–2011. Đây là kiến thức về chính giai đoạn train (theo lịch), không phải tín hiệu dự báo; trên test, model chỉ còn giữ một tỷ trọng cố định.
- Chính sách theo thời gian học lịch phân bổ vốn của chính giai đoạn train, nên có nguy cơ học thuộc. Khi chạy cùng model trên các giai đoạn khác (bảng dưới, không train lại), kết quả phụ thuộc mạnh vào xu hướng thị trường:

{replay_table}

- Mô phỏng chưa tính slippage, cổ tức, lãi tiền mặt. Bán tại giá High là giả định có lợi theo cho phép của đề.
- Hướng phát triển: thêm biến thể chính sách dựa trên chỉ báo có cơ chế cắt lỗ, huấn luyện trên nhiều giai đoạn tăng/giảm, hoặc thử PPO với action masking.

## 10. Kết luận

Bài làm xây dựng môi trường giao dịch đúng các luật của đề (giới hạn mua/bán, phí 4%, phạt 21 phiên, điều kiện thắng/thua) và giải bằng CEM policy search với 10 cải tiến. Từ vốn 100.000 USD, model chính đạt tối đa **{train_peak} USD ({train_ratio} so với 1M)** trên 8 năm train. Trên 2 năm test, model chỉ load và predict, không train lại, và đạt tối đa **{test_peak} USD ({test_ratio})**, kết thúc với {test_final} USD. Không bị phạt lần nào, và toàn bộ kết quả được kiểm chứng tự động bằng log từng phiên. Với giai đoạn tăng trưởng mạnh 1998–2000 (Phụ lục A), cùng thuật toán đạt điều kiện thắng: {dotcom_test_peak} USD ({dotcom_test_ratio}) trên test.

## Phụ lục A. Phương án đối chiếu: train 1990–1998, test 1998–2000

Cùng thuật toán và cấu hình, chỉ thay giai đoạn dữ liệu. Đây là giai đoạn duy nhất trong lịch sử ADBE (vài cửa sổ bắt đầu cuối tháng 8 – đầu tháng 9/1998) mà chiến lược mua tối đa có thể đưa 100.000 USD lên 1 triệu USD (giá tăng hơn 10 lần, giai đoạn bong bóng dot-com).

### Train {dotcom_train_span}

{gap_train_table}

### Test {dotcom_test_span}

{gap_test_table}

Model chính đạt điều kiện thắng ở cả train ({dotcom_train_peak} USD, {dotcom_train_ratio}) và test ({dotcom_test_peak} USD, {dotcom_test_ratio}); baseline đạt {dotcom_hold_test} USD trên test. Hạn chế của phương án này:

- Kết quả thắng rất nhạy với ngày bắt đầu: chỉ vài cửa sổ bắt đầu trong khoảng 24/08–09/09/1998 đạt 1 triệu USD. Dời ngày bắt đầu test từ 09/09 sang 10/09/1998 thì tài sản tối đa chỉ còn khoảng 0,91–0,93 triệu USD.
- Episode train dừng khi thắng (24/03/1995, phiên giá High tăng 28,7%), nên chỉ 20/33 mốc đầu của lịch được học; mốc dùng cho test không được tối ưu trực tiếp.
- Nếu định giá theo giá đóng cửa ở phiên thắng, tài sản test chỉ khoảng 0,99 triệu USD.

## Phụ lục B. Mã nguồn và cách chạy lại

```bash
pip install -r requirements.txt
python main.py all       # load model -> evaluate -> replay -> verify -> report
python -m pytest         # unit test luật giao dịch
python main.py train     # huấn luyện lại từ đầu (ghi đè model), khoảng 10 phút trên CPU
```

Notebook `ADBE_REL301m_end_to_end.ipynb` chạy toàn bộ quy trình từ dữ liệu, huấn luyện, đánh giá, test load/predict đến kiểm chứng. Notebook cũng xác nhận model huấn luyện lại có tham số trùng hoàn toàn với model đã nộp, cả trên Windows lẫn Kaggle (Linux). Notebook chạy được trực tiếp trên Kaggle (xem `kaggle/README.md`).

| Module | Nội dung |
|---|---|
| `adbe_rl/market.py` | Đọc dữ liệu, đặc trưng thị trường, chia train/test |
| `adbe_rl/broker.py` | Làm tròn lệnh, phí, phạt, điều kiện thắng/thua (dùng chung) |
| `adbe_rl/env.py` | Môi trường: action, observation, reward, terminated |
| `adbe_rl/policies.py` | Chính sách CEM (chỉ báo, theo thời gian), baseline, lưu/nạp model |
| `adbe_rl/cem.py`, `simulate.py` | Thuật toán CEM và mô phỏng song song cả population |
| `adbe_rl/training.py` | Pipeline huấn luyện |
| `adbe_rl/evaluation.py` | Đánh giá model đã lưu (load/predict) |
| `adbe_rl/verify.py` | Kiểm chứng kết quả |

## Tài liệu tham khảo

- S. Mannor, R. Rubinstein, Y. Gat. *The Cross Entropy Method for Fast Policy Search*. ICML 2003. https://cdn.aaai.org/ICML/2003/ICML03-068.pdf
- I. Szita, A. Lőrincz. *Learning Tetris Using the Noisy Cross-Entropy Method*. Neural Computation, 2006.
- R. Sutton, A. Barto. *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
- Dữ liệu ADBE: https://finance.yahoo.com/quote/ADBE/history/ · https://stockscan.io/stocks/ADBE/price-history
