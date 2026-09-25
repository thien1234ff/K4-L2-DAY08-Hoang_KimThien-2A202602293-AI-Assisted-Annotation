# Vì sao chọn lô này?

## 1. Đề xuất top 5 frame nếu ngân sách chỉ đủ rà 5 ảnh

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, nếu ngân sách rà nhãn bị giới hạn ở 5 ảnh, ta không thể chọn máy móc theo top 5 điểm cao nhất vì cụm thời gian từ 130s đến 152s chiếm đa số và chứa nhiều ảnh gần trùng (near-duplicate) với lượng box rất lớn (làm đội chi phí gán nhãn). Ta ưu tiên phân bổ đều theo trục thời gian, cân bằng giữa độ bất định cao và chi phí rà:

1. **`frame_0182.jpg`** (Thứ tự 1, Điểm: 0.9591, t = 72.8s): Đứng đầu toàn bộ pool với U = 0.9182, A = 1.0 (có 18 box mơ hồ). Tổng số box là 28, chi phí gán nhãn vừa phải, đại diện cho phân đoạn giữa video khi xe bắt đầu chuyển làn.
2. **`frame_0099.jpg`** (Thứ tự 8, Điểm: 0.9063, t = 39.6s): Đại diện cho đoạn đầu video (t ~ 39.6s). Có độ bất định U = 0.9460 cực kỳ cao (cao hơn cả top 1–7), 29 box và 14 box mơ hồ. Việc chọn frame này bổ sung phân phối góc chụp xéo ở đoạn đầu video mà không bị trùng thời gian với các frame phía sau.
3. **`frame_0326.jpg`** (Thứ tự 4, Điểm: 0.9155, t = 130.4s): Đại diện cho đoạn mật độ xe dày đặc (t ~ 130s), U = 0.9310, A = 0.8333, 39 box. Ta chọn frame này và **loại bỏ `frame_0331.jpg`** (Rank 5, t = 132.4s) vì 0331 chỉ cách 0326 đúng 2.0 giây, bối cảnh giao thông gần như trùng lặp và 0331 có tới 47 box (làm tăng chi phí gán nhãn quá mức mà thông tin trùng lặp cao).
4. **`frame_0270.jpg`** (Thứ tự 13, Điểm: 0.8878, t = 108.0s): Khỏa lấp khoảng trống dữ liệu lớn giữa giây 90 và giây 125, U = 0.9089, A = 0.7778, 35 box. Giúp mô hình bao quát trường hợp xe tải và xe con đan xen ở làn giữa.
5. **`frame_0380.jpg`** (Thứ tự 3, Điểm: 0.9170, t = 152.0s): Đại diện chốt cho đoạn cuối video (t > 150s), U = 0.9340, A = 0.8333, 40 box. Frame này có nhiều xe ở xa chỉ nhìn thấy cụm đèn đỏ hậu, giúp mô hình cải thiện khả năng phân biệt xe xa với nhiễu ánh sáng đường.

## 2. Ba frame thuộc lô 12 ảnh model chọn và bằng chứng phân tích

- **`frame_0182.jpg` (Rank 1)**: Score = 0.9591. Bằng chứng CSV: U = 0.9182, A = 1.0 (18 box mơ hồ / 28 box). Trên contact sheet `outputs/selection_round1.jpg`, frame này ghi nhận nhiều xe đi sát dải phân cách và xe ở làn xa bị chìm vào bóng tối, độ tương phản rất thấp khiến model dao động mạnh về ranh giới box.
- **`frame_0099.jpg` (Rank 8)**: Score = 0.9063. Bằng chứng CSV: U = 0.9460 (thuộc nhóm bất định cao nhất tập chọn), A = 0.7778 (14 box mơ hồ / 29 box). Trên contact sheet, đây là cảnh xe đông chạy chếch góc quay, có xe tối màu bị che một phần và ánh đèn xe rọi lóa mặt đường dẫn đến nguy cơ bỏ sót hoặc vẽ sai box rất cao.
- **`frame_0331.jpg` (Rank 5)**: Score = 0.9154. Bằng chứng CSV: n_boxes = 47, n_ambiguous = 18 (A = 1.0), U = 0.8308. Đây là frame có mật độ xe dày đặc nhất trong lô được chọn. Trên contact sheet, hàng loạt xe nối đuôi nhau sát nút, các vệt đèn hậu đỏ đan cài vào nhau làm xuất hiện nhiều box chồng lấn hoặc model gộp hai xe thành một.

## 3. Frame điểm cao không được chọn và lý do loại bỏ

- **`frame_0372.jpg` (Rank 6, Score = 0.9101, U = 0.9202, A = 0.8333, n_boxes = 42, t = 148.8s)**:
  - **Lý do không chọn**: Thuật toán áp dụng cơ chế greedy với ràng buộc khoảng cách thời gian tối thiểu `MIN_GAP_S = 2.0s`. `frame_0372.jpg` (t = 148.8s) xuất hiện chỉ sau `frame_0369.jpg` (Rank 2, t = 147.6s, đã được chọn) đúng **1.2 giây** (< 2.0s). Do tốc độ xe trên cao tốc không đổi quá nhanh trong 1.2s, hai frame này là ảnh gần trùng (near-duplicate) về bố cục, vị trí xe và góc chiếu sáng. Việc thuật toán bỏ qua frame này giúp tối ưu ngân sách rà nhãn, tránh nạp trùng dữ liệu gây lãng phí công sức và nguy cơ overfit.

## 4. Điều phép chọn này chưa chứng minh về chất lượng mô hình

Phép chọn mẫu dựa trên độ bất định (`U`), số box mơ hồ (`A`) và phân bố thời gian (`D`) chỉ chứng minh được rằng **mô hình hiện tại đang thiếu tự tin hoặc bối rối nhất** ở những frame này:
1. **Không đảm bảo tăng hiệu năng (AP50)**: Một frame có độ bất định cao có thể chứa nhiều nhiễu thị giác (lóa đèn, mờ nhòe chuyển động) mà việc gán nhãn có thể khiến mô hình học phải tín hiệu nhiễu thay vì đặc trưng khái quát.
2. **Lệch phân phối so với tập test**: Việc chọn 12 ảnh tối ưu trong pool không đồng nghĩa với việc cải thiện đúng các điểm yếu trên 20 ảnh của tập test. Nếu lỗi trong 12 ảnh train không lặp lại trong tập test, AP50 kiểm thử sẽ không tăng, thậm chí có thể giảm do kích thước tập train quá nhỏ (12 ảnh) gây biến động trọng số.
3. **Giới hạn của nhãn tham chiếu**: Bộ test dùng nhãn tham chiếu do mô hình tự động gán chứ chưa được chuyên gia rà soát từng box, do đó AP50 chỉ phản ánh độ khớp với nhãn tham chiếu đó chứ chưa chứng minh năng lực phát hiện thực địa.
