# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Hoàng Kim Thiện

Công cụ gán nhãn đã dùng: CVAT (chạy bằng Docker local trên máy cá nhân)

---

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (`pool`) và tập kiểm thử (`test set`) được trích xuất từ một video quay camera cố định trên cầu vượt đường cao tốc ban đêm. Do camera đứng yên và tốc độ lấy mẫu là 2.5 khung hình/giây (khoảng cách giữa hai frame liền kề chỉ là 0.4 giây), một chiếc xe lưu thông sẽ xuất hiện liên tục trong khung hình qua hàng chục frame liên tiếp. 

Nếu chia ngẫu nhiên (random split), cùng một chiếc xe (với cùng góc chiếu sáng, vệt đèn pha và vị trí thân xe) gần như chắc chắn sẽ xuất hiện ở cả tập huấn luyện và tập kiểm thử. Khi đó, mô hình chỉ đơn giản "học vẹt" và được đánh giá trên đúng những chiếc xe mà nó đã vừa nhìn thấy trong tập train. Điều này dẫn đến hiện tượng **rò rỉ dữ liệu (data leakage)** nghiêm trọng, làm số đo trên tập kiểm thử (đặc biệt là AP50 và Precision/Recall) bị **thổi phồng giả tạo (overly optimistic)**, không phản ánh đúng năng lực tổng quát hóa thực tế của mô hình trên các luồng xe mới.

Để giải quyết triệt để vấn đề này, dữ liệu được chia theo **trục thời gian (temporal split)** với cấu trúc:
- **Tập test (20 ảnh)**: Lấy mẫu tại 4 đoạn thời gian tách biệt quanh các mốc 20s, 60s, 100s và 140s (mỗi đoạn 5 ảnh cách nhau 1.2s).
- **Vùng đệm (buffer - 112 ảnh)**: Loại bỏ toàn bộ các ảnh nằm trong khoảng 4 giây trước và sau mỗi đoạn test, cùng các ảnh xen giữa. Nhờ vùng đệm này, ảnh pool gần nhất cũng cách ảnh test tối thiểu 4.4 giây (đủ để toàn bộ dòng xe cũ đi ra khỏi khung hình camera).
- **Tập pool (268 ảnh)**: Là các ảnh còn lại dùng cho quy trình học chủ động (Active Learning).

---

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng số đo vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào file kết quả `outputs/metrics_round0.json` và ảnh đối chiếu `outputs/compare_round0.jpg`:
- **Độ lệch so với nhãn tham chiếu**: Mô hình khởi đầu lạnh (YOLOv8n pretrained trên COCO, gộp 3 lớp car, bus, truck) đạt độ chính xác tương đối cao ở các xe lớn (P@0.25 = 0.925) nhưng bỏ sót rất nhiều xe (Recall tổng thể chỉ đạt 0.489, tức bỏ sót hơn một nửa số lượng xe: 206 False Negatives trên tổng số 403 box tham chiếu).
- **Độ phủ theo kích thước xe**:
  + Xe nhỏ (`R small`): Chỉ đạt **0.1818** (chỉ phát hiện được 12/66 xe). Các xe ở làn xa sát đường chân trời chỉ hiện lên dưới dạng các chấm sáng mờ nhạt bị mô hình bỏ qua gần như hoàn toàn.
  + Xe trung bình (`R medium`): Đạt **0.5473** (162/296 xe).
  + Xe lớn (`R large`): Đạt **0.5610** (23/41 xe).
- **Các trường hợp mô hình không khớp**:
  1. Xe tối màu di chuyển sát dải phân cách bên phải hoặc làn ngoài cùng thiếu đèn đường chiếu rọi trực tiếp.
  2. Xe bị che khuất một phần bởi thân xe khác hoặc xe chỉ nhìn thấy đuôi/đèn đỏ.
  3. Xe có vệt sáng đèn pha chiếu dài xuống mặt đường bị mô hình COCO vẽ box lan ra cả vệt sáng, hoặc nhầm biển báo phản quang/đèn đường thành xe.
- **Trường hợp cần rà lại nhãn tham chiếu**: Các xe ở rất xa sát mép trên đường chân trời có chiều cao box dưới 16 pixel hoặc các vùng tối mịt chỉ có hai đốm sáng lập lờ. Do nhãn tham chiếu được sinh tự động bởi một mô hình AI khác (chưa qua người rà soát từng box), nhiều trường hợp vệt phản chiếu đèn trên rào chắn bị nhãn tham chiếu gán là xe, khiến mô hình khởi đầu lạnh dù dự đoán đúng (bỏ qua nhiễu) nhưng vẫn bị tính là False Negative.

---

## 3. Chiến lược chọn mẫu

Thuật toán chọn lô ảnh học chủ động sử dụng hàm mục tiêu kết hợp:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$
Trong đó:
- $U$ (**Uncertainty** - độ bất định, trọng số $W_U = 0.5$): Đo lường mức độ thiếu tự tin của mô hình trên các box dự đoán (dựa trên entropy hoặc khoảng cách tới ngưỡng tin cậy 0.5). Box càng dao động quanh ngưỡng mập mờ thì $U$ càng cao.
- $A$ (**Ambiguity** - tỷ lệ box mơ hồ, trọng số $W_A = 0.3$): Chuẩn hóa số lượng box có confidence nằm trong vùng tranh chấp $[0.2, 0.7]$. Ảnh càng nhiều box mà AI đang phân vân thì $A$ càng cao.
- $D$ (**Diversity** - độ đa dạng thời gian, trọng số $W_D = 0.2$): Khuyến khích chọn các frame cách xa các frame đã chọn trong lô, tránh gom cụm dữ liệu.
- **Vai trò của `MIN_GAP_S = 2.0`**: Là tham số ràng buộc khoảng cách thời gian tối thiểu (2.0 giây) giữa hai frame bất kỳ được chọn trong cùng một lô. Do video có nền cố định và xe chạy liên tục, hai frame cách nhau dưới 2 giây có góc cảnh và dòng xe gần như trùng lặp (near-duplicate). Ràng buộc này đảm bảo thuật toán loại bỏ các frame trùng lặp dù điểm $U$ và $A$ rất cao, giúp phân bổ ngân sách gán nhãn trải rộng trên toàn bộ dòng thời gian.

**Đối chiếu các frame cụ thể (theo `reports/SELECTION.md` và `outputs/selection_round1.csv`)**:
1. **`frame_0182.jpg`** (Rank 1, Score: 0.9591, t = 72.8s): Được chọn vì có $U = 0.9182$ và $A = 1.0$ (18 box mơ hồ). Thuộc đoạn giữa video, bối cảnh ánh sáng phức tạp nhưng lượng box hợp lý (28 box), mang lại tỷ suất thông tin cao nhất trên chi phí rà nhãn.
2. **`frame_0099.jpg`** (Rank 8, Score: 0.9063, t = 39.6s): Được chọn vì có độ bất định $U = 0.9460$ cao vượt trội, đại diện độc lập cho vùng thời gian đầu video (39.6s) mà không bị trùng cụm với các frame sau.
3. **`frame_0326.jpg`** (Rank 4, Score: 0.9155, t = 130.4s): Được chọn làm đại diện cho cụm xe mật độ cao tại 130s. Ta chủ động ưu tiên frame này và loại bỏ **`frame_0331.jpg`** (Rank 5, t = 132.4s, 47 box) vì chỉ cách 2.0s nhưng có tới 47 box, gây đội chi phí gán nhãn mà dữ liệu lại trùng lặp.
4. **`frame_0372.jpg`** (Rank 6, Score: 0.9101, t = 148.8s): Dù điểm số thuộc top 6 nhưng **bị loại bỏ** hoàn toàn bởi cơ chế `MIN_GAP_S`, vì nó chỉ cách `frame_0369.jpg` (Rank 2, t = 147.6s, đã chọn) đúng **1.2 giây** (< 2.0s).

**Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?**
**Không**. Điểm bất định cao chỉ chứng minh mô hình đang *bối rối*, chứ không đảm bảo việc train trên ảnh đó sẽ nâng cao chất lượng mô hình. Nếu độ bất định xuất phát từ nhiễu thị giác không thể học được (mờ nhòe do tốc độ cao, chói lóa đèn xe đối diện, phản xạ mặt đường), việc ép mô hình học các mẫu này thậm chí làm sai lệch trọng số và giảm độ khái quát trên tập test.

---

## 4. Các vòng học chủ động (active learning)

Bảng so sánh tổng hợp từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 346 | 0.463 | -0.308 | 1.000 | 0.102 | 0.185 | 0.000 | 0.095 | 0.317 |

### Mức độ can thiệp và sửa đổi pre-label (theo `outputs/round1_diff.md`)
Trên 12 ảnh của vòng 1:
- Mô hình AI khởi đầu đề xuất ban đầu: **169 box**.
- Sau khi rà soát và gán nhãn thủ công trên CVAT, tổng số box thực tế là: **346 box**.
- Thống kê chi tiết thao tác:
  + **Giữ nguyên (`accepted`)**: 159 box (tỷ lệ chấp nhận đạt 94% số box do AI đề xuất).
  + **Chỉnh sửa biên (`edited`)**: 2 box (điều chỉnh viền box ôm sát thân xe thay vì ôm cả vệt đèn rọi mặt đường).
  + **Xóa bỏ (`deleted` - False Positive của AI)**: 8 box (xóa các box AI nhận nhầm ánh đèn phản chiếu trên mặt đường hoặc biển báo thành xe).
  + **Thêm mới (`added` - False Negative của AI)**: **185 box** (AI bỏ sót một lượng xe cực lớn ở các vùng tối và xe ở làn xa).

### Biến động số đo sau khi fine-tune Vòng 1
- **AP50**: Giảm từ **0.771** xuống **0.463** (giảm 0.308 so với khởi đầu lạnh).
- **Precision**: Tăng từ **0.925** lên mức tuyệt đối **1.000** (không có bất kỳ False Positive nào trên toàn bộ 20 ảnh test).
- **Recall**: Giảm mạnh từ **0.489** xuống **0.102** (chỉ bắt được 41/403 box tham chiếu, bỏ sót 362 xe).
- **Biến động theo kích thước xe**:
  + Xe nhỏ (`R small`): Rơi từ **0.182** về **0.000**.
  + Xe trung bình (`R medium`): Giảm từ **0.547** xuống **0.095**.
  + Xe lớn (`R large`): Giảm từ **0.561** xuống **0.317**.

**Nguyên nhân kỹ thuật**: Mô hình chỉ được fine-tune trên vỏn vẹn **12 ảnh** với 50 epochs mà không đóng băng backbone (backbone freezing). Do tập dữ liệu huấn luyện quá nhỏ so với 118k ảnh COCO ban đầu, mô hình bị rơi vào trạng thái **overfitting** nghiêm trọng và trở nên cực kỳ "thận trọng" (over-conservative). Nó chỉ dám dự đoán khi độ tự tin gần như tuyệt đối (dẫn đến Precision đạt 1.000), nhưng lại bỏ qua toàn bộ các xe nhỏ và xe trung bình ở làn xa (khiến Recall tụt dốc).

### Đối chiếu bằng chứng thực tế
- Dựa vào `outputs/compare_round1.jpg`, các xe lớn ở làn gần đi chậm tiếp tục được phát hiện rất sắc nét và chuẩn xác (không bị dính vệt đèn chiếu đường như ở vòng 0). Tuy nhiên, các vệt xe nhỏ ở làn đối diện hoàn toàn biến mất khỏi dự đoán của mô hình vòng 1.
- **Phân biệt 3 tầng dữ liệu**:
  1. *Quan sát độc lập (`BLIND_SCAN.md`)*: Trên `frame_0099.jpg`, mắt thường đếm được 26 xe và nhận diện rõ hai vị trí nguy cơ: xe tối màu sát dải phân cách bên phải và xe bị che khuất gần vùng đèn pha.
  2. *Lỗi pre-label AI đã sửa (`REVIEW_LOG.csv`)*: AI chỉ đề xuất 13 box, bỏ sót tới 12 xe tối và xe bị che. Người gán nhãn đã bổ sung 12 box thiếu và nắn chỉnh box ôm sát thân xe theo đúng quy tắc.
  3. *Kết quả mô hình sau train*: Mô hình sau train đã loại bỏ hoàn toàn các box ảo trên mặt đường, song do thiếu lượng mẫu đa dạng của xe nhỏ, nó chưa khôi phục được khả năng bắt các xe ở cự ly xa.
- **Tình huống khó theo guideline**: Trường hợp hai xe chạy song song sát nhau bị ánh đèn pha lóa hòa vào nhau. Theo guideline, người gán nhãn kiên quyết tách thành hai box độc lập dựa vào ranh giới nóc xe thay vì gộp chung, giúp dữ liệu train đạt tính nhất quán cao nhất.

---

## 5. Kết luận và giới hạn

### Đánh giá kết quả và quyết định dừng/tiếp tục
Kết quả vòng 1 cho thấy một sự đánh đổi điển hình: mô hình đạt độ chính xác hoàn hảo (Precision = 1.0) nhưng độ bao phủ (Recall) và AP50 suy giảm mạnh do hiện tượng co cụm dự đoán trên tập train siêu nhỏ (12 ảnh). 

Quyết định cho vòng tiếp theo là **CẦN TIẾP TỤC VÒNG 2**, nhưng phải điều chỉnh chiến lược huấn luyện (giảm learning rate, đóng băng một phần backbone hoặc bổ sung thêm kỹ thuật data augmentation) nhằm khôi phục Recall cho nhóm xe nhỏ và trung bình.

### Đề xuất 2 ca cho vòng tiếp theo (từ `outputs/selection_round2.csv`)
Dựa trên kết quả chọn mẫu vòng 2:
1. **`frame_0070.jpg`** (t = 28.0s): Nằm ở khoảng trống thời gian đầu video, chứa nhiều xe chạy ở làn xa có kích thước nhỏ và vừa. Việc đưa frame này vào sẽ trực tiếp bổ sung mẫu huấn luyện cho nhóm `small` và `medium` đang có Recall = 0.
2. **`frame_0276.jpg`** (t = 110.4s): Bối cảnh xe tải lớn che khuất một phần các xe con phía sau. Giúp mô hình học lại đặc trưng xe bị che khuất (occluded) ở cự ly trung bình.
- *Cân nhắc chi phí & rủi ro*: Cả hai frame đều có số lượng box vừa phải (khoảng 25–35 box), cách xa các frame đã train ở vòng 1 trên 15 giây, triệt tiêu hoàn toàn nguy cơ trùng lặp cảnh (*near-duplicate*).

### Các giới hạn thực nghiệm
1. **Kích thước tập test nhỏ (20 ảnh)**: Dung lượng 20 ảnh khiến các chỉ số rất nhạy cảm với biến động ngẫu nhiên; chỉ cần bỏ sót thêm vài xe ở 1–2 frame là AP50 sụt giảm đáng kể.
2. **Quy tắc bỏ qua xe quá nhỏ (< 16 px)**: Giúp giảm bớt sự mơ hồ ở đường chân trời, nhưng đồng thời tạo ra một "ranh giới cứng" giữa các xe 15 px và 17 px.
3. **Nhãn tham chiếu do mô hình AI tạo ra**: Đây là giới hạn lớn nhất. Nhãn tham chiếu chưa được con người thẩm định 100%, do đó AP50 chỉ đo **mức độ trùng khớp với mô hình sinh nhãn tham chiếu** chứ không phải chân lý tuyệt đối của bài toán thực tế.

### Quy trình kiểm tra nếu AP50 giảm
Nếu AP50 tiếp tục giảm ở vòng sau, các bước kiểm định bắt buộc trước khi train thêm bao gồm:
1. Kiểm tra ma trận nhầm lẫn và phân bố lỗi (phân tích xem AP50 giảm do False Positives tăng hay do False Negatives tăng).
2. Kiểm tra chất lượng nhãn xuất ra từ công cụ gán nhãn: đối chiếu file `.txt` xem có box nào bị lệch tọa độ ($cx, cy, w, h$) hoặc sai class id ngoài `0` hay không.
3. Điều chỉnh siêu tham số fine-tune: áp dụng đóng băng các tầng đầu của backbone (`freeze = 10`), hạ thấp learning rate (`lr0 = 0.001`) để mô hình giữ lại các đặc trưng tổng quát đã học từ COCO thay vì bị overfit vào 12 ảnh mới.
