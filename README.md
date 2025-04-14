## Cài đặt
- Clone source code và các thư viện cần thiết:  
```bash
git clone https://github.com/vietdai-bk/simulation_control_vehicle
pip install -r requirements.txt
```
## Chạy mô phỏng
- Mở trình mô phỏng: [VIA Simulation](https://via-sim.makerviet.org)  
- Chạy file ```drive.py``` để gửi tín hiệu điều khiển qua websockets.
## Lưu ý
- Chạy ```drive.py``` trước khi start xe trong VIA.  
- Trong qua trình chạy không nên click vào màn hình để tránh tín hiệu gửi đến xe bị gián đoạn làm xe chạy lệch.
- Thay đổi các tính toán góc lái và tốc độ, xử lí nhận diện lane trong code ```lane_line_detection.py```
