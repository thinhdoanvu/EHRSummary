import os
import shutil
import pandas as pd

# 1. Định nghĩa các đường dẫn
SOURCE_DIR = r'C:\Users\VU\Documents\NLP\Demo2\datasets\individual_posts'
EMPTY_DIR = r'C:\Users\VU\Documents\NLP\Demo2\datasets\empty'

# Tạo thư mục đích nếu chưa tồn tại
if not os.path.exists(EMPTY_DIR):
    os.makedirs(EMPTY_DIR)

# 2. Quét qua các file trong thư mục nguồn
files_processed = 0
files_moved = 0

print("Đang kiểm tra các tệp tin...")

for filename in os.listdir(SOURCE_DIR):
    if filename.endswith('.csv'):
        file_path = os.path.join(SOURCE_DIR, filename)

        try:
            # Đọc file csv
            df = pd.read_csv(file_path)

            # Kiểm tra nếu cột 'post_content' không tồn tại hoặc dữ liệu rỗng
            # Chúng ta dùng .astype(str) để xử lý các giá trị NaN và dùng .strip() để loại bỏ khoảng trắng
            if 'post_content' not in df.columns:
                is_empty = True
            else:
                # Kiểm tra nếu tất cả hàng trong cột post_content đều rỗng
                # (Thông thường mỗi file của bạn chỉ có 1 bài viết, nên dùng .iloc[0])
                content = str(df['post_content'].iloc[0]) if not df['post_content'].empty else ""
                is_empty = content.strip() == "" or content.lower() == "nan"

            if is_empty:
                dest_path = os.path.join(EMPTY_DIR, filename)
                shutil.move(file_path, dest_path)
                files_moved += 1

            files_processed += 1

        except Exception as e:
            print(f"Lỗi khi xử lý file {filename}: {e}")

# 3. Tổng kết
print("---")
print(f"Hoàn thành!")
print(f"Tổng số file đã kiểm tra: {files_processed}")
print(f"Tổng số file đã di chuyển: {files_moved}")