import pandas as pd
import json
from datetime import datetime
import itertools

class RawDataProcessor:
    def __init__(self, file_path):
        """
        Khởi tạo với đường dẫn file CSV
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Đọc file CSV bài đăng"""
        try:
            # Đọc file với encoding utf-8 để tránh lỗi ký tự đặc biệt
            self.df = pd.read_csv(self.file_path)
            return True
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            return False

    def _format_date(self, date_str):
        """Chuẩn hóa ngày tháng sang YYYY-MM-DD"""
        try:
            # Xử lý trường hợp có chữ T (ISO format) hoặc khoảng trắng
            clean_date = str(date_str).split('T')[0].split(' ')[0]
            return clean_date
        except:
            return "NA"

    def process_all_posts(self):
        """
        Duyệt qua từng dòng và đóng gói thành cấu trúc Block
        Return: Dictionary với key là Post ID
        """
        if self.df is None:
            if not self.load_data():
                return {}

        all_blocks = {}

        for _, row in self.df.iterrows():
            post_id = str(row.get('post_id', ''))

            # Cấu trúc lưu trữ dạng Block cho từng Post
            post_entry = {
                "IDENTIFIER_BLOCK": {
                    "GROUP_START": "IDENTIFIER_BLOCK",
                    "Post ID": post_id,
                    "Source": "Reddit",
                    "Date": self._format_date(row.get('time')),
                    "num_comments": row.get('num_comments', 0),
                    "GROUP_END": "IDENTIFIER_BLOCK"
                },
                "CONTENT_BLOCK": {
                    "GROUP_START": "CONTENT_BLOCK",
                    "post_title": str(row.get('post_title', '')),
                    "post_content": str(row.get('post_content', '')),
                    # Kết hợp để Council dễ đọc một lần
                    "full_text": f"{row.get('post_title', '')}. {row.get('post_content', '')}",
                    "GROUP_END": "CONTENT_BLOCK"
                }
            }

            all_blocks[post_id] = post_entry

        return all_blocks

# --- VÍ DỤ CÁCH SỬ DỤNG ---
if __name__ == "__main__":
    file_path = r"C:\Users\VU\Documents\NLP\Demo2\datasets\posts.csv"
    processor = RawDataProcessor(file_path)

    # Thực hiện bóc tách toàn bộ 6000 bài
    data_dict = processor.process_all_posts()

    # In thông báo tổng quát
    print(f"--- Đã bóc tách thành công: {len(data_dict)} bài viết ---")
    print("--- Trích xuất 10 bài đầu tiên để kiểm tra Blocks ---\n")

    # Lấy 10 phần tử đầu tiên từ Dictionary
    first_10_items = dict(itertools.islice(data_dict.items(), 10))

    for post_id, blocks in first_10_items.items():
        print(f"=== [CHECKING BLOCKS FOR ID: {post_id}] ===")

        # Kiểm tra IDENTIFIER_BLOCK
        ident = blocks['IDENTIFIER_BLOCK']
        print(f"[{ident['GROUP_START']}] -> Date: {ident['Date']}, Comments: {ident['num_comments']}")

        # Kiểm tra CONTENT_BLOCK (Chỉ in 100 ký tự đầu của full_text để dễ nhìn)
        content = blocks['CONTENT_BLOCK']
        summary_text = content['full_text'][:100] + "..." if len(content['full_text']) > 100 else content['full_text']
        print(f"[{content['GROUP_START']}] -> Text: {summary_text}")

        print(f"=== [END {post_id}] ===\n")

# # Sau này hàm tổng hợp của bạn sẽ gọi như thế này:
# # for pid, blocks in data_dict.items():
# #     raw_text = blocks['CONTENT_BLOCK']['full_text']
# #     # Gửi raw_text này cho Council biểu quyết...