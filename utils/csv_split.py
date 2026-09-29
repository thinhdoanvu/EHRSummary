import pandas as pd
import os


class CSVSplitter:
    def __init__(self, input_file, output_dir):
        """
        input_file: Đường dẫn file csv tổng (6000 dòng)
        output_dir: Thư mục sẽ chứa các file con post_id.csv
        """
        self.input_file = input_file
        self.output_dir = output_dir

        # Tạo thư mục đầu ra nếu chưa tồn tại
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def split_by_post_id(self):
        """Thực hiện tách file"""
        print(f"--- Đang đọc file tổng: {self.input_file} ---")

        # Đọc toàn bộ file CSV
        try:
            df = pd.read_csv(self.input_file)
        except Exception as e:
            print(f"❌ Lỗi khi đọc file: {e}")
            return

        total_rows = len(df)
        print(f"✅ Đã tải {total_rows} dòng. Bắt đầu tách...")

        for index, row in df.iterrows():
            post_id = str(row['post_id'])

            # Tạo DataFrame mới từ 1 dòng hiện tại
            single_row_df = pd.DataFrame([row])

            # Đường dẫn file: output_dir/post_id.csv
            file_name = f"{post_id}.csv"
            output_path = os.path.join(self.output_dir, file_name)

            # Lưu ra file CSV (không lưu index)
            single_row_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            if (index + 1) % 100 == 0:
                print(f"🚀 Đã xử lý {index + 1}/{total_rows} file...")

        print(f"\n✨ HOÀN THÀNH: Đã tạo {total_rows} file tại {self.output_dir}")


# --- THỰC THI ---
if __name__ == "__main__":
    INPUT_CSV = r"C:\Users\VU\Documents\NLP\Demo2\datasets\posts.csv"
    OUTPUT_FOLDER = r"C:\Users\VU\Documents\NLP\Demo2\datasets\individual_posts"

    splitter = CSVSplitter(INPUT_CSV, OUTPUT_FOLDER)
    splitter.split_by_post_id()