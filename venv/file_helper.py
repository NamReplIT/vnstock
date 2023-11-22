import os
class FileHelper:
    @staticmethod
    def read_file(file_path):
        """Đọc nội dung của file từ đường dẫn được cung cấp."""
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            return "File không tồn tại."

    @staticmethod
    def write_file(file_path, content):
        """Viết nội dung vào file tại đường dẫn được cung cấp."""
        with open(file_path, 'w') as file:
            file.write(content)

    @staticmethod
    def file_exists(file_path,create_dir=False):
        """Kiểm tra xem file có tồn tại tại đường dẫn được cung cấp hay không.
        Nếu không, tạo thư mục chứa file đó."""
        if os.path.exists(file_path):
            return True
        else:
            if create_dir:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            return False
    
    @staticmethod
    def read_dir(directory_path):
        """Đọc và trả về danh sách tên file trong thư mục được cung cấp."""
        try:
            file_list = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
            return file_list
        except FileNotFoundError:
            return "Thư mục không tồn tại."
        except Exception as e:
            return f"Lỗi khi đọc thư mục: {e}"

# # Ví dụ sử dụng
# file_path = 'example.txt'

# # Kiểm tra file tồn tại
# print("File tồn tại:", FileHelper.file_exists(file_path))

# # Ghi nội dung vào file
# FileHelper.write_file(file_path, "Xin chào!")

# # Đọc nội dung từ file
# print("Nội dung file:", FileHelper.read_file(file_path))
