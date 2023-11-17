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
    def file_exists(file_path):
        """Kiểm tra xem file có tồn tại tại đường dẫn được cung cấp hay không."""
        try:
            with open(file_path, 'r'):
                return True
        except FileNotFoundError:
            return False

# # Ví dụ sử dụng
# file_path = 'example.txt'

# # Kiểm tra file tồn tại
# print("File tồn tại:", FileHelper.file_exists(file_path))

# # Ghi nội dung vào file
# FileHelper.write_file(file_path, "Xin chào!")

# # Đọc nội dung từ file
# print("Nội dung file:", FileHelper.read_file(file_path))
