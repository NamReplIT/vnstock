from datetime import datetime,date
import json

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):  # Check if the object is a date
            return obj.isoformat()   # Use the same ISO format for dates
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)
class JSONHelper:
    @staticmethod
    def read_json(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Invalid JSON in file: {file_path}")
            return None

    @staticmethod
    def write_json(file_path,data):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, cls=DateTimeEncoder, ensure_ascii=False, indent=4)
            print(f"Data written to {file_path}")
            return True
        except Exception as e:
            print(f"Error writing to file: {e}")
            return False
        
    @staticmethod
    def append_json(file_path,new_data):
        try:
            # Read the existing data
            existing_data = JSONHelper.read_json(file_path)
            if existing_data is None:
                existing_data = []

            # Check if the existing data is a list
            if not isinstance(existing_data, list):
                print("Error: JSON data is not a list.")
                return False

            # Append new data
            existing_data.append(new_data)

            # Write the updated data back to the file
            return JSONHelper.write_json(existing_data, file_path)
        except Exception as e:
            print(f"Error appending to file: {e}")
            return False
