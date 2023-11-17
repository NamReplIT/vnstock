from functools import cmp_to_key
class ListHelper:
    @staticmethod
    def map(lst, func):
        return [func(x) for x in lst]

    @staticmethod
    def filter(lst, func):
        return [x for x in lst if func(x)]

    @staticmethod
    def reduce(lst, func, initial):
        result = initial
        for x in lst:
            result = func(result, x)
        return result

    @staticmethod
    def find(lst, func):
        for x in lst:
            if func(x):
                return x
        return None

    @staticmethod
    def concat(*lists):
        return [item for sublist in lists for item in sublist]

    @staticmethod
    def slice(lst, start, end=None):
        return lst[start:end]

    @staticmethod
    def splice(lst, start, delete_count=None, *items):
        if delete_count is None:
            delete_count = len(lst) - start
        removed_items = lst[start:start+delete_count]
        lst[start:start+delete_count] = items
        return removed_items

    @staticmethod
    def index_of(lst, item):
        try:
            return lst.index(item)
        except ValueError:
            return -1

    @staticmethod
    def last_index_of(lst, item):
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] == item:
                return i
        return -1

    @staticmethod
    def reverse(lst):
        return lst[::-1]

    @staticmethod
    def includes(lst, item):
        return item in lst

    @staticmethod
    def find_index(lst, func):
        for index, value in enumerate(lst):
            if func(value):
                return index
        return -1

    @staticmethod
    def join(lst, separator=''):
        return separator.join(map(str, lst))

    @staticmethod
    def fill(lst, value, start=0, end=None):
        if end is None:
            end = len(lst)
        for i in range(start, end):
            if i < len(lst):
                lst[i] = value
        return lst
    
    @staticmethod
    def sort(lst, compare_func=None):
        if compare_func:
            key_func = cmp_to_key(compare_func)
            return sorted(lst, key=key_func)
        else:
            return sorted(lst)

# # Example usage
# numbers = [1, 2, 3, 4, 5]

# # Map
# squared_numbers = ListHelper.map(numbers, lambda x: x**2)
# print("Squared Numbers:", squared_numbers)

# # Filter
# even_numbers = ListHelper.filter(numbers, lambda x: x % 2 == 0)
# print("Even Numbers:", even_numbers)

# # Reduce
# sum_of_numbers = ListHelper.reduce(numbers, lambda a, b: a + b, 0)
# print("Sum of Numbers:", sum_of_numbers)

# # Find
# first_even_number = ListHelper.find(numbers, lambda x: x % 2 == 0)
# print("First Even Number:", first_even_number)

# # Concat
# list1 = [1, 2, 3]
# list2 = [4, 5, 6]
# concatenated_list = ListHelper.concat(list1, list2)
# print("Concatenated List:", concatenated_list)


# # Example usage
# numbers = [1, 2, 3, 4, 5]

# # Slice
# sliced_numbers = ListHelper.slice(numbers, 1, 4)
# print("Sliced Numbers:", sliced_numbers)

# # Splice
# spliced_numbers = ListHelper.splice(numbers, 1, 2, 8, 9)
# print("After Splice:", numbers)
# print("Spliced Numbers:", spliced_numbers)

# # Index Of
# index = ListHelper.index_of(numbers, 3)
# print("Index of 3:", index)

# # Last Index Of
# last_index = ListHelper.last_index_of(numbers, 9)
# print("Last Index of 9:", last_index)

# # Reverse
# reversed_numbers = ListHelper.reverse(numbers)
# print("Reversed Numbers:", reversed_numbers)


# # Example usage
# numbers = [1, 2, 3, 4, 5]

# # Includes
# print("Includes 3:", ListHelper.includes(numbers, 3))
# print("Includes 7:", ListHelper.includes(numbers, 7))

# # Find Index
# index = ListHelper.find_index(numbers, lambda x: x > 3)
# print("Index of first number greater than 3:", index)

# # Join
# joined_string = ListHelper.join(numbers, '-')
# print("Joined String:", joined_string)

# # Fill
# filled_list = ListHelper.fill(numbers, 0, 2, 4)
# print("Filled List:", filled_list)

# # Example usage
# numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# # Sort using a compare function (ascending)
# sorted_numbers = ListHelper.sort(numbers, compare_func=lambda a, b: a - b)
# print("Sorted Numbers (Ascending):", sorted_numbers)

# # Sort using a compare function (descending)
# sorted_numbers_desc = ListHelper.sort(numbers, compare_func=lambda a, b: b - a)
# print("Sorted Numbers (Descending):", sorted_numbers_desc)