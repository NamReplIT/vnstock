class ObjectHelper:
    @staticmethod
    def keys(obj):
        return list(obj.keys())

    @staticmethod
    def values(obj):
        return list(obj.values())

    @staticmethod
    def entries(obj):
        return list(obj.items())

    @staticmethod
    def assign(target, *args):
        for obj in args:
            target.update(obj)
        return target
    
    @staticmethod
    def has_key(obj, key):
        return key in obj

# # Example usage
# my_dict = {'a': 1, 'b': 2}

# # Get keys
# print("Keys:", ObjectHelper.keys(my_dict))

# # Get values
# print("Values:", ObjectHelper.values(my_dict))

# # Get entries
# print("Entries:", ObjectHelper.entries(my_dict))

# # Assign
# new_dict = {'c': 3}
# ObjectHelper.assign(my_dict, new_dict)
# print("After Assign:", my_dict)
