import json
json_path = './mostRepeated.json'

def convertIndex(items):
    # Read the JSON data from the file
    with open(json_path, 'r') as file:
        json_dataset = json.load(file)

    # items = [{
    #     "num_01": "01",
    #     "num_02": "13",
    #     "num_03": "16",
    #     "num_04": "18",
    #     "num_05": "23",
    #     "num_06": "25",
    # }]

    list_result = []

    for item in items:

        results = []
        for key in item:
            try:
                index = json_dataset[key].index(item[key])
            except ValueError:
                index = 'N/A'
            results.append(f"{key.split('_')[1]}-{index}")

        list_result.append('::'.join(results))

    return list_result