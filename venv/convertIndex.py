import json
json_path = './mostRepeated.json'

def convertIndex(item):
    # Read the JSON data from the file
    with open(json_path, 'r') as file:
        json_dataset = json.load(file)

    # item = {
    #     "num_01": "01",
    #     "num_02": "13",
    #     "num_03": "16",
    #     "num_04": "18",
    #     "num_05": "23",
    #     "num_06": "25",
    # }

    results = []
    for key in item:
        try:
            index = json_dataset[key].index(item[key])
        except ValueError:
            index = 'N/A'
        results.append(f"{key.split('_')[1]}-{index}")

    return '::'.join(results)

def main():
    y='01-0::02-8::03-4::04-12::05-20::06-18'
    next = '01-6::02-14::03-10::04-0::05-9::06-12'

    #next = '01-0::02-8::03-4::04-12::05-20::06-18'
    #y= '01-1::02-0::03-24::04-25::05-22::06-8'
    y = convertIndex( {
        "num_01": "02",
        "num_02": "07",
        "num_03": "09",
        "num_04": "13",
        "num_05": "22",
        "num_06": "38",
    })
    print(y)
# Entry point of the script
if __name__ == "__main__":
    main()