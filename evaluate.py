import json

if __name__ == '__main__':
    correct_num = 0
    total_num = 0
    with open(file='./generated_predictions.txt', mode='r', encoding='utf-8') as fp:
        for line in fp.readlines():
            total_num += 1
            line = line.strip()
            json_obj = json.loads(line)
            if json_obj['labels'] == json_obj['predict']:
                correct_num += 1
    print(f"The accuracy is: {(100*correct_num/total_num):0.2f}%, [{correct_num}/{total_num}]")