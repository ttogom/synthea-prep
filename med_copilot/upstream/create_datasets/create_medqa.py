import json
jsonl_path='train.jsonl'
numbers=200
json_path=f'medqa_{numbers}.json'
data=[]
with open(jsonl_path,'r',encoding='utf-8',errors='ignore') as f:
    for i,line in enumerate(f):
        if i>=numbers:
            break
        data.append(json.loads(line))

with open(json_path,'w',encoding='utf-8',errors='ignore') as f:
    json.dump(data,f,ensure_ascii=False,indent=4)

print("successfully stored all the info into a new json file")
