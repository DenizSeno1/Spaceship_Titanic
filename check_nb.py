import json

path = 'c:/Users/deniz/python_projects/Spaceship_Titanic/spaceship_titanic.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:80].replace('\n', ' ')
    print(f"{i}: [{c['cell_type']}] {src}")
