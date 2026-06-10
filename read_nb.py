import json

path = 'c:/Users/deniz/python_projects/Spaceship_Titanic/spaceship_titanic.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Toplam hücre sayısı: {len(nb['cells'])}\n")
print("="*80)
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    print(f"\n--- Hücre {i} [{c['cell_type']}] ---")
    print(src[:300])
    print("...")
