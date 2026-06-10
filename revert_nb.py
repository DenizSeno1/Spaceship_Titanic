import json

path = 'c:/Users/deniz/python_projects/Spaceship_Titanic/spaceship_titanic.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Mevcut hücre sayısı: {len(nb['cells'])}")

# Orijinal BÖLÜM 6 kodu (ilk stacking_submission'ı üreten)
original_b6_code = [
    "from sklearn.model_selection import cross_val_score, StratifiedKFold\n",
    "from sklearn.ensemble import StackingClassifier, VotingClassifier\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from catboost import CatBoostClassifier\n",
    "from lightgbm import LGBMClassifier\n",
    "from xgboost import XGBClassifier\n",
    "import time\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "X = df_train.drop(columns=['Transported'], errors='ignore').copy()\n",
    "y = df_train['Transported'].copy()\n",
    "\n",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "cat_model = CatBoostClassifier(verbose=0, random_state=42)\n",
    "lgb_model = LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)\n",
    "xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')\n",
    "\n",
    "print('\\n🚀 Yeni özelliklerle modeller tekrar eğitiliyor...')\n",
    "results = {}\n",
    "for name, model in [('CatBoost', cat_model), ('LightGBM', lgb_model), ('XGBoost', xgb_model)]:\n",
    "    t = time.time()\n",
    "    sc = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)\n",
    "    results[name] = sc\n",
    "    print(f'✅ {name}: %{sc.mean()*100:.2f} (±{sc.std()*100:.2f}) | {time.time()-t:.1f}s')\n",
    "\n",
    "print('\\n🏗️ Stacking Ensemble kuruluyor...')\n",
    "print('(Modellerin tahminlerini üzerine bir Logistic Regression oturtuyor)')\n",
    "stack_model = StackingClassifier(\n",
    "    estimators=[('cat', cat_model), ('lgb', lgb_model), ('xgb', xgb_model)],\n",
    "    final_estimator=LogisticRegression(), cv=5, n_jobs=-1\n",
    ")\n",
    "t = time.time()\n",
    "sc = cross_val_score(stack_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)\n",
    "print(f'\\n✅ Stacking: %{sc.mean()*100:.2f} (±{sc.std()*100:.2f}) | {time.time()-t:.1f}s')\n",
    "\n",
    "import pandas as pd\n",
    "print('\\n' + '='*55)\n",
    "print('     🏁 YENİ ÖZELLİKLER + STACKING FİNAL TABLOSU')\n",
    "print('='*55)\n",
    "all_results = {**{k: v.mean() for k,v in results.items()}, 'Stacking (Cat+LGB+XGB)': sc.mean()}\n",
    "df_results = pd.DataFrame([(k, round(v*100,2)) for k,v in sorted(all_results.items(), key=lambda x: -x[1])], columns=['Model', 'Doğruluk (%)'])\n",
    "print(df_results.to_string(index=False))\n",
    "\n",
    "print('\\n📦 Şampiyon model (Stacking) tüm veriyle eğitilip submission dosyası üretiliyor...')\n",
    "stack_model.fit(X, y)\n",
    "preds = stack_model.predict(df_test[X.columns]).astype(bool)\n",
    "test_raw = pd.read_csv('test.csv')\n",
    "sub = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Transported': preds})\n",
    "sub.to_csv('stacking_submission.csv', index=False)\n",
    "print(\"🎉 'stacking_submission.csv' başarıyla oluşturuldu! Kaggle'a yüklemeye hazır.\")\n"
]

# BÖLÜM 6 ve 7 hücrelerini bul ve güncelle
cells_to_remove = []
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if cell['cell_type'] == 'code' and 'FamilySurvivalRate' in src:
        cells_to_remove.append(i)
    if cell['cell_type'] == 'markdown' and 'BÖLÜM 7' in src:
        cells_to_remove.append(i)
    if cell['cell_type'] == 'code' and 'from sklearn.model_selection import cross_val_score' in src and 'best_cat' not in src and 'FamilySurvivalRate' not in src:
        nb['cells'][i]['source'] = original_b6_code
        print(f"✅ Bölüm 6 kodu orijinaline döndürüldü (hücre {i})")

# Silinecek hücreleri büyükten küçüğe sırala ve sil
for i in sorted(set(cells_to_remove), reverse=True):
    print(f"🗑️  Hücre {i} silindi: {nb['cells'][i]['cell_type']} - {''.join(nb['cells'][i]['source'])[:40]}")
    nb['cells'].pop(i)

print(f"\nYeni hücre sayısı: {len(nb['cells'])}")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook orijinal haline döndürüldü!")
