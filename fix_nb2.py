import json

path = 'c:/Users/deniz/python_projects/Spaceship_Titanic/spaceship_titanic.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# BÖLÜM 6 kodu: X ve modelleri tanımla + eğit
b6_code = [
    "from sklearn.model_selection import cross_val_score, StratifiedKFold\n",
    "from sklearn.ensemble import StackingClassifier\n",
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
    "# X ve y tanımla\n",
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
    "    results[name] = sc.mean()\n",
    "    print(f'✅ {name}: %{sc.mean()*100:.2f} (±{sc.std()*100:.2f}) | {time.time()-t:.1f}s')\n",
    "\n",
    "print('\\n🏗️ Stacking Ensemble kuruluyor...')\n",
    "stack = StackingClassifier(\n",
    "    estimators=[('cat', cat_model), ('lgb', lgb_model), ('xgb', xgb_model)],\n",
    "    final_estimator=LogisticRegression(), cv=5, n_jobs=-1\n",
    ")\n",
    "t = time.time()\n",
    "sc = cross_val_score(stack, X, y, cv=cv, scoring='accuracy', n_jobs=-1)\n",
    "print(f'✅ Stacking: %{sc.mean()*100:.2f} (±{sc.std()*100:.2f}) | {time.time()-t:.1f}s')\n",
    "\n",
    "print('\\n📦 Şampiyon model tüm veriyle eğitilip submission dosyası üretiliyor...')\n",
    "stack.fit(X, y)\n",
    "preds = stack.predict(df_test).astype(bool)\n",
    "test_raw = pd.read_csv('test.csv')\n",
    "sub = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Transported': preds})\n",
    "sub.to_csv('stacking_submission.csv', index=False)\n",
    "print(\"🎉 'stacking_submission.csv' başarıyla oluşturuldu!\")\n"
]

# BÖLÜM 7 kodu: Grandmaster özellikleri
b7_code = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "print('🚀 Grandmaster Özellikleri Üretiliyor...')\n",
    "\n",
    "# X ve y tanımla (Bölüm 6'da zaten tanımlıydı ama sıfırdan da çalışsın diye burada da var)\n",
    "if 'X' not in dir():\n",
    "    X = df_train.drop(columns=['Transported'], errors='ignore').copy()\n",
    "y = df_train['Transported'].copy()\n",
    "\n",
    "# Orijinal verileri arka planda okuyoruz\n",
    "raw_train = pd.read_csv('train.csv')\n",
    "raw_test = pd.read_csv('test.csv')\n",
    "raw_train['Group'] = raw_train['PassengerId'].apply(lambda x: x.split('_')[0])\n",
    "raw_train['Surname'] = raw_train['Name'].fillna('Unknown Unknown').str.split(' ').str[-1]\n",
    "raw_test['Group'] = raw_test['PassengerId'].apply(lambda x: x.split('_')[0])\n",
    "raw_test['Surname'] = raw_test['Name'].fillna('Unknown Unknown').str.split(' ').str[-1]\n",
    "\n",
    "# ---- 1. AİLE KURTULMA ORANI ----\n",
    "family_stats = raw_train.groupby('Surname')['Transported'].agg(['sum','count']).reset_index()\n",
    "family_stats.columns = ['Surname','Family_Transported_Sum','Family_Count']\n",
    "\n",
    "tr_fam = pd.merge(raw_train[['PassengerId','Surname','Transported']], family_stats, on='Surname', how='left')\n",
    "tr_fam['Family_Transported_Sum'] = tr_fam['Family_Transported_Sum'] - tr_fam['Transported']\n",
    "tr_fam['Family_Count'] = tr_fam['Family_Count'] - 1\n",
    "tr_fam['FamilySurvivalRate'] = tr_fam['Family_Transported_Sum'] / tr_fam['Family_Count']\n",
    "\n",
    "te_fam = pd.merge(raw_test[['PassengerId','Surname']], family_stats, on='Surname', how='left')\n",
    "te_fam['FamilySurvivalRate'] = te_fam['Family_Transported_Sum'] / te_fam['Family_Count']\n",
    "\n",
    "X['FamilySurvivalRate'] = tr_fam['FamilySurvivalRate'].fillna(-1).values\n",
    "df_test['FamilySurvivalRate'] = te_fam['FamilySurvivalRate'].fillna(-1).values\n",
    "\n",
    "# ---- 2. GRUP KURTULMA ORANI ----\n",
    "group_stats = raw_train.groupby('Group')['Transported'].agg(['sum','count']).reset_index()\n",
    "group_stats.columns = ['Group','Group_Transported_Sum','Group_Count']\n",
    "\n",
    "tr_grp = pd.merge(raw_train[['PassengerId','Group','Transported']], group_stats, on='Group', how='left')\n",
    "tr_grp['Group_Transported_Sum'] = tr_grp['Group_Transported_Sum'] - tr_grp['Transported']\n",
    "tr_grp['Group_Count'] = tr_grp['Group_Count'] - 1\n",
    "tr_grp['GroupSurvivalRate'] = tr_grp['Group_Transported_Sum'] / tr_grp['Group_Count']\n",
    "\n",
    "te_grp = pd.merge(raw_test[['PassengerId','Group']], group_stats, on='Group', how='left')\n",
    "te_grp['GroupSurvivalRate'] = te_grp['Group_Transported_Sum'] / te_grp['Group_Count']\n",
    "\n",
    "X['GroupSurvivalRate'] = tr_grp['GroupSurvivalRate'].fillna(-1).values\n",
    "df_test['GroupSurvivalRate'] = te_grp['GroupSurvivalRate'].fillna(-1).values\n",
    "\n",
    "# ---- 3. GEMİ BÖLGE HARİTALAMASI ----\n",
    "if 'CabinNum' in X.columns:\n",
    "    X['CabinRegion'] = pd.qcut(X['CabinNum'], q=4, labels=[1,2,3,4]).astype(int)\n",
    "    df_test['CabinRegion'] = pd.qcut(df_test['CabinNum'], q=4, labels=[1,2,3,4]).astype(int)\n",
    "    X.drop('CabinNum', axis=1, inplace=True)\n",
    "    df_test.drop('CabinNum', axis=1, inplace=True)\n",
    "\n",
    "print(f'✅ Grandmaster Özellikleri Tamam! X sütun sayısı: {X.shape[1]}')\n",
    "print(f'   Yeni sütunlar: FamilySurvivalRate, GroupSurvivalRate, CabinRegion')\n",
    "print('\\n📋 X sütunları:')\n",
    "print(list(X.columns))\n"
]

# Bölüm 6 ve 7'yi bul ve kaynak kodunu güncelle
b6_found = False
b7_found = False

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if cell['cell_type'] == 'code' and 'from sklearn.model_selection import cross_val_score' in src and 'BÖLÜM 6' not in src:
        # Bu Bölüm 6'nın kod hücresi (index 48)
        nb['cells'][i]['source'] = b6_code
        b6_found = True
        print(f"✅ Bölüm 6 kodu güncellendi (hücre {i})")
    if cell['cell_type'] == 'code' and 'FamilySurvivalRate' in src:
        # Bu Bölüm 7'nin kod hücresi (index 50)
        nb['cells'][i]['source'] = b7_code
        b7_found = True
        print(f"✅ Bölüm 7 kodu güncellendi (hücre {i})")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nBölüm 6 bulundu mu: {b6_found}")
print(f"Bölüm 7 bulundu mu: {b7_found}")
print("Notebook kaydedildi ✅")
