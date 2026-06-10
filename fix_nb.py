import json

path = 'c:/Users/deniz/python_projects/Spaceship_Titanic/spaceship_titanic.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The cells we want to insert.
# 1. BÖLÜM 6 Markdown
b6_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# BÖLÜM 6: MODELLEME VE ENSEMBLE (V1)\n",
        "Modellerimizi kuruyor ve Stacking Ensemble ile %80.5'lik skorumuzu üretiyoruz."
    ]
}

# 2. BÖLÜM 6 Code
b6_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from sklearn.model_selection import cross_val_score, StratifiedKFold\n",
        "from catboost import CatBoostClassifier\n",
        "from lightgbm import LGBMClassifier\n",
        "from xgboost import XGBClassifier\n",
        "from sklearn.ensemble import StackingClassifier\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "import pandas as pd\n",
        "import time\n",
        "\n",
        "X = df_train.drop(['Transported'], axis=1)\n",
        "y = df_train['Transported']\n",
        "test_ids = pd.read_csv('test.csv')['PassengerId']\n",
        "\n",
        "cat_model = CatBoostClassifier(verbose=0, random_state=42)\n",
        "lgb_model = LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)\n",
        "xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')\n",
        "\n",
        "models = [('CatBoost', cat_model), ('LightGBM', lgb_model), ('XGBoost', xgb_model)]\n",
        "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "\n",
        "for name, model in models:\n",
        "    start_time = time.time()\n",
        "    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)\n",
        "    print(f\"✅ {name}: %{scores.mean()*100:.2f} (±{scores.std()*100:.2f}) | {time.time()-start_time:.1f}s\")\n",
        "\n",
        "estimators = [('cat', cat_model), ('lgb', lgb_model), ('xgb', xgb_model)]\n",
        "stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5, n_jobs=-1)\n",
        "\n",
        "start_time = time.time()\n",
        "stack_scores = cross_val_score(stack_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)\n",
        "print(f\"\\n✅ Stacking: %{stack_scores.mean()*100:.2f} (±{stack_scores.std()*100:.2f}) | {time.time()-start_time:.1f}s\")\n",
        "\n",
        "stack_model.fit(X, y)\n",
        "voting_preds = stack_model.predict(df_test)\n",
        "voting_preds = voting_preds.astype(bool)\n",
        "\n",
        "stacking_submission = pd.DataFrame({'PassengerId': test_ids, 'Transported': voting_preds})\n",
        "stacking_submission.to_csv('stacking_submission.csv', index=False)\n",
        "print(\"🎉 'stacking_submission.csv' başarıyla oluşturuldu! Kaggle'a yüklemeye hazır.\")\n"
    ]
}

# 3. BÖLÜM 7 Code (Grandmaster Features)
b7_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "print(\"🚀 Grandmaster Özellikleri Üretiliyor...\")\n",
        "\n",
        "raw_train = pd.read_csv('train.csv')\n",
        "raw_test = pd.read_csv('test.csv')\n",
        "\n",
        "raw_train['Group'] = raw_train['PassengerId'].apply(lambda x: x.split('_')[0])\n",
        "raw_train['Surname'] = raw_train['Name'].str.split(' ').str[-1]\n",
        "raw_test['Group'] = raw_test['PassengerId'].apply(lambda x: x.split('_')[0])\n",
        "raw_test['Surname'] = raw_test['Name'].str.split(' ').str[-1]\n",
        "\n",
        "family_stats = raw_train.groupby('Surname')['Transported'].agg(['sum', 'count']).reset_index()\n",
        "family_stats.columns = ['Surname', 'Family_Transported_Sum', 'Family_Count']\n",
        "\n",
        "train_family = pd.merge(raw_train[['PassengerId', 'Surname', 'Transported']], family_stats, on='Surname', how='left')\n",
        "train_family['Family_Transported_Sum'] = train_family['Family_Transported_Sum'] - train_family['Transported']\n",
        "train_family['Family_Count'] = train_family['Family_Count'] - 1\n",
        "train_family['FamilySurvivalRate'] = train_family['Family_Transported_Sum'] / train_family['Family_Count']\n",
        "\n",
        "test_family = pd.merge(raw_test[['PassengerId', 'Surname']], family_stats, on='Surname', how='left')\n",
        "test_family['FamilySurvivalRate'] = test_family['Family_Transported_Sum'] / test_family['Family_Count']\n",
        "\n",
        "group_stats = raw_train.groupby('Group')['Transported'].agg(['sum', 'count']).reset_index()\n",
        "group_stats.columns = ['Group', 'Group_Transported_Sum', 'Group_Count']\n",
        "\n",
        "train_group = pd.merge(raw_train[['PassengerId', 'Group', 'Transported']], group_stats, on='Group', how='left')\n",
        "train_group['Group_Transported_Sum'] = train_group['Group_Transported_Sum'] - train_group['Transported']\n",
        "train_group['Group_Count'] = train_group['Group_Count'] - 1\n",
        "train_group['GroupSurvivalRate'] = train_group['Group_Transported_Sum'] / train_group['Group_Count']\n",
        "\n",
        "test_group = pd.merge(raw_test[['PassengerId', 'Group']], group_stats, on='Group', how='left')\n",
        "test_group['GroupSurvivalRate'] = test_group['Group_Transported_Sum'] / test_group['Group_Count']\n",
        "\n",
        "X['FamilySurvivalRate'] = train_family['FamilySurvivalRate'].fillna(-1).values\n",
        "df_test['FamilySurvivalRate'] = test_family['FamilySurvivalRate'].fillna(-1).values\n",
        "X['GroupSurvivalRate'] = train_group['GroupSurvivalRate'].fillna(-1).values\n",
        "df_test['GroupSurvivalRate'] = test_group['GroupSurvivalRate'].fillna(-1).values\n",
        "\n",
        "X['CabinRegion'] = pd.qcut(X['CabinNum'], q=4, labels=[1, 2, 3, 4]).astype(int)\n",
        "df_test['CabinRegion'] = pd.qcut(df_test['CabinNum'], q=4, labels=[1, 2, 3, 4]).astype(int)\n",
        "\n",
        "if 'CabinNum' in X.columns:\n",
        "    X.drop('CabinNum', axis=1, inplace=True)\n",
        "if 'CabinNum' in df_test.columns:\n",
        "    df_test.drop('CabinNum', axis=1, inplace=True)\n",
        "\n",
        "print(f\"✅ Sızıntısız Soyağacı ve Fiziksel Haritalama Tamam! X sütun sayısı: {X.shape[1]}\")\n"
    ]
}

# Find the index of BÖLÜM 7 markdown
b7_idx = -1
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown' and 'BÖLÜM 7' in ''.join(c['source']):
        b7_idx = i
        break

if b7_idx != -1:
    # Insert BÖLÜM 6 stuff right before BÖLÜM 7 markdown
    nb['cells'].insert(b7_idx, b6_md)
    nb['cells'].insert(b7_idx + 1, b6_code)
    # Now BÖLÜM 7 markdown is at b7_idx + 2
    # Insert BÖLÜM 7 code right after it
    nb['cells'].insert(b7_idx + 3, b7_code)

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Kayıp hücreler geri yüklendi.")
