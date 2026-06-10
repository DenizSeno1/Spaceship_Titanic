import json

path = 'c:/Users/deniz/python_projects/Spaceship_Titanic/spaceship_titanic.ipynb'

print("Notebook okunuyor...")
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

modifications = {
    0: "# BÖLÜM 1: GİRİŞ VE VERİ YÜKLEME\n## 1.1 Kütüphaneler",
    2: "## 1.2 Veri Seti Yükleme ve İnceleme",
    9: "# BÖLÜM 2: ÖZELLİK MÜHENDİSLİĞİ (V1)\n## 2.1 Temel Özellik Çıkarımı",
    13: "## 2.2 Eksik Veri Analizi",
    17: "# BÖLÜM 3: EKSİK VERİ DOLDURMA (IMPUTATION)\n## 3.1 Temel Veri Doldurma",
    21: "## 3.2 VIP ve CryoSleep Kuralları",
    26: "## 3.3 Harcama Eksiklikleri",
    30: "## 3.4 Aile ve Grup İçi Doldurma",
    32: "## 3.5 Aile İçi Harcama Dağılımı",
    34: "## 3.6 Genel Temizlik",
    36: "## 3.7 Kabin (Sosyokültürel) Doldurma",
    38: "# BÖLÜM 4: VERİ HAZIRLIĞI\n## 4.1 Model Öncesi Sütun Düşürme",
    41: "## 4.2 Sayısallaştırma (Encoding)",
    43: "# BÖLÜM 5: KEŞİFSEL VERİ ANALİZİ (EDA)\n*(Not: Modelleme, Optimizasyon ve Stacking hücreleri bu analizden sonra yer almaktadır.)*"
}

for idx, new_source in modifications.items():
    if idx < len(nb['cells']) and nb['cells'][idx]['cell_type'] == 'markdown':
        nb['cells'][idx]['source'] = [new_source]

new_markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "<br>\n",
        "# ==================================================\n",
        "# BÖLÜM 7: ŞAMPİYONLAR LİGİ (0.83+ HEDEFİ)\n",
        "# ==================================================\n",
        "\n",
        "Bu bölümde Kaggle Grandmaster taktiklerini kullanarak skorumuzu maksimize edeceğiz:\n",
        "- Gemi İçi Fiziksel Haritalama (Cabin Binning)\n",
        "- Aile Kurtulma Oranları (Family Survival Rate)\n",
        "- Target Encoding\n",
        "- Optuna ile Kapsamlı Hiperparametre Araması\n"
    ]
}

nb['cells'].append(new_markdown_cell)

print("Notebook kaydediliyor...")
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Düzenleme Başarılı!")
