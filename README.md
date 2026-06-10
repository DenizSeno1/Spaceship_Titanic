# Spaceship Titanic Kaggle Competition

![Kaggle Score](https://img.shields.io/badge/Kaggle_Score-0.80500-blue)
![Top Percentage](https://img.shields.io/badge/Top-~39%25-brightgreen)

Bu proje, Kaggle'daki **Spaceship Titanic** yarışması için hazırlanmış bir Makine Öğrenmesi çözümüdür. Amaç, yolcuların çarpışma sırasında "başka bir boyuta ışınlanıp ışınlanmadığını" (Transported) tahmin etmektir.

## 🚀 Proje Özeti
- **Kaggle Skoru (Public LB):** `0.80500`
- **Algoritmalar:** CatBoost, LightGBM, XGBoost
- **Ensemble Yöntemi:** Stacking Classifier (Final Estimator: Logistic Regression)

## 🧠 Kullanılan Teknikler
1. **Keşifsel Veri Analizi (EDA):** Veri dağılımlarının incelenmesi, eksik verilerin tespiti.
2. **Özellik Mühendisliği (Feature Engineering):**
   - Yolcu ID'lerinden Grup ve Aile büyüklüklerinin çıkarılması.
   - Kabin numaralarından Deck (Güverte) ve Side (Taraf) bilgilerinin ayrıştırılması.
   - Harcama kalemlerinin toplanarak `TotalSpent` (Toplam Harcama) özelliğinin oluşturulması.
3. **Veri Ön İşleme:**
   - Eksik değerlerin (Missing Values) mantıklı stratejilerle (median, mod vb.) doldurulması.
   - Kategorik verilerin One-Hot Encoding ile sayısal hale getirilmesi.
4. **Modelleme & Stacking:**
   - Birbirinden farklı mimarilere sahip 3 güçlü Gradient Boosting modeli eğitildi.
   - Modellerin tahminleri bir `LogisticRegression` modeli ile harmanlanarak (Stacking) maksimum performansa ulaşıldı.

## 📂 Dosya Yapısı
- `spaceship_titanic.ipynb`: Tüm EDA, Özellik Mühendisliği ve modelleme kodlarını içeren ana Jupyter Notebook dosyası.
- `.gitignore`: Büyük veri setlerini (CSV dosyaları) repodan uzak tutmak için.

## 🔧 Nasıl Çalıştırılır?
1. Repoyu klonlayın.
2. Kaggle'dan `train.csv` ve `test.csv` dosyalarını indirip projenin ana dizinine koyun.
3. `spaceship_titanic.ipynb` dosyasını baştan sona çalıştırın.

---
*Bu proje Makine Öğrenmesi gelişim yolculuğumun bir parçasıdır.*
