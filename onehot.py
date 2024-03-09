import json
import pandas as pd
from sklearn.ensemble import IsolationForest

# JSON dosyasını oku
with open('data.json') as f:
    data = f.readlines()

# JSON verisini DataFrame'e dönüştür
df = pd.DataFrame([json.loads(line) for line in data])

# One Hot Encoding için kategorik sütunları seç
categorical_cols = ["SourceName", "Channel"]
X_categorical = df[categorical_cols]

# Sayısal sütunları seç
numerical_cols = ["Level", "EventID"]
X_numerical = df[numerical_cols]

# Anormallik tespiti için tüm sütunları birleştir
X = pd.concat([X_categorical, X_numerical], axis=1)

# One Hot Encoding uygula
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Anormallik tespiti modelini oluştur
clf = IsolationForest(contamination=0.1)

# Modeli eğit
clf.fit(X_encoded)

# Tahminleri yap
predictions = clf.predict(X_encoded)

# Anormallik olarak işaretlenmiş örnekleri filtrele
anomalies = df[predictions == -1]

print("Anormallikler:")
print(anomalies)

# X_numerical'ı CSV dosyası olarak dışa aktar
X_numerical.to_csv('X_numerical.csv', index=False)