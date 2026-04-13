# ==========================================
# Скрипт 04: Random Forest для ShannonIndex с графиком важности
# ==========================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# ==========================================
# ШАГ 0 -----> Проверка файлов
# ==========================================
if not os.path.exists("Results"):
    os.makedirs("Results")

for f in ["Results/model_table_data.csv", "Results/matrix_samples_phylum.csv"]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} не найден. Сначала запустите скрипты 01 и 03.")

# ==========================================
# ШАГ 1 -----> Чтение таблицы модели
# ==========================================
final_df = pd.read_csv("Results/model_table_data.csv", encoding="utf-8-sig")

# ==========================================
# ШАГ 2 -----> Чтение матрицы филумов
# ==========================================
matrix = pd.read_csv("Results/matrix_samples_phylum.csv", index_col=0)

# ==========================================
# ШАГ 3 -----> Multi-hot кодирование категориальных признаков
# ==========================================
def multi_hot(series):
    series = series.fillna("")
    split = series.apply(lambda x: [i.strip() for i in str(x).split(",")])
    unique_vals = list(set([item for sublist in split for item in sublist if item]))
    oh_df = pd.DataFrame(0, index=series.index, columns=unique_vals)
    for idx, cats in split.items():
        for cat in cats:
            if cat:
                oh_df.loc[idx, cat] = 1
    return oh_df

final_df_indexed = final_df.drop_duplicates(subset="Sample").set_index("Sample")

soiltype_ohe = multi_hot(final_df_indexed["SoilType"])
subtype_ohe = multi_hot(final_df_indexed["SubType"])
gran_df = multi_hot(final_df_indexed["Granularity"])
subregion_ohe = multi_hot(final_df_indexed["SubRegion"])
parentmat_ohe = multi_hot(final_df_indexed["Parent_Mat"])

if "type_area" in final_df_indexed.columns:
    typearea_ohe = multi_hot(final_df_indexed["type_area"])
else:
    typearea_ohe = pd.DataFrame(index=final_df_indexed.index)

# ==========================================
# ШАГ 3.1 -----> Сохранение multi-hot таблиц
# ==========================================

soiltype_ohe.to_csv("Results/ohe_soiltype.csv", encoding="utf-8-sig")
subtype_ohe.to_csv("Results/ohe_subtype.csv", encoding="utf-8-sig")
gran_df.to_csv("Results/ohe_granularity.csv", encoding="utf-8-sig")
subregion_ohe.to_csv("Results/ohe_subregion.csv", encoding="utf-8-sig")
parentmat_ohe.to_csv("Results/ohe_parentmat.csv", encoding="utf-8-sig")

if not typearea_ohe.empty:
    typearea_ohe.to_csv("Results/ohe_typearea.csv", encoding="utf-8-sig")

print("\n✅ Multi-hot таблицы сохранены в папке Results/")
print("\n=== Размеры multi-hot таблиц ===")
print("SoilType:", soiltype_ohe.shape)
print("SubType:", subtype_ohe.shape)
print("Granularity:", gran_df.shape)
print("SubRegion:", subregion_ohe.shape)
print("Parent_Mat:", parentmat_ohe.shape)

if not typearea_ohe.empty:
    print("type_area:", typearea_ohe.shape)
# ==========================================
# ШАГ 4 -----> Объединяем признаки
# ==========================================
common_samples = matrix.index.intersection(final_df_indexed.index)

features = pd.concat([
    matrix.loc[common_samples],
    soiltype_ohe.loc[common_samples],
    subtype_ohe.loc[common_samples],
    gran_df.loc[common_samples],
    subregion_ohe.loc[common_samples],
    parentmat_ohe.loc[common_samples]#,
    #typearea_ohe.loc[common_samples]
], axis=1)

y = final_df_indexed.loc[common_samples, "ShannonIndex"]

print("Размер X:", features.shape)
print("Размер y:", y.shape)

# ==========================================
# ШАГ 5 -----> Random Forest с кросс-валидацией
# ==========================================
model = RandomForestRegressor(n_estimators=500, random_state=42)
r2_scores = cross_val_score(model, features, y, cv=5, scoring="r2")
print("R2 по кросс-валидации:", r2_scores)
print("Средний R2:", r2_scores.mean())

# ==========================================
# ШАГ 6 -----> Обучаем модель на всех данных
# ==========================================
model.fit(features, y)

# ==========================================
# ШАГ 7 -----> Важность признаков
# ==========================================
importance = pd.DataFrame({
    "Feature": features.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nТоп признаков:\n", importance.head(20))

# ==========================================
# ШАГ 7.1 -----> Сохранение важности признаков
# ==========================================
importance.to_csv("Results/feature_importance_full.csv", index=False, encoding="utf-8-sig")
importance.to_excel("Results/feature_importance_full.xlsx", index=False)

print("\n✅ Таблица важности признаков сохранена:")
print(" - Results/feature_importance_full.csv")
print(" - Results/feature_importance_full.xlsx")

# ==========================================
# ШАГ 7.2 -----> Разделение признаков
# ==========================================
# Микроорганизмы (филумы)
microbes_importance = importance[importance["Feature"].isin(matrix.columns)]

# Почвенные признаки
soil_importance = importance[~importance["Feature"].isin(matrix.columns)]

# Сохраняем отдельно
microbes_importance.to_csv("Results/importance_microbes.csv", index=False, encoding="utf-8-sig")
soil_importance.to_csv("Results/importance_soil.csv", index=False, encoding="utf-8-sig")

print("\n✅ Отдельные таблицы сохранены:")
print(" - importance_microbes.csv")
print(" - importance_soil.csv")

print("\n✅ Random Forest модель обучена и сохранена!")

# ==========================================
# ШАГ 8 -----> Визуализация топ-20 признаков
# ==========================================
top_features = importance.head(20)
plt.figure(figsize=(10,6))
plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color="skyblue")
plt.xlabel("Importance")
plt.title("Top 20 признаков Random Forest")
plt.tight_layout()
plt.savefig("Results/feature_importance_top20.png", dpi=300)
plt.show()
print("\n✅ График топ-20 признаков сохранен в Results/feature_importance_top20.png")
