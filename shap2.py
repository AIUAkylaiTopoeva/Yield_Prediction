import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import shap
import matplotlib.pyplot as plt
import os

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================
df = pd.read_csv('data/kg_standards.csv')  # Ваши данные

# ==================== 2. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ ====================
def generate_synthetic_data(df, n_samples_per_record=400):
    """
    Генерация синтетических данных на основе экспертных стандартов
    """
    synthetic_data = []
    
    for _, row in df.iterrows():
        for _ in range(n_samples_per_record):
            # 1. Параметры почвы с шумом
            N = max(0, row['Optimal_N'] + np.random.uniform(-30, 30))
            P = max(0, row['Optimal_P'] + np.random.uniform(-20, 20))
            K = max(0, row['Optimal_K'] + np.random.uniform(-50, 50))
            
            # 2. pH с шумом
            pH_noise = np.random.uniform(-0.7, 0.7)
            pH = max(5.0, min(8.0, row['Optimal_pH'] + pH_noise))
            
            # 3. Осадки
            rainfall = np.random.normal(450, 150)
            rainfall = max(100, min(1000, rainfall))
            
            # 4. Расчет урожайности с штрафами
            base_yield = row['Expected_Yield_ha']
            yield_factor = 1.0
            
            # Штраф за дефицит N (менее 80% от оптимального)
            if N < 0.8 * row['Optimal_N']:
                penalty = 0.3 * (1 - N/row['Optimal_N'])
                yield_factor *= (1 - penalty)
            
            # Критический дефицит P (менее 70% от оптимального)
            if P < 0.7 * row['Optimal_P']:
                penalty = 0.4 * (1 - P/row['Optimal_P'])
                yield_factor *= (1 - penalty)
            elif P < row['Optimal_P']:
                penalty = 0.2 * (1 - P/row['Optimal_P'])
                yield_factor *= (1 - penalty)
            
            # Дефицит K (менее 80% от оптимального)
            if K < 0.8 * row['Optimal_K']:
                penalty = 0.15 * (1 - K/row['Optimal_K'])
                yield_factor *= (1 - penalty)
            
            # Штраф за pH (отклонение > 0.5)
            pH_deviation = abs(pH - row['Optimal_pH'])
            if pH_deviation > 0.5:
                penalty = 0.15 * (pH_deviation - 0.5)
                yield_factor *= (1 - penalty)
            
            # Штраф за осадки (вне диапазона 300-700 мм)
            if rainfall < 300 or rainfall > 700:
                deviation = min(abs(rainfall-300), abs(rainfall-700)) / 100
                penalty = 0.15 * min(deviation, 1.0)
                yield_factor *= (1 - penalty)
            
            # Нижняя граница урожайности (30% от базовой)
            final_yield = base_yield * max(0.3, yield_factor)
            
            synthetic_data.append({
                'Region': row['Region'],
                'Crop': row['Crop'],
                'N': N,
                'P': P,
                'K': K,
                'pH': pH,
                'Rainfall': rainfall,
                'Yield': final_yield
            })
    
    return pd.DataFrame(synthetic_data)

# Генерируем 5000 образцов
print("Генерация синтетических данных...")
synthetic_df = generate_synthetic_data(df, n_samples_per_record=400)

print(f"Создано {len(synthetic_df)} синтетических образцов")
print(synthetic_df.head())

# ==================== 3. ПОДГОТОВКА ДАННЫХ ====================
# One-hot encoding для категориальных признаков
encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_features = encoder.fit_transform(synthetic_df[['Region', 'Crop']])
categorical_df = pd.DataFrame(
    categorical_features,
    columns=encoder.get_feature_names_out(['Region', 'Crop'])
)

# Объединяем все признаки
X = pd.concat([
    synthetic_df[['N', 'P', 'K', 'pH', 'Rainfall']].reset_index(drop=True),
    categorical_df.reset_index(drop=True)
], axis=1)

y = synthetic_df['Yield'].values

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nРазмеры данных:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# ==================== 4. ОБУЧЕНИЕ МОДЕЛИ ====================
print("\nОбучение модели Random Forest...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Оценка модели
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nРезультаты модели:")
print(f"MAE: {mae:.3f} т/га")
print(f"R²: {r2:.3f}")

# ==================== 5. СОЗДАНИЕ ГРАФИКОВ SHAP ====================
print("\nСоздание SHAP графиков...")
os.makedirs('figures', exist_ok=True)

# Инициализация SHAP
explainer = shap.TreeExplainer(model)

# Вычисляем SHAP values для тестовых данных
# Берем подмножество для скорости (первые 100 образцов)
shap_values = explainer.shap_values(X_test.iloc[:100])

# 1. Глобальная важность признаков (Bar Plot)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test.iloc[:100], plot_type="bar", show=False)
plt.title("Global Feature Importance (SHAP values)", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('figures/shap_global.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: figures/shap_global.png")

# 2. Summary plot (показывает распределение влияния)
plt.figure(figsize=(14, 8))
shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
plt.title("SHAP Summary Plot", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('figures/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Сохранен: figures/shap_summary.png")

# 3. Локальное объяснение для примера с яблоками в Чуйской области
# Найдем пример с яблоками в тестовых данных
apple_mask = X_test['Crop_Apple'] == 1 if 'Crop_Apple' in X_test.columns else False

if apple_mask.any():
    apple_idx = X_test[apple_mask].index[0]
    
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[apple_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[apple_idx],
        feature_names=X_test.columns
    ), max_display=10)
    plt.title(f"Local SHAP Explanation (Apple in Chuy Oblast)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('figures/shap_local.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Сохранен: figures/shap_local.png")
else:
    # Если нет яблок, берем первый пример
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns
    ), max_display=10)
    plt.title(f"Local SHAP Explanation (Sample #{0})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('figures/shap_local.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Сохранен: figures/shap_local.png (первый пример)")

# ==================== 6. СОХРАНЕНИЕ МОДЕЛИ И ДАННЫХ ====================
import joblib

# Сохраняем модель
joblib.dump(model, 'agroexpert_model.pkl')
print("✓ Сохранена модель: agroexpert_model.pkl")

# Сохраняем кодировщик
joblib.dump(encoder, 'encoder.pkl')
print("✓ Сохранен кодировщик: encoder.pkl")

# Сохраняем синтетические данные
synthetic_df.to_csv('synthetic_dataset.csv', index=False)
print("✓ Сохранены синтетические данные: synthetic_dataset.csv")

# ==================== 7. ИНФОРМАЦИЯ О ПРИЗНАКАХ ====================
print("\n" + "="*60)
print("СПИСОК ПРИЗНАКОВ В МОДЕЛИ:")
print("="*60)
for i, col in enumerate(X.columns):
    print(f"{i+1:2d}. {col}")

print("\n" + "="*60)
print("ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
print("="*60)
# Важность признаков из Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

print("\n" + "="*60)
print("ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКУ 'figures/'")
print("="*60)