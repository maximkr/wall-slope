# filename: wall_analysis_correct_axes.py
# execution: true

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import math

# Параметры
MAYAK_THICKNESS = 0.008  # толщина маяка в метрах (6 мм)
MAX_VERTICAL_ANGLE = 2.0  # максимальный наклон по вертикали в градусах
MAX_NEGATIVE_DEPTH = 0.015  # максимальное заглубление в метрах (15 мм)
ALLOWED_NEGATIVE_POINTS_RATIO = 0  # доля точек с отрицательным расстоянием (50%)

# Читаем данные из CSV
df = pd.read_csv('wall_points.csv')
points = df[['x', 'y', 'z']].values  # x - ширина, y - глубина, z - высота
x = points[:, 0]  # ширина
y = points[:, 1]  # глубина
z = points[:, 2]  # высота

def calculate_distances(params):
    """Рассчитывает расстояния от точек до плоскости маяков"""
    a, b, c = params
    # y = ax + bz + c - уравнение плоскости маяков
    return a * x + b * z + c - y

def objective_function(params):
    """Функция для минимизации с улучшенными штрафами"""
    a, b, c = params
    distances = calculate_distances(params)

    # Базовая сумма положительных расстояний
    positive_distances = np.maximum(0, distances)
    base_cost = np.sum(positive_distances)

    # Штраф за наклон по вертикали более 3 градусов
    vertical_angle = math.degrees(math.atan(abs(b)))
    angle_penalty = 100000.0 * max(0, vertical_angle - MAX_VERTICAL_ANGLE)**2

    # Проверка количества точек с заглублением
    negative_points = np.sum(distances < 0)
    max_negative_points = int(len(distances) * ALLOWED_NEGATIVE_POINTS_RATIO)
    negative_count_penalty = 100000.0 * max(0, negative_points - max_negative_points)

    # Штраф за заглубление более 5 мм
    too_deep_penalty = 100000.0 * np.sum(np.maximum(0, -(distances + MAX_NEGATIVE_DEPTH)))

    # Штраф за толщину меньше маяка для положительных точек
    positive_points = distances > 0
    thickness_penalty = 100000.0 * np.sum(
        np.maximum(0, MAYAK_THICKNESS - distances) * positive_points
    )

    return (base_cost + angle_penalty + 
            negative_count_penalty + too_deep_penalty + thickness_penalty)

# Начальное приближение для вертикальной стены
initial_guess = [0.0, 0.0, MAYAK_THICKNESS]  # предполагаем параллельную стену

# Оптимизация
result = minimize(objective_function, initial_guess, method='Nelder-Mead')
a, b, c = result.x
distances = calculate_distances([a, b, c])

# Расчет статистики
wall_width = max(x) - min(x)
wall_height = max(z) - min(z)
wall_area = wall_width * wall_height
angle_x = math.degrees(math.atan(abs(a)))
angle_z = math.degrees(math.atan(abs(b)))

# Создаем словарь с результатами
stats = {
    "Параметры стены": {
        "Ширина стены": f"{wall_width:.2f} м",
        "Высота стены": f"{wall_height:.2f} м",
        "Площадь стены": f"{wall_area:.2f} м²"
    },
    "Параметры маяков": {
        "Толщина маяка": f"{MAYAK_THICKNESS*1000:.1f} мм",
        "Уравнение плоскости": f"y = {a:.4f}x + {b:.4f}z + {c:.4f}",
        "Наклон по ширине (ось X)": f"{angle_x:.2f}°",
        "Наклон по высоте (ось Z)": f"{angle_z:.2f}°"
    },
    "Параметры штукатурки": {
        "Минимальная толщина": f"{np.min(distances)*1000:.1f} мм",
        "Средняя толщина": f"{np.mean(distances)*1000:.1f} мм",
        "Максимальная толщина": f"{np.max(distances)*1000:.1f} мм",
        "Общий объем штукатурки": f"{np.sum(distances):.3f} м³",
    }
}

# Вывод статистики
print("\nСТАТИСТИКА АНАЛИЗА СТЕНЫ:")
print("="*50)
for category, params in stats.items():
    print(f"\n{category}:")
    print("-"*50)
    for param, value in params.items():
        print(f"{param:<{max(len(p) for p in params.keys())+2}}: {value}")


# Добавляем вывод данных по точкам
print("\nТОЛЩИНА ШТУКАТУРКИ ПО ТОЧКАМ:")
print("="*50)
print(f"{'Номер точки':>11} {'X (м)':>10} {'Y (м)':>10} {'Z (м)':>10} {'Толщина (мм)':>13}")
print("-"*54)
for i in range(len(x)):
    print(f"{df['point_number'].iloc[i]:11d} {x[i]:10.3f} {y[i]:10.3f} {z[i]:10.3f} {distances[i]*1000:13.1f}")
