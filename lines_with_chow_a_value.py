import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sqlalchemy import create_engine
from scipy.stats import f 


# Подключение к PostgreSQL
engine = create_engine("postgresql://gis:123456@10.0.62.59:55432/gis")

# Загрузка данных
query = 'SELECT year, a FROM "earthquakes_gr_low"."values_results_rfm"'
df = pd.read_sql(query, engine)

df = df.sort_values('year').reset_index(drop=True)

print("Первые строки:")
print(df.head())
print(f"\nДиапазон лет: {df.year.min()} — {df.year.max()}")
print(f"Количество наблюдений: {len(df)}")

# pooled модель (без разрыва)
Xp = sm.add_constant(df[['year']])
pooled_model = sm.OLS(df['a'], Xp).fit()
RSS_pooled = np.sum(pooled_model.resid**2)

# Перебор годов разрыва
results = []

break_years = range(1970, 2016)

for break_year in break_years:

    df['E'] = np.where(df['year'] <= break_year, 1, 0)
    df['t'] = df['year']
    df['t_E'] = df['t'] * df['E']

    X = sm.add_constant(df[['t', 'E', 't_E']])
    y = df['a']

    model = sm.OLS(y, X).fit()

    #величина скачка
    jump = model.params['E'] + model.params['t_E'] * break_year

    # критерий Чоу
    RSS_full = np.sum(model.resid**2)

    k = 2
    n = len(df)

    F_stat = ((RSS_pooled - RSS_full) / k) / (RSS_full / (n - 2*k))
    p_value = 1 - f.cdf(F_stat, k, n - 2*k)

    # сохраняем результаты
    results.append({
        'break_year': break_year,
        'rsquared': model.rsquared,
        'jump': jump,
        'F_stat': F_stat,
        'p_value': p_value
    })

# Таблица результатов
results_df = pd.DataFrame(results)

# Оптимальный год по Чоу
best_row = results_df.loc[results_df['F_stat'].idxmax()]

best_year = int(best_row['break_year'])
best_F = best_row['F_stat']
best_p = best_row['p_value']
best_jump = best_row['jump']

print("\n Оптимальный год разрыва (тест Чоу):")
print("Год:", best_year)
print("F:", round(best_F, 3))
print("p-value:", round(best_p, 6))
print("Скачок:", round(best_jump, 4))


# Итоговая модель
df['E'] = np.where(df['year'] <= best_year, 1, 0)
df['t'] = df['year']
df['t_E'] = df['t'] * df['E']

X = sm.add_constant(df[['t', 'E', 't_E']])
model = sm.OLS(df['a'], X).fit()

print("\nИтоговая регрессия")
print(model.summary())

# Итоговые уравнения
const = model.params['const']
beta = model.params['t']
gamma = model.params['E']
delta = model.params['t_E']

a_before = const + gamma
b_before = beta + delta

a_after = const
b_after = beta

print("\n Итоговые уравнения:")

print(f"\nДо {best_year}:")
print(f"a(t) = {a_before:.6f} + {b_before:.6f} * t")

print(f"\nПосле {best_year}:")
print(f"a(t) = {a_after:.6f} + {b_after:.6f} * t")

print("\nLaTeX запись:")
print(f"До {best_year}: $a(t) = {a_before:.4f} + {b_before:.6f}t$")
print(f"После {best_year}: $a(t) = {a_after:.4f} + {b_after:.6f}t$")

# Сохранение данных для LaTeX
def save_coords(filename, x, y):
    with open(filename, "w") as f:
        for xi, yi in zip(x, y):
            f.write(f"({xi:.3f},{yi:.6f})\n")

save_coords("F_plot.txt",
            results_df.break_year,
            results_df.F_stat)

save_coords("jump_plot.txt",
            results_df.break_year,
            results_df.jump)

save_coords("raw_points.txt",
            df.year,
            df.a)

print("\n Данные графиков сохранены для LaTeX!")

# Графики
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.plot(results_df.break_year, results_df.F_stat, 'o-')
ax1.axvline(best_year, linestyle='--', label=f'Лучший T={best_year}')
ax1.set_title("F-статистика теста Чоу")
ax1.set_xlabel("Год разрыва")
ax1.set_ylabel("F")
ax1.legend()
ax1.grid()

ax2.plot(results_df.break_year, results_df.jump, 'o-')
ax2.axhline(0, color='black')
ax2.set_title("Величина скачка")
ax2.set_xlabel("Год разрыва")
ax2.set_ylabel("Jump")
ax2.grid()

plt.tight_layout()
plt.savefig('chow_a_value_min_mc.png', dpi=300, bbox_inches='tight')
print("График сохранен как 'chow_a_value_min_mc.png'")
plt.show()


# Итоговая модель для лучшего года
print(f"\nДетальная модель для T = {best_year}")

df['E'] = np.where(df['year'] <= best_year, 1, 0)
df['t'] = df['year']
df['t_E'] = df['t'] * df['E']

X = sm.add_constant(df[['t', 'E', 't_E']])
model = sm.OLS(df['a'], X).fit()

print(model.summary())

# Визуализация
plt.figure(figsize=(12, 7))

plt.plot(df.year, df.a,'o-', label="Данные")

const = model.params['const']
beta = model.params['t']
gamma = model.params['E']
delta = model.params['t_E']

years_before = np.linspace(df.year.min(), best_year, 100)
years_after = np.linspace(best_year, df.year.max(), 100)

pred_before = (const + gamma) + (beta + delta)*years_before
pred_after = const + beta*years_after

plt.plot(years_before, pred_before, linewidth=3, label="До разрыва")
plt.plot(years_after, pred_after, linewidth=3, label="После разрыва")

plt.axvline(best_year, linestyle='--', label="Разрыв")

plt.title(f"Оптимальный разрыв: {best_year}")
plt.xlabel("Год")
plt.ylabel("a-value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f'chow_best_{best_year}_a_value_min_mc.png', dpi=300, bbox_inches='tight')
print(f"График регрессии сохранен как 'chow_best_{best_year}_a_value_min_mc.png'")
plt.show()
