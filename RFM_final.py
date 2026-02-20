from collections import defaultdict
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import Integer, Numeric
import statsmodels.api as sm
from statsmodels.robust.norms import TukeyBiweight
import matplotlib.pyplot as plt
import os

def bin_magnitudes(magnitudes, bin_size=0.1, m_min=None, m_max=None):
    """
    Группирует магнитуды в бины и вычисляет некумулятивное N(M) и кумулятивное O(M).

    Parameters:
    - magnitudes: list or np.array of magnitudes
    - bin_size: размер бина (default 0.1)
    - m_min, m_max: диапазон (если None, берутся из данных)

    Returns:
    - bins: centers of bins (M)
    - N: non-cumulative counts
    - O: cumulative counts (O(M) = sum N for M' >= M)
    """
    if m_min is None:
        m_min = np.min(magnitudes)
    if m_max is None:
        m_max = np.max(magnitudes)

    # Создаём бины
    bins = np.arange(m_min, m_max + bin_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Центры бинов

    # Подсчёт N(M) в каждом бине
    N = np.zeros(len(bin_centers))
    for mag in magnitudes:
        # Находим индекс бина
        idx = np.searchsorted(bins, mag, side='right') - 1
        if 0 <= idx < len(N):
            N[idx] += 1

    # Кумулятивное O(M)
    O = np.cumsum(N[::-1])[::-1]  # Кумулятивная сумма от конца

    # Удаляем бины с O=0 в конце
    valid_idx = O > 0
    bin_centers = bin_centers[valid_idx]
    N = N[valid_idx]
    O = O[valid_idx]

    return bin_centers, N, O


def estimate_b_a_rfm(M, O, max_iter=15, c=6.946):  
    """
    Robust fit на log O(M) = a - b M (cumulative).
    """
    log_O = np.log10(O + 1e-10)
    n = len(M)

    X = np.column_stack([np.ones(n), M])

    # Начальный fit (polyfit на log_O)
    slope, intercept = np.polyfit(M, log_O, 1)
    b_init = -slope
    a_init = intercept

    for _ in range(max_iter):
        log_O_pred = a_init - b_init * M
        residuals = log_O - log_O_pred

        XtX_inv = np.linalg.inv(X.T @ X)
        hat_matrix = X @ XtX_inv @ X.T
        leverages = np.diag(hat_matrix)

        adjusted_residuals = residuals / np.sqrt(np.maximum(1 - leverages, 1e-10))

        # MAD на adjusted residuals (центрируем на median)
        med_adj = np.median(adjusted_residuals)
        mad = np.median(np.abs(adjusted_residuals - med_adj))
        if mad == 0:
            mad = 1e-10 
        r_scaled = adjusted_residuals / (c * mad)

        weights = np.where(np.abs(r_scaled) < 1, (1 - r_scaled ** 2) ** 2, 0)

        W = np.diag(weights)
        XtWX = X.T @ W @ X
        XtWY = X.T @ W @ log_O

        if np.linalg.det(XtWX) < 1e-10:  
            break

        params = np.linalg.solve(XtWX, XtWY) 
        a_new = params[0]
        slope_new = params[1]
        b_new = -slope_new

        if np.abs(b_new - b_init) < 1e-6 and np.abs(a_new - a_init) < 1e-6:
            break

        b_init, a_init = b_new, a_new

    return b_init, a_init

def compute_S(M, a, b):
    """
    Синтетическое кумулятивное распределение S(M) = 10^(a - b M)
    """
    return 10 ** (a - b * M)


def compute_R(M, O, S):
    """
    Метрика R: процентное отклонение
    R = sum |log O - log S| / sum log O * 100%
    """
    log_O = np.log10(O + 1e-10)
    log_S = np.log10(S + 1e-10)
    numerator = np.sum(np.abs(log_O - log_S))
    denominator = np.sum(log_O)
    if denominator == 0:
        return np.inf
    return (numerator / denominator) * 100


def estimate_mc(magnitudes, bin_size=0.1, m_min=None, m_max=None, mi_step=0.1, max_mc=None):
    """
    Основная функция: оценка Mc по минимуму R с использованием RFM.

    Parameters:
    - magnitudes: list of magnitudes
    - bin_size: размер бина
    - mi_step: шаг для кандидатов Mi
    - max_mc: максимальное значение Mc (если None, то не ограничивается)

    Returns:
    - mc: оценённое Mc
    - R_values: dict {Mi: R}
    - bins, O_full: для plotting
    """
    bins_full, N_full, O_full = bin_magnitudes(magnitudes, bin_size, m_min, m_max)

    if m_min is None:
        m_min = np.min(magnitudes)
    if m_max is None:
        m_max = np.max(magnitudes)

    mi_candidates = np.arange(m_min, m_max - 0.5, mi_step)
    R_values = {}

    for mi in mi_candidates:
        valid_idx = bins_full >= mi
        if np.sum(valid_idx) < 3:  
            R_values[mi] = np.inf
            continue

        M_mi = bins_full[valid_idx]
        N_mi = N_full[valid_idx]
        O_mi = O_full[valid_idx]

        b, a = estimate_b_a_rfm(M_mi, O_mi)

        S_mi = compute_S(M_mi, a, b)

        R = compute_R(M_mi, O_mi, S_mi)
        R_values[mi] = R

    mi_array = np.array(list(R_values.keys()))
    r_array = np.array(list(R_values.values()))
    valid_r = r_array[np.isfinite(r_array)]
    if len(valid_r) == 0:
        return None, None, None, R_values, bins_full, O_full

    mc_idx = np.argmin(valid_r)
    mc = mi_array[np.isfinite(r_array)][mc_idx]

    # Ограничение Mc максимальным значением, если указано
    if max_mc is not None and mc > max_mc:
        mc = max_mc

    # Для выбранного mc, извлекаем a и b
    if mc is not None:
        # Повторяем оценку для mc (чтобы получить a и b)
        valid_idx = bins_full >= mc
        M_mc = bins_full[valid_idx]
        N_mc = N_full[valid_idx]
        O_mc = O_full[valid_idx]
        b, a = estimate_b_a_rfm(M_mc, O_mc)  # a и b для этого mc
    else:
        a, b = None, None

    return mc, a, b, R_values, bins_full, O_full

if __name__ == "__main__":
    # Создаем папку для графиков R_vs_Mi
    r_vs_mi_folder = "R_vs_Mi"
    if not os.path.exists(r_vs_mi_folder):
        os.makedirs(r_vs_mi_folder)

    engine = create_engine('postgresql://gis:123456@10.0.62.59:55432/gis')

    # SQL-запрос (фильтрация по году)
    query = """
    SELECT * 
    FROM earthquakes_gr_low.final_events_with_magnitude 
    WHERE EXTRACT(YEAR FROM event_dttm) BETWEEN 1960 AND 2021;
    """
    df = pd.read_sql_query(query, engine)
    df['year'] = pd.to_datetime(df['event_dttm']).dt.year

    mc_results = {}

    # Цикл по годам с 1960 по 2021
    for year in range(1960, 2022):
        # Фильтруем данные за текущий год
        magnitudes = df[df['year'] == year]['mag_value'].dropna()

        # Определяем max_mc: для годов >= 2000 ограничиваем до 4.0
        max_mc = 4.0 if year >= 2000 else None

        mc, a, b, R_values, bins, O_full = estimate_mc(magnitudes, bin_size=0.1, mi_step=0.1, max_mc=max_mc)

        # Сохраняем Mc, a, b в словарь
        if mc is not None:
            mc_results[year] = {'mc': mc, 'a': a, 'b': b}

        if mc is not None:
            print(f"Год {year}: Оценённое Mc: {mc:.1f}, a: {a:.2f}, b: {b:.2f}")
        else:
            print(f"Год {year}: Mc не оценено (недостаточно данных)")

        print(f"R для кандидатов: { {k: f'{v:.2f}%' for k, v in sorted(R_values.items()) if np.isfinite(v)} }")


        # График R(Mi) для текущего года
        mi_array = np.array(list(R_values.keys()))
        r_array = np.array(list(R_values.values()))
        plt.figure(figsize=(8, 5))
        plt.plot(mi_array, r_array, 'o-', label='R (RFM)')
        if mc is not None:
            plt.axvline(mc, color='r', linestyle='--', label=f'Mc = {mc:.1f}')
        plt.xlabel('Candidate Mi')
        plt.ylabel('R (%)')
        plt.title(f'R as function of Mi for {year} (Minimum at Mc)')
        plt.legend()
        plt.grid(True)
        
        # Сохраняем график в папку R_vs_Mi
        filename = os.path.join(r_vs_mi_folder, f'R_vs_Mi_{year}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    # Итоговый вывод: Mc, a, b по годам
    print("\nИтоговые Mc, a, b по годам:")
    for year, data in sorted(mc_results.items()):
        print(f"{year}: Mc={data['mc']:.1f}, a={data['a']:.2f}, b={data['b']:.2f}")

    # График Mc по годам
    years = [year for year, data in mc_results.items()]
    mcs = [data['mc'] for data in mc_results.values()]
    plt.figure(figsize=(10, 6))
    plt.plot(years, mcs, 'o-', color='blue', label='Mc (RFM)')
    plt.xlabel('Год')
    plt.ylabel('Mc')
    plt.title('Mc по годам (1960-2021)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Mc_vs_Year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График Mc по годам сохранён как Mc_vs_Year.png")

    # График b по годам
    years_b = [year for year, data in mc_results.items()]
    b_values = [data['b'] for data in mc_results.values()]
    plt.figure(figsize=(10, 6))
    plt.plot(years_b, b_values, 'o-', color='green', label='b (RFM)')
    plt.xlabel('Год')
    plt.ylabel('b-value')
    plt.title('b-values по годам (1960-2021)')
    plt.legend()
    plt.grid(True)
    plt.savefig('b_vs_Year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График b-values по годам сохранён как b_vs_Year.png")

    # График a по годам
    a_values = [data['a'] for data in mc_results.values()]
    plt.figure(figsize=(10, 6))
    plt.plot(years_b, a_values, 'o-', color='blue', label='a (RFM)')
    plt.xlabel('Год')
    plt.ylabel('a-value')
    plt.title('a-values по годам (1960-2021)')
    plt.legend()
    plt.grid(True)
    plt.savefig('a_vs_Year.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("График a-values по годам сохранён как a_vs_Year.png")

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame.from_dict(mc_results, orient='index')
    results_df.reset_index(inplace=True)
    results_df.columns = ['year', 'mc', 'a', 'b']
    results_df = results_df[['year', 'mc', 'a', 'b']]

    # Сохраняем в PostgreSQL
    try:
        results_df.to_sql(
            'values_results_rfm', 
            engine, 
            schema='earthquakes_gr_low', 
            if_exists='replace',
            index=False,
            dtype={
                'year': Integer(),
                'mc': Numeric(4,1),
                'a': Numeric(6,2),
                'b': Numeric(6,2)
            }
        )
        print("Результаты успешно сохранены в PostgreSQL")
    except Exception as e:
        print(f"Ошибка при сохранении в PostgreSQL: {e}")
