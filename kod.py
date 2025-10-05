import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve, brentq


def format_table(headers, data):
    """Форматирование таблицы без использования tabulate"""
    # Определяем ширину колонок
    col_widths = []
    for i in range(len(headers)):
        col_width = max(len(str(headers[i])),
                        max(len(str(row[i])) for row in data) if data else 0)
        col_widths.append(col_width + 2)  # Добавляем отступы

    # Создаем разделительную строку
    separator = "+" + "+".join("-" * width for width in col_widths) + "+"

    # Формируем таблицу
    table = []
    table.append(separator)

    # Заголовки
    header_row = "|" + "|".join(f" {headers[i]:<{col_widths[i] - 1}}" for i in range(len(headers))) + "|"
    table.append(header_row)
    table.append(separator)

    # Данные
    for row in data:
        data_row = "|" + "|".join(f" {str(row[i]):<{col_widths[i] - 1}}" for i in range(len(row))) + "|"
        table.append(data_row)

    table.append(separator)
    return "\n".join(table)


def postroit_sravnenie(f_array, f_scalar, equation_name, roots_bisec, roots_iter, roots_scipy, segments, x0_list):
    plt.figure(figsize=(15, 6))
    all_points = []

    # Собираем все точки для определения границ графика
    if segments:
        for a, b in segments:
            all_points.extend([a, b])
    if roots_bisec:
        all_points.extend(roots_bisec)
    if roots_iter:
        all_points.extend(roots_iter)
    if roots_scipy:
        all_points.extend(roots_scipy)
    if x0_list:
        all_points.extend(x0_list)

    if not all_points:
        all_points = [-2, 2]

    x_min, x_max = min(all_points) - 0.5, max(all_points) + 0.5
    x = np.linspace(x_min, x_max, 1000)

    # Для построения графика используем numpy-совместимые функции
    y = f_array(x)

    # Левый график - метод половинного деления
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Отмечаем отрезки
    if segments:
        for i, (a, b) in enumerate(segments):
            plt.axvspan(a, b, alpha=0.2, color='red', label=f'Отрезок {i + 1}' if i == 0 else "")

    # Корни метода половинного деления
    if roots_bisec:
        for i, root in enumerate(roots_bisec):
            try:
                # Используем скалярную функцию для вычисления значения в точке корня
                y_root = f_scalar(root)
                plt.plot(root, y_root, 'ro', markersize=8,
                         label=f'Корень (полов. деления): {root:.10f}' if i == 0 else f'Корень: {root:.10f}')
            except (ValueError, ZeroDivisionError):
                continue

    # Корни SciPy
    if roots_scipy:
        for i, root in enumerate(roots_scipy):
            try:
                # Используем скалярную функцию для вычисления значения в точке корня
                y_root = f_scalar(root)
                plt.plot(root, y_root, 'm*', markersize=12,
                         label=f'Корень (SciPy): {root:.10f}' if i == 0 else f'SciPy: {root:.10f}')
            except (ValueError, ZeroDivisionError):
                continue

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Метод половинного деления\n{equation_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(x_min, x_max)

    # Правый график - метод простой итерации
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # Начальные приближения
    if x0_list:
        for i, x0 in enumerate(x0_list):
            plt.axvline(x=x0, color='orange', linestyle='--', alpha=0.7,
                        label=f'Начальное x0: {x0:.10f}' if i == 0 else "")

    # Корни метода простой итерации
    if roots_iter:
        for i, root in enumerate(roots_iter):
            try:
                # Используем скалярную функцию для вычисления значения в точке корня
                y_root = f_scalar(root)
                plt.plot(root, y_root, 'go', markersize=8,
                         label=f'Корень (итерации): {root:.10f}' if i == 0 else f'Корень: {root:.10f}')
            except (ValueError, ZeroDivisionError):
                continue

    # Корни SciPy
    if roots_scipy:
        for i, root in enumerate(roots_scipy):
            try:
                # Используем скалярную функцию для вычисления значения в точке корня
                y_root = f_scalar(root)
                plt.plot(root, y_root, 'm*', markersize=12,
                         label=f'Корень (SciPy): {root:.10f}' if i == 0 else f'SciPy: {root:.10f}')
            except (ValueError, ZeroDivisionError):
                continue

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Метод простой итерации\n{equation_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(x_min, x_max)

    plt.tight_layout()
    plt.show()


def bezopasnaya_funkciya(f, x):
    """Безопасное вычисление функции с обработкой ошибок"""
    try:
        return f(x)
    except (ValueError, ZeroDivisionError):
        return float('inf') if x > 0 else float('-inf')


def metod_polovinnogo_deleniya(f, a, b, epsilon=0.00001, max_iterations=100):
    print("\n" + "=" * 60)
    print("МЕТОД ПОЛОВИННОГО ДЕЛЕНИЯ".center(60))
    print("=" * 60)

    # Безопасная проверка знаков
    fa = bezopasnaya_funkciya(f, a)
    fb = bezopasnaya_funkciya(f, b)

    # Проверка на неопределенность функции
    if abs(fa) == float('inf') or abs(fb) == float('inf'):
        print(f"ОШИБКА: Функция не определена на границах отрезка!")
        print(f"f({a:.10f}) = {'не опред.' if abs(fa) == float('inf') else f'{fa:.10f}'}")
        print(f"f({b:.10f}) = {'не опред.' if abs(fb) == float('inf') else f'{fb:.10f}'}")
        return None, 0, []

    if fa * fb > 0:
        print("ОШИБКА: f(a) и f(b) имеют одинаковые знаки!")
        print(f"f({a:.10f}) = {fa:.10f}")
        print(f"f({b:.10f}) = {fb:.10f}")
        return None, 0, []

    iterations = 0
    history = []

    # Создаем таблицу для итераций
    table_data = []
    headers = ["Итерация", "a", "b", "c", "f(c)", "|b-a|"]

    for i in range(max_iterations):
        c = (a + b) / 2
        fc = bezopasnaya_funkciya(f, c)
        iterations += 1
        history.append(c)
        interval_length = abs(b - a)

        # Форматируем значения с точностью 10 знаков
        fc_str = f"{fc:.10f}" if abs(fc) != float('inf') else "не опред."

        table_data.append([
            i + 1,
            f"{a:.10f}",
            f"{b:.10f}",
            f"{c:.10f}",
            fc_str,
            f"{interval_length:.10f}"
        ])

        if abs(fc) < epsilon or interval_length < epsilon:
            break

        # Безопасное определение нового отрезка
        fa = bezopasnaya_funkciya(f, a)
        if fa * fc < 0:
            b = c
        else:
            a = c
    else:
        c = (a + b) / 2
        print("Достигнут максимум итераций")

    print(format_table(headers, table_data))

    print(f"\nРешение найдено за {iterations} итераций!")
    print(f"Приближенный корень: x = {c:.10f}")
    final_fc = bezopasnaya_funkciya(f, c)
    fc_str = f"{final_fc:.10f}" if abs(final_fc) != float('inf') else "не опред."
    print(f"f(x) = {fc_str}")

    return c, iterations, history


def prostoy_metod(phi, x0, epsilon=0.00001, max_iterations=100):
    print("\n" + "=" * 60)
    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ".center(60))
    print("=" * 60)

    x_old = x0
    story_list = [x0]
    iterations = 0

    table_data = []
    headers = ["Итерация", "x_old", "x_new", "|dx|"]

    for i in range(max_iterations):
        try:
            x_new = phi(x_old)
        except (ValueError, ZeroDivisionError):
            print(f"Ошибка при вычислении phi(x) на итерации {i + 1}")
            break

        iterations += 1
        story_list.append(x_new)
        dx = abs(x_new - x_old)

        table_data.append([
            i + 1,
            f"{x_old:.10f}",
            f"{x_new:.10f}",
            f"{dx:.10f}"
        ])

        if dx < epsilon:
            break

        x_old = x_new
    else:
        print("Достигнут максимум итераций")

    print(format_table(headers, table_data))

    print(f"\nРешение найдено за {iterations} итераций!")
    print(f"Приближенный корень: x = {x_new:.10f}")

    return x_new, iterations, story_list


def reshenie_scipy(f_scalar, segments, x0_list):
    """Решение уравнения с помощью SciPy"""
    print("\n" + "=" * 60)
    print("РЕШЕНИЕ С ПОМОЩЬЮ SCIPY".center(60))
    print("=" * 60)

    roots = []
    table_data = []

    # Создаем numpy-совместимую функцию для SciPy
    def f_numpy(x):
        if hasattr(x, '__iter__'):
            # Если x - массив, обрабатываем каждый элемент
            result = []
            for xi in x:
                try:
                    result.append(float(f_scalar(xi)))
                except (ValueError, ZeroDivisionError):
                    result.append(float('inf'))
            return np.array(result)
        else:
            # Если x - скаляр
            try:
                return float(f_scalar(x))
            except (ValueError, ZeroDivisionError):
                return float('inf')

    # Используем brentq для отрезков (более надежный)
    if segments:
        for i, (a, b) in enumerate(segments):
            try:
                # Проверяем, что функция определена на концах отрезка
                fa = f_numpy(a)
                fb = f_numpy(b)
                if abs(fa) == float('inf') or abs(fb) == float('inf'):
                    print(f"Функция не определена на отрезке [{a:.10f}, {b:.10f}]")
                    continue

                root = brentq(f_numpy, a, b)
                roots.append(root)
                # Проверяем, что функция определена в найденном корне
                try:
                    f_root = f_scalar(root)
                    f_root_str = f"{f_root:.10f}"
                except (ValueError, ZeroDivisionError):
                    f_root_str = "---"

                table_data.append([
                    f"Отрезок {i + 1}",
                    f"[{a:.10f}, {b:.10f}]",
                    f"{root:.10f}",
                    f_root_str
                ])
            except Exception as e:
                print(f"Ошибка для отрезка [{a:.10f}, {b:.10f}]: {e}")

    # Используем fsolve для начальных приближений
    if x0_list:
        for i, x0 in enumerate(x0_list):
            try:
                # Проверяем, что функция определена в начальной точке
                fx0 = f_numpy(x0)
                if abs(fx0) == float('inf'):
                    print(f"Функция не определена в точке x0 = {x0:.10f}")
                    continue

                root = fsolve(f_numpy, x0)[0]
                roots.append(root)
                # Проверяем, что функция определена в найденном корне
                try:
                    f_root = f_scalar(root)
                    f_root_str = f"{f_root:.10f}"
                except (ValueError, ZeroDivisionError):
                    f_root_str = "---"

                table_data.append([
                    f"Нач. приближение {i + 1}",
                    f"x0 = {x0:.10f}",
                    f"{root:.10f}",
                    f_root_str
                ])
            except Exception as e:
                print(f"Ошибка для x0 = {x0:.10f}: {e}")

    if table_data:
        headers = ["Метод", "Параметры", "Корень", "f(x)"]
        print(format_table(headers, table_data))
    else:
        print("Не удалось найти корни с помощью SciPy")

    return roots


def poluchit_otrezki_ot_polzovatelya():
    """Получение отрезков от пользователя"""
    print("\nВВОД ОТРЕЗКОВ ДЛЯ ПОИСКА КОРНЕЙ")
    print("Введите отрезки в формате: a1 b1, a2 b2, ...")
    print("Пример: 0.1 0.5, -1.0 1.0")
    print("ВАЖНО: Для уравнения ln(x) + (x+1)^3 = 0 отрезок должен быть > 0")

    while True:
        try:
            segments_input = input("Введите отрезки: ").strip()
            if not segments_input:
                return []

            segments = []
            pairs = segments_input.split(',')

            for pair in pairs:
                a, b = map(float, pair.strip().split())

                # Проверка для первого уравнения (логарифм)
                if a <= 0 or b <= 0:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Для уравнения с ln(x) отрезок должен быть > 0")
                    print(f"Введен отрезок [{a:.10f}, {b:.10f}] - будут использованы положительные значения")
                    a = max(0.001, a)  # Минимальное положительное значение
                    b = max(0.001, b)

                segments.append((a, b))

            print("Введенные отрезки:")
            for i, (a, b) in enumerate(segments):
                print(f"   Отрезок {i + 1}: [{a:.10f}, {b:.10f}]")

            confirm = input("Все верно? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', 'д', 'да']:
                return segments
            else:
                print("Повторите ввод отрезков.")

        except ValueError:
            print("Ошибка формата! Используйте формат: число1 число2, число3 число4, ...")
        except Exception as e:
            print(f"Ошибка: {e}")


def poluchit_nachalnye_priblizheniya():
    """Получение начальных приближений от пользователя"""
    print("\nВВОД НАЧАЛЬНЫХ ПРИБЛИЖЕНИЙ")
    print("Введите начальные приближения через пробел")
    print("Пример: 0.3 1.0 -1.0")
    print("ВАЖНО: Для уравнения ln(x) + (x+1)^3 = 0 x0 должен быть > 0")

    while True:
        try:
            x0_input = input("Введите начальные приближения: ").strip()
            if not x0_input:
                return []

            x0_list = list(map(float, x0_input.split()))

            # Проверка для первого уравнения
            corrected_x0_list = []
            for x0 in x0_list:
                if x0 <= 0:
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Для уравнения с ln(x) x0 должен быть > 0")
                    print(f"Введено x0 = {x0:.10f} - будет использовано значение 0.001")
                    corrected_x0_list.append(0.001)
                else:
                    corrected_x0_list.append(x0)

            print("Введенные начальные приближения:")
            for i, x0 in enumerate(corrected_x0_list):
                print(f"   x0{i + 1} = {x0:.10f}")

            confirm = input("Все верно? (y/n): ").strip().lower()
            if confirm in ['y', 'yes', 'д', 'да']:
                return corrected_x0_list
            else:
                print("Повторите ввод начальных приближений.")

        except ValueError:
            print("Ошибка формата! Используйте только числа, разделенные пробелами.")
        except Exception as e:
            print(f"Ошибка: {e}")


def solve_equation_1():
    print("\n" + "=" * 80)
    print("УРАВНЕНИЕ 1: ln(x) + (x+1)^3 = 0".center(80))
    print("=" * 80)

    # Функция для вычислений (работает со скалярами)
    def f_scalar(x):
        if x <= 0:
            return float('inf')  # Логарифм не определен для x <= 0
        return math.log(x) + (x + 1) ** 3

    # Функция для графиков (работает с массивами)
    def f_array(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.log(x) + (x + 1) ** 3
            # Для массива используем векторные операции
            result = np.where(np.isnan(result), np.inf, result)
            return result

    def phi(x):
        # Для метода простой итерации используем преобразование x = e^(-(x+1)^3)
        if x <= 0:
            return 0.001  # Защита от отрицательных значений
        return math.exp(-(x + 1) ** 3)

    # Получаем данные от пользователя
    segments = poluchit_otrezki_ot_polzovatelya()
    if not segments:
        # Используем более безопасный отрезок
        segments = [(0.1, 0.5)]
        print(f"Используется отрезок по умолчанию: [{segments[0][0]:.10f}, {segments[0][1]:.10f}]")

    x0_list = poluchit_nachalnye_priblizheniya()
    if not x0_list:
        x0_list = [0.3]
        print(f"Используется начальное приближение по умолчанию: x0 = {x0_list[0]:.10f}")

    # Метод половинного деления (используем скалярную функцию)
    roots_bisec = []
    total_iter_bisec = 0
    all_hist_bisec = []

    for i, (a, b) in enumerate(segments):
        print(f"\nОбработка отрезка {i + 1}: [{a:.10f}, {b:.10f}]")
        root, iterations, history = metod_polovinnogo_deleniya(f_scalar, a, b)
        if root is not None:
            roots_bisec.append(root)
            total_iter_bisec += iterations
            all_hist_bisec.extend(history)

    # Метод простой итерации
    roots_iter = []
    total_iter_iter = 0
    all_hist_iter = []

    for i, x0 in enumerate(x0_list):
        print(f"\nОбработка начального приближения {i + 1}: x0 = {x0:.10f}")
        root, iterations, history = prostoy_metod(phi, x0=x0)
        if root is not None:
            roots_iter.append(root)
            total_iter_iter += iterations
            all_hist_iter.extend(history)

    # Решение с помощью SciPy
    roots_scipy = reshenie_scipy(f_scalar, segments, x0_list)

    # Графическое сравнение (передаем обе функции)
    print("\nГРАФИЧЕСКОЕ СРАВНЕНИЕ МЕТОДОВ")
    postroit_sravnenie(f_array, f_scalar, "ln(x) + (x+1)^3 = 0", roots_bisec, roots_iter, roots_scipy, segments,
                       x0_list)

    return (roots_bisec, total_iter_bisec, all_hist_bisec), \
        (roots_iter, total_iter_iter, all_hist_iter), \
        roots_scipy


def solve_equation_2():
    print("\n" + "=" * 80)
    print("УРАВНЕНИЕ 2: sin(0.5x) + 1 = x^2".center(80))
    print("=" * 80)

    # Функция для вычислений (работает со скалярами)
    def f_scalar(x):
        return math.sin(0.5 * x) + 1 - x ** 2

    # Функция для графиков (работает с массивами)
    def f_array(x):
        return np.sin(0.5 * x) + 1 - x ** 2

    def phi_positive(x):
        # Для положительного корня: x = sqrt(sin(0.5x) + 1)
        return math.sqrt(math.sin(0.5 * x) + 1)

    def phi_negative(x):
        # Для отрицательного корня: x = -sqrt(sin(0.5x) + 1)
        return -math.sqrt(math.sin(0.5 * x) + 1)

    # Получаем данные от пользователя
    segments = poluchit_otrezki_ot_polzovatelya()
    if not segments:
        segments = [(-1.5, -0.5), (0.5, 1.5)]
        print("Используются отрезки по умолчанию:")
        for i, (a, b) in enumerate(segments):
            print(f"   Отрезок {i + 1}: [{a:.10f}, {b:.10f}]")

    x0_list = poluchit_nachalnye_priblizheniya()
    if not x0_list:
        x0_list = [-1.0, 1.0]
        print("Используются начальные приближения по умолчанию:")
        for i, x0 in enumerate(x0_list):
            print(f"   x0{i + 1} = {x0:.10f}")

    # Метод половинного деления (используем скалярную функцию)
    roots_bisec = []
    total_iter_bisec = 0
    all_hist_bisec = []

    for i, (a, b) in enumerate(segments):
        print(f"\nОбработка отрезка {i + 1}: [{a:.10f}, {b:.10f}]")
        root, iterations, history = metod_polovinnogo_deleniya(f_scalar, a, b)
        if root is not None:
            roots_bisec.append(root)
            total_iter_bisec += iterations
            all_hist_bisec.extend(history)

    # Метод простой итерации
    roots_iter = []
    total_iter_iter = 0
    all_hist_iter = []

    for i, x0 in enumerate(x0_list):
        print(f"\nОбработка начального приближения {i + 1}: x0 = {x0:.10f}")
        if x0 >= 0:
            root, iterations, history = prostoy_metod(phi_positive, x0=x0)
        else:
            root, iterations, history = prostoy_metod(phi_negative, x0=x0)

        if root is not None:
            roots_iter.append(root)
            total_iter_iter += iterations
            all_hist_iter.extend(history)

    # Решение с помощью SciPy
    roots_scipy = reshenie_scipy(f_scalar, segments, x0_list)

    # Графическое сравнение (передаем обе функции)
    print("\nГРАФИЧЕСКОЕ СРАВНЕНИЕ МЕТОДОВ")
    postroit_sravnenie(f_array, f_scalar, "sin(0.5x) + 1 = x^2", roots_bisec, roots_iter, roots_scipy, segments,
                       x0_list)

    return (roots_bisec, total_iter_bisec, all_hist_bisec), \
        (roots_iter, total_iter_iter, all_hist_iter), \
        roots_scipy


def sravnit_rezultaty(equation_name, roots_bisec, iter_bisec, roots_iter, iter_iter, roots_scipy, f):
    """Сравнение результатов всех методов"""
    print("\n" + "=" * 80)
    print(f"СРАВНЕНИЕ РЕЗУЛЬТАТОВ ({equation_name})".center(80))
    print("=" * 80)

    table_data = []

    # Добавляем результаты метода половинного деления
    for i, root in enumerate(roots_bisec):
        try:
            f_root = f(root)
            table_data.append([
                f"Половинное деление {i + 1}",
                f"{root:.10f}",
                f"{f_root:.10f}",
                iter_bisec
            ])
        except (ValueError, ZeroDivisionError):
            table_data.append([
                f"Половинное деление {i + 1}",
                f"{root:.10f}",
                "---",
                iter_bisec
            ])

    # Добавляем результаты метода простой итерации
    for i, root in enumerate(roots_iter):
        try:
            f_root = f(root)
            table_data.append([
                f"Простая итерация {i + 1}",
                f"{root:.10f}",
                f"{f_root:.10f}",
                iter_iter
            ])
        except (ValueError, ZeroDivisionError):
            table_data.append([
                f"Простая итерация {i + 1}",
                f"{root:.10f}",
                "---",
                iter_iter
            ])

    # Добавляем результаты SciPy
    for i, root in enumerate(roots_scipy):
        try:
            f_root = f(root)
            table_data.append([
                f"SciPy {i + 1}",
                f"{root:.10f}",
                f"{f_root:.10f}",
                "-"
            ])
        except (ValueError, ZeroDivisionError):
            table_data.append([
                f"SciPy {i + 1}",
                f"{root:.10f}",
                "---",
                "-"
            ])

    headers = ["Метод", "Корень", "f(x)", "Итерации"]
    print(format_table(headers, table_data))

    # Анализ эффективности методов
    print("\nАНАЛИЗ ЭФФЕКТИВНОСТИ МЕТОДОВ:")

    if roots_bisec and roots_iter:
        print(f"Метод половинного деления: {iter_bisec} итераций")
        print(f"Метод простой итерации: {iter_iter} итераций")

        if iter_bisec < iter_iter:
            print("Метод половинного деления сходится быстрее")
        elif iter_iter < iter_bisec:
            print("Метод простой итерации сходится быстрее")
        else:
            print("Оба метода сошлись за одинаковое количество итераций")

    if roots_scipy:
        print("SciPy методы показывают высокую точность и надежность")


def main():
    print("СРАВНЕНИЕ МЕТОДОВ РЕШЕНИЯ НЕЛИНЕЙНЫХ УРАВНЕНИЙ")
    print("Включая сравнение с SciPy")
    print("Точность вычислений: 0.00001")

    # Уравнение 1
    result1_bisec, result1_iter, result1_scipy = solve_equation_1()
    roots1_bisec, iter1_bisec, hist1_bisec = result1_bisec
    roots1_iter, iter1_iter, hist1_iter = result1_iter

    def f1(x):
        if x <= 0:
            return float('inf')
        return math.log(x) + (x + 1) ** 3

    sravnit_rezultaty("Уравнение 1", roots1_bisec, iter1_bisec, roots1_iter, iter1_iter, result1_scipy, f1)

    # Уравнение 2
    result2_bisec, result2_iter, result2_scipy = solve_equation_2()
    roots2_bisec, iter2_bisec, hist2_bisec = result2_bisec
    roots2_iter, iter2_iter, hist2_iter = result2_iter

    def f2(x):
        return math.sin(0.5 * x) + 1 - x ** 2

    sravnit_rezultaty("Уравнение 2", roots2_bisec, iter2_bisec, roots2_iter, iter2_iter, result2_scipy, f2)

    print("\n" + "=" * 80)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()