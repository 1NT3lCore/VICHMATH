import math

def create_funtction(preobrazovanie, x0, epsilon=0.00001, max_iterations=100):
    print("\nМЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print(f"Начальное приближение: x0 = {x0}")
    print(f"Точность: ε = {epsilon:.6f}")
    print(f"Максимальное число итераций: {max_iterations}")

    x_old = x0
    story_list = [x0]
    iterations = 0
    for i in range(max_iterations):
        x_new = preobrazovanie(x_old)
        iterations += 1
        story_list.append(x_new)
        solve = abs(x_new - x_old)
        print(f"Итерация {i+1}: x = {x_new:.8f}, |△x| = {solve:.8f}")

        if solve < epsilon:
            print("\n")
            print(f"Приближенное значение корня x: {x_new:.8f}")
            print(f"Решение найдено за {iterations} итераций")
            print(f"Точность: {solve:.8f}")
            return x_new, iterations, story_list
        x_old = x_new

    print(f"Достигнут максимум итераций ({max_iterations})")
    print(f"Лучшее приближение: x ≈ {x_old:.8f}")
    return x_old, max_iterations, story_list


def solve_equation_1():
    """Решение уравнения: ln(x) + (x+1)^3 = 0"""
    print("\n" + "=" * 60)
    print("УРАВНЕНИЕ 1: ln(x) + (x+1)^3 = 0")
    print("=" * 60)

    # Преобразуем уравнение к виду x = φ(x)
    # ln(x) + (x+1)^3 = 0 → ln(x) = -(x+1)^3 → x = e^(-(x+1)^3)
    def preobrazovanie(x):
        return math.exp(-(x + 1) ** 3)

    # Проверим условие сходимости |φ'(x)| < 1
    # φ'(x) = -3(x+1)^2 * e^(-(x+1)^3)
    def preobrazovanie_derivative(x):
        return -3 * (x + 1) ** 2 * math.exp(-(x + 1) ** 3)

    print("Проверка сходимости:")
    print(f"|φ'(0.5)| = {abs(preobrazovanie_derivative(0.5)):.4f}")
    print(f"|φ'(1.0)| = {abs(preobrazovanie_derivative(1.0)):.4f}")

    # Запускаем метод
    root, iterations, story_list = create_funtction(preobrazovanie, x0=0.5, epsilon=1e-6)

    # Проверка результата
    result_check = math.log(root) + (root + 1) ** 3
    print(f"🔍 Проверка: ln({root:.6f}) + ({root:.6f}+1)³ = {result_check:.2e}")

    return root, iterations, story_list


def solve_equation_2():
    """Решение уравнения: sin(0.5x) + 1 = x^2"""
    print("\n" + "=" * 60)
    print("УРАВНЕНИЕ 2: sin(0.5x) + 1 = x²")
    print("=" * 60)

    def preobrazovanie(x):
        return math.sqrt(math.sin(0.5 * x) + 1)

    def preobrazovanie_alternative(x):
        if x == 0:
            return 1.0
        return (math.sin(0.5 * x) + 1) / x

    print("Пробуем первый вариант преобразования: x = √(sin(0.5x) + 1)")

    initial_guesses = [0.5, 1.0, 1.5]

    for x0 in initial_guesses:
        print(f"\n--- Попытка с x0 = {x0} ---")
        try:
            root, iterations, story_list = create_funtction(preobrazovanie, x0=x0, epsilon=1e-6)

            result_check = math.sin(0.5 * root) + 1 - root ** 2
            print(f"🔍 Проверка: sin(0.5×{root:.6f}) + 1 - ({root:.6f})² = {result_check:.2e}")

            if abs(result_check) < 1e-4:
                return root, iterations, story_list

        except (ValueError, ZeroDivisionError) as e:
            print(f"❌ Ошибка: {e}")
            continue

    print("\n⚠️  Первый метод не дал хорошего результата, пробуем альтернативный...")

    def preobrazovanie2(x):
        return (math.sin(0.5 * x) + 1) / x if x != 0 else 1.0

    root, iterations, story_list = create_funtction(preobrazovanie2, x0=1.0, epsilon=1e-6)

    result_check = math.sin(0.5 * root) + 1 - root ** 2
    print(f"🔍 Проверка: sin(0.5×{root:.6f}) + 1 - ({root:.6f})² = {result_check:.2e}")

    return root, iterations, story_list


def main():
    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print("Решение нелинейных уравнений")
    print("=" * 60)

    root1, iter1, story_list1 = solve_equation_1()

    root2, iter2, story_list2 = solve_equation_2()

    print("\n" + "=" * 60)
    print("ИТОГИ РЕШЕНИЯ:")
    print("=" * 60)
    print(f"Уравнение 1: ln(x) + (x+1)³ = 0")
    print(f"Корень: x ≈ {root1:.8f}")
    print(f"Итераций: {iter1}")
    print(f"Проверка: {math.log(root1) + (root1 + 1) ** 3:.2e}")

    print(f"\nУравнение 2: sin(0.5x) + 1 = x²")
    print(f"Корень: x ≈ {root2:.8f}")
    print(f"Итераций: {iter2}")
    print(f"Проверка: {math.sin(0.5 * root2) + 1 - root2 ** 2:.2e}")

if __name__ == "__main__":
    main()