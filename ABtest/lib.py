from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm, ttest_ind, mannwhitneyu
from statsmodels.stats.power import tt_ind_solve_power
import matplotlib.pyplot as plt
import seaborn as sns


class ABtest:

    def __init__(self, control=None, test=None):
        """
        Инициализация выборок.

        :param control: list или np.array — контрольная выборка
        :param test: list или np.array — тестовая выборка
        """
        self.control = np.array(control)
        self.test = np.array(test)

    def get_test(self):
        return self.test

    def get_control(self):
        return self.control

    def generate(self, control_size, conv_ctrl, test_size, conv_test):
        """
        Метод для создания выборок на основе конверсий

        :param control_size: int
            Объем контрольной группы
        :param conv_ctrl: int
            Количество успехов в контрольной группе (conv_ctrl / control_size - конверсия)
        :param test_size: int
            Объем тестовой выборки
        :param conv_test: int
            Количество успехов в тестовой группе (conv_ctrl / control_size - конверсия)
        """
        # Генерируем тестовую выборку
        test_sample = np.array([1] * conv_test + [0] * (test_size - conv_test))
        np.random.shuffle(test_sample)

        # Генерируем контрольную выборку
        control_sample = np.array([1] * conv_ctrl + [0] * (control_size - conv_ctrl))
        np.random.shuffle(control_sample)

        self.control = control_sample
        self.test = test_sample

    def t_test(self, test=None, control=None, alternative='two-sided', equal_var=False, alpha=0.05, comparisons=1,
               power=0.8, plot=True):
        """
        Двухвыборочный t-тест.
        Рекомендуется использовать, если длина выборок превышает 10000 наблюдений, а дисперсии выборок равны.

        :param test: list или np.array
            Тестовая выборка.
        :param control: list или np.array
            Контрольная выборка.
        :param alternative: str, optional
            Тип альтернативной гипотезы, одно из {'two-sided'} (по умолчанию).
        :param equal_var: bool, optional
            Предположение о равенстве дисперсий выборок (по умолчанию False).
        :param alpha: float
            Уровень значимости
        :param comparisons: int
            Количество сравнений (для поправки Бонферрони)
        :param power: float
            Целевая мощность теста (зарезервировано для будущего)
        :return: dict с результатами теста

        Нулевая гипотеза: mean_control = mean_test
        Альтернативы:
            two-sided: mean_control != mean_test
            greater: mean_control > mean_test
        """
        test = np.array(test) if test is not None else self.test
        control = np.array(control) if control is not None else self.control

        stat, p_value = ttest_ind(control, test, equal_var=equal_var)

        corrected_alpha = alpha / comparisons

        # Обработка альтернативных гипотез
        if alternative == 'greater':
            # H1: mean_control >= mean_test  => test: mean_test > mean_control
            if stat > 0:
                p_value /= 2
            else:
                p_value = 1 - p_value / 2
        elif alternative == 'less':
            # H1: mean_control <= mean_test  => test: mean_test < mean_control
            if stat < 0:
                p_value /= 2
            else:
                p_value = 1 - p_value / 2
        elif alternative != 'two-sided':
            raise ValueError("Допустимые значения альтернативы: 'two-sided', 'greater', 'less'")

        result = {
            "statistic": stat,
            "p_value": p_value,
            "corrected_alpha": corrected_alpha,
            "passed": p_value < corrected_alpha,
            "test_mean": np.mean(test),
            "control_mean": np.mean(control),
            "alternative": alternative,
            "power": power  # зарезервировано
        }

        return result

    def z_test(self, test=None, control=None, alternative='two-sided',
               alpha=0.05, comparisons=1, power=0.8):
        """
        Z-тест для двух независимых выборок (большие выборки, известные или предполагаемые равные дисперсии).

        :param test: np.array или list,
            Тестовая выборка
        :param control: np.array или list,
            Контрольная выборка
        :param alternative: str
            'two-sided', 'greater', 'less'
        :param alpha: float
            Уровень значимости
        :param comparisons: int
            Количество сравнений (для поправки Бонферрони)
        :param power: float
            Целевая мощность теста (зарезервировано для будущего)
        :return:
            dict с результатами теста
        """

        test = np.array(test) if test is not None else self.test
        control = np.array(control) if control is not None else self.control

        n1, n2 = len(test), len(control)
        mean1, mean2 = np.mean(test), np.mean(control)
        var1, var2 = np.var(test, ddof=1), np.var(control, ddof=1)

        # Z-статистика
        se = np.sqrt(var1 / n1 + var2 / n2)
        z_stat = (mean1 - mean2) / se

        # p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        elif alternative == 'greater':
            p_value = 1 - norm.cdf(z_stat)
        elif alternative == 'less':
            p_value = norm.cdf(z_stat)
        else:
            raise ValueError("Допустимые значения альтернативы: 'two-sided', 'greater', 'less'")

        corrected_alpha = alpha / comparisons
        relative_uplift = ((mean1 - mean2) / mean2) * 100 if mean2 != 0 else np.nan

        result = {
            "z_statistic": z_stat,
            "p_value": p_value,
            "corrected_alpha": corrected_alpha,
            "passed": p_value < corrected_alpha,
            "test_mean": mean1,
            "control_mean": mean2,
            "relative_uplift": relative_uplift,
            "alternative": alternative,
            "power": power  # зарезервировано
        }

        return result

    def calculate_effect_size(self, p1=0.1, uplift=1.1):
        """
        Метод для подсчета размера аплифта/даунлифта (используется в методе вычисления продолжительности теста)

        :param p1: float
            Конверсии в контрольной выборке
        :param uplift: float
            Аплифт, который мы хотим получить
        """
        p2 = p1 * uplift
        p = (p1 + p2) / 2
        return (p2 - p1) / np.sqrt(p * (1 - p))

    def calculate_parametres(self, effect_size=None, nobs1=None, alpha=None, power=None, ratio=1.0,
                             alternative='two-sided'):
        """
        t-test для подсчета параметров теста.

        :effect_size: float
            Эффект, который мы хотим получить
        :nobs1: int
            Объем контрольной выборка
        :alpha: float
            Уровень значимости
        :power: float
            Уровень мощности критерия
        :ratio: float
            Соотношение тестовой и контрольной группы
        :alternative: str
            Альтернативная гипотеза ('two-sided', 'larger', 'smaller')
        :return: any
            Параметр, который был задан как None
        """
        return tt_ind_solve_power(effect_size, nobs1, alpha, power, ratio, alternative)

    def classic_bootstrap(self,
                          sample1,
                          sample2,
                          num_samples=10000,
                          alpha=0.05,
                          fix_seed=False):
        """
        Классический бутстрап для оценки разницы средних двух выборок.

        :param sample1: list или np.array
            Первая выборка данных.
        :param sample2: list или np.array
            Вторая выборка данных.
        :param num_samples: int, optional
            Количество бутстрэп-итераций (по умолчанию 10000).
        :param alpha: float, optional
            Уровень значимости для доверительного интервала (по умолчанию 0.05).
        :param fix_seed: bool, optional
            Если True, фиксируется случайное зерно для воспроизводимости (по умолчанию False).
        :return: tuple (p-value, доверительный интервал [low, high], массив бутстрап-значений разницы)
        """

        if fix_seed:
            np.random.seed(43)  # Фиксируем random seed для воспроизводимости

        n1, n2 = len(sample1), len(sample2)  # Размеры выборок
        # Фактическая разница средних
        observed_diff = np.mean(sample1) - np.mean(sample2)

        # Массив для хранения бутстрэп-разностей
        boot_diffs = np.zeros(num_samples)

        for i in range(num_samples):
            # Генерируем бутстрэп-выборки с заменой отдельно для каждой группы
            boot_sample1 = np.random.choice(sample1, size=n1, replace=True)
            boot_sample2 = np.random.choice(sample2, size=n2, replace=True)

            # Вычисляем разницу средних в бутстрэп-выборке
            boot_diffs[i] = np.mean(boot_sample1) - np.mean(boot_sample2)

        # Вычисляем доверительный интервал
        lower_bound, upper_bound = np.percentile(
            boot_diffs, [alpha / 2 * 100, (1 - alpha / 2) * 100])

        # p-value (двусторонний тест)
        p_value = 2 * np.mean((boot_diffs < 0)
                              if observed_diff > 0 else (boot_diffs > 0))

        return p_value, (lower_bound, upper_bound), boot_diffs

    def bootstrap_densities(self,
                            sample,
                            num_samples=10000,
                            alpha=0.05,
                            fix_seed=False):
        """
        Генерации новой выборки с помощью бутстрапа
        :param sample: list или np.array
            Выборка данных.
        :param num_samples: int, optional
            Количество бутстрэп-итераций (по умолчанию 10000).
        :param alpha: float, optional
            Уровень значимости для доверительного интервала (по умолчанию 0.05).
        :param fix_seed: bool, optional
            Если True, фиксируется случайное зерно для воспроизводимости (по умолчанию False).
        :return: tuple (p-value, доверительный интервал [low, high], массив бутстрап-значений разницы)
        """

        if fix_seed:
            np.random.seed(43)  # Фиксируем random seed для воспроизводимости

        n1 = len(sample)  # Размер выборки

        # Массив для хранения бутстрэп-средних
        boot_avgs = np.zeros(num_samples)

        for i in range(num_samples):
            # Генерируем бутстрэп-выборки с заменой отдельно для каждой группы
            boot_sample1 = np.random.choice(sample, size=n1, replace=True)
            boot_avgs[i] = np.mean(boot_sample1)

        return boot_avgs

    def visualize_distributions(self, sample1=None, sample2=None, alpha=0.05, comparisons=1, num_samples=10000,
                                fix_seed=False):

        def is_binary(data):
            return set(np.unique(data)).issubset({0, 1})

        def bootstrap_ci(data, alpha, n_bootstrap=10000, fix_seed=False):
            rng = np.random.default_rng(seed=42 if fix_seed else None)
            means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_bootstrap)]
            lower = np.percentile(means, 100 * (alpha / 2))
            upper = np.percentile(means, 100 * (1 - alpha / 2))
            return lower, upper

        sample1 = np.array(sample1) if sample1 is not None else self.control
        sample2 = np.array(sample2) if sample2 is not None else self.test

        if is_binary(sample1) and is_binary(sample2):
            sample1 = self.bootstrap_densities(sample1, len(sample1), alpha, fix_seed)
            sample2 = self.bootstrap_densities(sample2, len(sample2), alpha, fix_seed)

        corrected_alpha = alpha / comparisons
        ci1 = bootstrap_ci(sample1, corrected_alpha, num_samples, fix_seed)
        ci2 = bootstrap_ci(sample2, corrected_alpha, num_samples, fix_seed)

        plt.figure(figsize=(10, 5))

        # KDE
        sns.kdeplot(sample1, label='Контроль', color='blue', shade=True, alpha=0.3)
        sns.kdeplot(sample2, label='Тест', color='orange', shade=True, alpha=0.3)

        # CI линии
        for ci, color, label in zip([ci1, ci2], ['blue', 'orange'], ['Контроль', 'Тест']):
            plt.axvline(ci[0], color=color, linestyle='--', alpha=0.8,
                        label=f'{label} CI {int((1 - corrected_alpha) * 100)}%')
            plt.axvline(ci[1], color=color, linestyle='--', alpha=0.8)

        # Заштриховка области перекрытия
        overlap_left = max(ci1[0], ci2[0])
        overlap_right = min(ci1[1], ci2[1])
        if overlap_left < overlap_right:
            plt.axvspan(overlap_left, overlap_right, color='gray', alpha=0.2, label='Перекрытие CI')

        plt.title("Распределения выборок с доверительными интервалами")
        plt.xlabel("Значения")
        plt.ylabel("Плотность")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def trim_by_percentiles(self, lower=5, upper=95):
        """
        Обрезает значения выборок по заданным перцентилям.

        :param lower: float
            Нижний перцентиль (в процентах, от 0 до 100)
        :param upper: float
            Верхний перцентиль (в процентах, от 0 до 100)
        """
        if not (0 <= lower < upper <= 100):
            raise ValueError("Параметры lower и upper должны быть в диапазоне [0, 100] и lower < upper")

        def trim_sample(sample, lower, upper):
            lower_bound = np.percentile(sample, lower)
            upper_bound = np.percentile(sample, upper)
            return sample[(sample >= lower_bound) & (sample <= upper_bound)]

        self.control = trim_sample(self.control, lower, upper)
        self.test = trim_sample(self.test, lower, upper)

