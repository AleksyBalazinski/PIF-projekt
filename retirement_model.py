import numpy as np
from scipy.optimize import root_scalar


class RetirementModel:
    def __init__(self, inflation, bond_return, salary_growth, t0, T, td):
        self.inflation = inflation
        self.bond_return = bond_return
        self.salary_growth = salary_growth
        self.t0 = t0
        self.T = T
        self.td = td

        self.months_working = (self.T - self.t0) * 12
        self.months_retired = (self.td - self.T) * 12

        self.monthly_bond_return = (1 + self.bond_return)**(1/12) - 1
        self.monthly_inflation = (1 + self.inflation)**(1/12) - 1
        self.monthly_salary_growth = (1 + self.salary_growth)**(1/12) - 1

    def simulate_finances(self, p, initial_salary=1000):
        capital = 0
        current_salary = initial_salary

        # 1. Working Phase (Accumulation)
        for month in range(self.months_working):
            contribution = current_salary * p
            capital += contribution

            capital *= (1 + self.monthly_bond_return)

            current_salary *= (1 + self.monthly_salary_growth)

        current_pension = current_salary

        # 2. Retirement Phase (Distribution)
        for month in range(self.months_retired):
            capital -= current_pension

            capital *= (1 + self.monthly_bond_return)

            current_pension *= (1 + self.monthly_inflation)

        return capital

    def find_optimal_p(self):
        """
        Finds the required savings percentage 'p' so that capital at death is exactly 0.
        """
        result = root_scalar(
            lambda p: self.simulate_finances(p),
            bracket=[0.001, 0.9999],
            method='brentq'
        )
        return result.root

    def calculate_total_acc_wealth(self, p, intial_salary=1000):
        N = self.months_working
        k_indices = np.arange(1, N + 1)
        accumulation_base = (1 + self.monthly_salary_growth) / \
            (1 + self.monthly_bond_return)
        accumulation_sum = np.sum(accumulation_base ** k_indices)

        acc_wealth = ((1 + self.monthly_bond_return) ** N) * accumulation_sum
        acc_wealth /= (1 + self.monthly_salary_growth)
        acc_wealth *= p * intial_salary

        return acc_wealth

    def calculate_retirement_pv(self, intial_salary=1000):
        N = self.months_working
        M = self.months_retired

        j_indices = np.arange(1, M + 1)
        numerator_terms = ((1 + self.monthly_inflation) ** (j_indices - 1)) / \
            ((1 + self.monthly_bond_return) ** j_indices)
        distribution_sum = np.sum(numerator_terms)

        retirement_pv = ((1 + self.monthly_salary_growth)
                         ** N) * distribution_sum
        retirement_pv *= intial_salary

        return retirement_pv

    def calculate_exact_p(self):
        denominator = self.calculate_total_acc_wealth(p=1)
        numerator = self.calculate_retirement_pv()
        exact_p = numerator / denominator

        return exact_p

    def run_vectorized_mc(self, p, sw_start, sw_end, qu, qd, pu, initial_salary=1000, num_sims=10000, seed=0):
        rng = np.random.default_rng(seed)

        capital = np.zeros(num_sims)
        current_salary = np.full(num_sims, initial_salary, dtype=float)

        years_working = self.months_working // 12
        years_retired = self.months_retired // 12

        glide_path = np.linspace(sw_start, sw_end, years_working)

        for year in range(years_working):
            sw = glide_path[year]

            is_up_market = rng.random(num_sims) < pu
            annual_stock_return = np.where(is_up_market, qu, qd)

            annual_portfolio_return = (
                sw * annual_stock_return) + ((1 - sw) * self.bond_return)
            monthly_portfolio_return = (
                1 + annual_portfolio_return)**(1/12) - 1

            for month in range(12):
                capital += current_salary * p
                capital *= (1 + monthly_portfolio_return)
                current_salary *= (1 + self.monthly_salary_growth)

        current_pension = current_salary

        for year in range(years_retired):
            is_up_market = rng.random(num_sims) < pu
            annual_stock_return = np.where(is_up_market, qu, qd)

            annual_portfolio_return = (
                sw_end * annual_stock_return) + ((1 - sw_end) * self.bond_return)
            monthly_portfolio_return = (
                1 + annual_portfolio_return)**(1/12) - 1

            for month in range(12):
                capital -= current_pension
                capital *= (1 + monthly_portfolio_return)
                current_pension *= (1 + self.monthly_inflation)

        success_rate = np.mean(capital >= 0) * 100

        return capital, success_rate

    def optimize_portfolio(self, qu, qd, pu, start_weights, end_weights, target_success=95.0, num_sims=10000, verbose=False):
        if verbose:
            print("Starting portfolio optimization...")

        absolute_best_p = 1.0
        best_strategy = None

        for sw_start in start_weights:
            for sw_end in end_weights:
                if sw_end > sw_start:
                    continue

                low, high = 0.01, 0.99
                optimal_p_for_strategy = 1.0

                for _ in range(8):
                    mid_p = (low + high) / 2
                    _, success_rate = self.run_vectorized_mc(
                        p=mid_p, sw_start=sw_start, sw_end=sw_end,
                        qu=qu, qd=qd, pu=pu, num_sims=num_sims
                    )

                    if success_rate >= target_success:
                        optimal_p_for_strategy = mid_p
                        high = mid_p
                    else:
                        low = mid_p

                if verbose:
                    print(
                        f"Strategy [{sw_start*100:.0f}% -> {sw_end*100:.0f}%]: Necessary p = {optimal_p_for_strategy*100:.1f}%")

                if optimal_p_for_strategy < absolute_best_p:
                    absolute_best_p = optimal_p_for_strategy
                    best_strategy = (sw_start, sw_end)

        if verbose:
            print("-" * 40)
            print(f"Optimal portfolio found!")
            print(f"Starting weight: {best_strategy[0]*100}%")
            print(f"Retirement weight: {best_strategy[1]*100}%")
            print(f"Minimized p: {absolute_best_p*100:.2f}%")

        return absolute_best_p, best_strategy
