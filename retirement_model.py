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
