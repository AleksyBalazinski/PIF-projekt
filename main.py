from retirement_model import RetirementModel
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("--- Retirement Savings Optimization ---")

    model = RetirementModel(
        inflation=0.02,
        bond_return=0.03,
        salary_growth=0.05,
        t0=0,
        T=40,
        td=60
    )

    optimal_p = model.find_optimal_p()
    print(f"Simulation: {optimal_p * 100:.4f}%")

    p_exact = model.calculate_exact_p()
    print(f"Exact value: {p_exact * 100:.4f}%")

    p_list = np.linspace(0.01, 0.99)
    rem_balance_formula = (
        model.calculate_total_acc_wealth(
            p=p_list) - model.calculate_retirement_pv()
    ) * (1 + model.monthly_bond_return) ** model.months_retired
    rem_balance_sim = [model.simulate_finances(p=p) for p in p_list]
    plt.plot(p_list, rem_balance_formula, label='formula')
    plt.plot(p_list, rem_balance_sim, label='simulation', linestyle='--')
    plt.xlabel(r'$p$')
    plt.ylabel('Remaining balance at death')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
