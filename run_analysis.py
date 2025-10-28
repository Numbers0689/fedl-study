import matplotlib.pyplot as plt
import time
import os

from main import run_simulation

experiments = [
    {
        'label': 'IID Baseline',
        'params': {'data_dist': 'iid', 'C': 0.1, 'E': 5, 'B': 32}
    },
    {
        'label': 'Non-IID Baseline (E=5, C=0.1)',
        'params': {'data_dist': 'non-iid', 'C': 0.1, 'E': 5, 'B': 32}
    },
    {
        'label': 'Non-IID (High C, C=0.5)',
        'params': {'data_dist': 'non-iid', 'C': 0.5, 'E': 5, 'B': 32}
    },
    {
        'label': 'Non-IID (High E, E=10)',
        'params': {'data_dist': 'non-iid', 'C': 0.1, 'E': 10, 'B': 32}
    },
    {
        'label': 'Non-IID (Low E, E=1)',
        'params': {'data_dist': 'non-iid', 'C': 0.1, 'E': 1, 'B': 32}
    },
    {
        'label': 'Non-IID (Combined Mitigation, C=0.5, E=1)',
        'params': {'data_dist': 'non-iid', 'C': 0.5, 'E': 1, 'B': 32}
    }
]

results_history = {}
summary_data = []
total_start_time = time.time()

output_dir = "simulation_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Starting full simulation and analysis pipeline...")
print(f"Results will be saved in '{output_dir}/' directory.\n")

for exp in experiments:
    label = exp['label']
    params = exp['params']
    
    print(f"--- Running Experiment: {label} ---")
    start_time = time.time()
    
    history = run_simulation(
        num_rounds=50,
        num_clients=100,
        C=params['C'],
        E=params['E'],
        B=params['B'],
        data_dist=params['data_dist']
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    results_history[label] = history
    summary_data.append({
        'Experiment': label,
        'Final Accuracy (%)': history[-1],
        'Time (s)': elapsed,
        'C': params['C'],
        'E': params['E'],
        'Dist': params['data_dist']
    })
    
    print(f"--- Finished in {elapsed:.2f}s. Final Accuracy: {history[-1]:.2f}% --- \n")

print(f"All experiments finished in {time.time() - total_start_time:.2f}s.\n")

print("--- Simulation Results Summary (Table 1) ---")
print("| Experiment                                | Dist    | C   | E  | Final Acc. (%) | Time (s) |")
print("|-------------------------------------------|---------|-----|----|----------------|----------|")
for item in summary_data:
    print(f"| {item['Experiment']:<41} | {item['Dist']:<7} | {item['C']:<3} | {item['E']:<2} | {item['Final Accuracy (%)']:>14.2f} | {item['Time (s)']:>8.2f} |")
print("\n")

print("Generating and saving plots...")
rounds = range(1, 51)
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 6))
plt.plot(rounds, results_history['IID Baseline'], label='IID Baseline', marker='s', markersize=4)
plt.plot(rounds, results_history['Non-IID Baseline (E=5, C=0.1)'], label='Non-IID Baseline', marker='o', markersize=4)
plt.title('Experiment 1: IID vs. Non-IID Baseline', fontsize=16)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Global Model Accuracy (%)', fontsize=12)
plt.legend(fontsize=11)
plt.ylim(0, 100)
plt.savefig(os.path.join(output_dir, 'fig_1_iid_vs_non_iid.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'fig_1_iid_vs_non_iid.png')}")

plt.figure(figsize=(10, 6))
plt.plot(rounds, results_history['Non-IID Baseline (E=5, C=0.1)'], label='Non-IID Baseline (C=0.1)', marker='o', linestyle='--')
plt.plot(rounds, results_history['Non-IID (High C, C=0.5)'], label='Non-IID (High C=0.5)', marker='^')
plt.title('Experiment 2: Impact of Client Fraction (C) in Non-IID Setting', fontsize=16)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Global Model Accuracy (%)', fontsize=12)
plt.legend(fontsize=11)
plt.ylim(0, 100)
plt.savefig(os.path.join(output_dir, 'fig_2_impact_of_C.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'fig_2_impact_of_C.png')}")

plt.figure(figsize=(10, 6))
plt.plot(rounds, results_history['Non-IID (Low E, E=1)'], label='Non-IID (Low E=1)', marker='x')
plt.plot(rounds, results_history['Non-IID Baseline (E=5, C=0.1)'], label='Non-IID Baseline (E=5)', marker='o', linestyle='--')
plt.plot(rounds, results_history['Non-IID (High E, E=10)'], label='Non-IID (High E=10)', marker='s')
plt.title('Experiment 3: Impact of Local Epochs (E) in Non-IID Setting', fontsize=16)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Global Model Accuracy (%)', fontsize=12)
plt.legend(fontsize=11)
plt.ylim(0, 100)
plt.savefig(os.path.join(output_dir, 'fig_3_impact_of_E.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'fig_3_impact_of_E.png')}")

plt.figure(figsize=(10, 6))
plt.plot(rounds, results_history['IID Baseline'], label='IID Baseline', marker='s', linestyle=':')
plt.plot(rounds, results_history['Non-IID Baseline (E=5, C=0.1)'], label='Non-IID Baseline', marker='o', linestyle='--')
plt.plot(rounds, results_history['Non-IID (Combined Mitigation, C=0.5, E=1)'], label='Non-IID (Combined Mitigation)', marker='^')
plt.title('Experiment 5: Combined Mitigation Strategy vs. Baselines', fontsize=16)
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Global Model Accuracy (%)', fontsize=12)
plt.legend(fontsize=11)
plt.ylim(0, 100)
plt.savefig(os.path.join(output_dir, 'fig_5_combined_mitigation.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'fig_5_combined_mitigation.png')}")

print("\nAnalysis complete. All plots and tables generated.")