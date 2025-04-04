import numpy as np
from mosa.mosa import Anneal

# Parse the data
from parse import parse_data
data = parse_data("dataset/s0m0.dat")

# Define the objective function
def fobj(wards, days) -> tuple:
    n_patients = len(data['patients'])
    spec_to_idx = {spec['id']: idx for idx, spec in enumerate(data['specializations'])}

    # ---- Objective 1: Operational Cost ----
    ot_usage = np.zeros((len(data['specializations']), data['days']))
    admission_delays = 0

    for i, patient in enumerate(data['patients']):
        ward = wards[i]
        day = days[i]

        # Delay
        delay = max(0, day - patient['earliest_admission'])
        admission_delays += delay * data['weights']['delay']

        # OT usage
        spec_idx = spec_to_idx[patient['specialization']]
        ot_usage[spec_idx, day] += patient['surgery_duration']

    # OT over and under utilization
    ot_over = np.maximum(0, ot_usage - np.array([spec['OT_availability'] for spec in data['specializations']]))
    ot_under = np.maximum(0, np.array([spec['OT_availability'] for spec in data['specializations']]) - ot_usage)

    ot_util = ot_over.sum() + ot_under.sum()

    bed_capacity_violations = 0
    for ward in range(len(data['wards'])):
        for day in range(data['days']):
            assigned_patients = sum(
                1 for i, patient in enumerate(data['patients'])
                if wards[i] == ward and days[i] <= day < days[i] + patient['length_of_stay']
            )
            total_patients = assigned_patients + data['wards'][ward]['carryover_patients'][day]
            if total_patients > data['wards'][ward]['bed_capacity']:
                bed_capacity_violations += total_patients - data['wards'][ward]['bed_capacity']

    operational_cost = ot_util + admission_delays + bed_capacity_violations

    # ---- Objective 2: Maximum Workload ----
    workload = np.zeros((len(data['wards']), data['days']))

    for i, patient in enumerate(data['patients']):
        ward = wards[i]
        start_day = days[i]

        for day in range(start_day, start_day + patient['length_of_stay']):
            if day < data['days']:
                workload[ward, day] += patient['workload_per_day'][day - start_day]

    max_workload = np.max(workload)

    return (operational_cost, max_workload)

# Initialize MOSA
opt = Anneal()

# Set the population
n_patients = len(data['patients'])
n_wards = len(data['wards'])
n_days = data['days']

opt.set_population(
    wards=list(range(n_wards)),
    days=list(range(n_days))
)

# Set group parameters
opt.set_group_params("wards", number_of_elements=n_patients, distinct_elements=False)
opt.set_group_params("days", number_of_elements=n_patients, distinct_elements=False)

# Configure MOSA parameters
opt.initial_temperature = 1.0
opt.temperature_decrease_factor = 0.9
opt.number_of_temperatures = 10
opt.number_of_iterations = 1000
opt.archive_size = 100

# Run the optimization
opt.evolve(fobj)

# Print the solutions
print("Solutions:")
for i, (solution, values) in enumerate(zip(opt.archive["Solution"], opt.archive["Values"])):
    print(f"Solution {i + 1}:")
    print(f"  Ward Assignments: {solution['wards']}")
    print(f"  Day Assignments: {solution['days']}")
    print(f"  Objectives: {values}")