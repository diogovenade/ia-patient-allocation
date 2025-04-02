import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from parse import parse_data
from pymoo.core.repair import Repair
from pymoo.visualization.scatter import Scatter
import matplotlib

class RepairOperator(Repair):
    def _do(self, problem, X, **kwargs):
        n_patients = problem.n_patients
        n_wards = problem.n_wards
        n_days = problem.n_days

        for i in range(X.shape[0]):
            ward_assignments = X[i, :n_patients].astype(int)
            day_assignments = X[i, n_patients:].astype(int)

            bed_usage = np.zeros((n_wards, n_days), dtype=int)

            for j in range(n_patients):
                ward = ward_assignments[j]
                start_day = day_assignments[j]
                patient = problem.data['patients'][j]

                # Check if the ward assignment is valid
                ward_data = problem.data['wards'][ward]
                if not (
                    patient['specialization'] == ward_data['major_specialization'] or
                    patient['specialization'] in ward_data['minor_specializations']
                ):
                    # Reassign to a valid ward
                    valid_wards = [
                        k for k, wd in enumerate(problem.data['wards'])
                        if patient['specialization'] == wd['major_specialization'] or
                        patient['specialization'] in wd['minor_specializations']
                    ]
                    if valid_wards:
                        ward_assignments[j] = np.random.choice(valid_wards)
                    else:
                        continue

                # Check if the day assignment is valid
                if not (patient['earliest_admission'] <= start_day <= patient['latest_admission']):
                    # Reassign to a valid day within the admission window
                    day_assignments[j] = np.random.randint(
                        patient['earliest_admission'], patient['latest_admission'] + 1
                    )

                # Update bed usage
                for day in range(start_day, start_day + patient['length_of_stay']):
                    if day < n_days:
                        bed_usage[ward_assignments[j], day] += 1

            # Repair bed capacity violations
            for ward in range(n_wards):
                for day in range(n_days):
                    while bed_usage[ward, day] > problem.data['wards'][ward]['bed_capacity']:
                        # Find a patient to reassign
                        for j in range(n_patients):
                            if ward_assignments[j] == ward and day_assignments[j] <= day < day_assignments[j] + problem.data['patients'][j]['length_of_stay']:
                                # Try to reassign the patient to another ward
                                valid_wards = [
                                    k for k, wd in enumerate(problem.data['wards'])
                                    if problem.data['patients'][j]['specialization'] == wd['major_specialization'] or
                                    problem.data['patients'][j]['specialization'] in wd['minor_specializations']
                                ]
                                valid_wards = [k for k in valid_wards if k != ward]  # Exclude the current ward
                                if valid_wards:
                                    new_ward = np.random.choice(valid_wards)
                                    ward_assignments[j] = new_ward
                                    for d in range(day_assignments[j], day_assignments[j] + problem.data['patients'][j]['length_of_stay']):
                                        if d < n_days:
                                            bed_usage[ward, d] -= 1
                                            bed_usage[new_ward, d] += 1
                                    break
                                else:
                                    # If no valid ward is found, adjust the admission day
                                    new_day = np.random.randint(problem.data['patients'][j]['earliest_admission'], problem.data['patients'][j]['latest_admission'] + 1)
                                    for d in range(day_assignments[j], day_assignments[j] + problem.data['patients'][j]['length_of_stay']):
                                        if d < n_days:
                                            bed_usage[ward, d] -= 1
                                    day_assignments[j] = new_day
                                    for d in range(new_day, new_day + problem.data['patients'][j]['length_of_stay']):
                                        if d < n_days:
                                            bed_usage[ward, d] += 1
                                    break

            # Update the solution
            X[i, :n_patients] = ward_assignments
            X[i, n_patients:] = day_assignments

        return X

class TuplePointCrossover(Crossover):
   def __init__(self, n_points=1, **kwargs):
      super().__init__(2, 2, **kwargs)
      self.n_points = n_points

   def _do(self, _, X, **kwargs):

      # Reshape the solution vector into tuples (w, d)
      n_var = X.shape[-1]
      n_tuples = n_var // 2

      # Adjust for the extra dimension (population size)
      n_parents, n_individuals, _ = X.shape
      X_tuples = X.reshape(n_parents, n_individuals, n_tuples, 2)

      # Get the X of parents and count the matings
      _, n_matings, _, _ = X_tuples.shape

      # Start point of crossover
      r = np.row_stack([np.random.permutation(n_tuples - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
      r.sort(axis=1)
      r = np.column_stack([r, np.full(n_matings, n_tuples)])

      # The mask to do the crossover
      M = np.full((n_matings, n_tuples), False)

      # Create for each individual the crossover range
      for i in range(n_matings):
         j = 0
         while j < r.shape[1] - 1:
            a, b = r[i, j], r[i, j + 1]
            M[i, a:b] = True
            j += 2

      Xp = np.empty_like(X_tuples)
      for i in range(n_matings):
         for j in range(2):  # Two parents
            Xp[j, i, M[i]] = X_tuples[1 - j, i, M[i]]
            Xp[j, i, ~M[i]] = X_tuples[j, i, ~M[i]]

      return Xp.reshape(X.shape)

class AdmissionDayMutation(Mutation):
   def _do(self, problem, X, **kwargs):
      for i in range(len(X)):
         patient_idx = np.random.randint(0, problem.n_patients)
         
         patient = problem.data['patients'][patient_idx]
         earliest = patient['earliest_admission']
         latest = patient['latest_admission']
         
         # Randomly select a new day within the time window
         new_day = np.random.randint(earliest, latest + 1)
         
         X[i, problem.n_patients + patient_idx] = new_day
      
      return X

class BalancedWorkload(ElementwiseProblem):
   def __init__(self, data):
      n_patients = len(data['patients'])
      n_wards = len(data['wards'])
      n_days = data['days']
      patients = data['patients']

      super().__init__(n_var=2 * n_patients,  # 2 variables per patient: ward and day
                        n_obj=2,  # 2 objectives: operational cost and max workload
                        n_eq_constr=2,  # 1 equality constraint
                        xl=np.zeros(2 * n_patients),  # Ward indices + Days
                        xu=np.concatenate((np.full(n_patients, n_wards - 1), 
                                    np.full(n_patients, n_days - 1))),  # Max bounds
                        vtype=int)
      self.data = data
      self.n_patients = n_patients
      self.n_wards = n_wards
      self.n_days = n_days
      self.patients = patients

      # Convert specialization IDs to indices
      self.spec_to_idx = {spec['id']: idx for idx, spec in enumerate(data['specializations'])}

   def _evaluate(self, x, out, *args, **kwargs):
      # Ward and day assignments from the solution array
      ward_assignments = x[:self.n_patients].astype(int)
      day_assignments = x[self.n_patients:].astype(int)

      # ---- Objective 1: Operational Cost ----
      ot_usage = np.zeros((len(self.data['specializations']), self.n_days))
      admission_delays = 0

      for i, patient in enumerate(self.data['patients']):
         ward = ward_assignments[i]
         day = day_assignments[i]

         # Delay
         delay = max(0, day - patient['earliest_admission'])
         admission_delays += delay * self.data['weights']['delay']

         # OT usage
         spec_idx = self.spec_to_idx[patient['specialization']]
         ot_usage[spec_idx, day] += patient['surgery_duration']
      
      # OT over and under utilization
      ot_over = np.maximum(0, ot_usage - np.array([spec['OT_availability'] for spec in self.data['specializations']]))
      ot_under = np.maximum(0, np.array([spec['OT_availability'] for spec in self.data['specializations']]) - ot_usage)
      
      ot_over = ot_over * self.data['weights']['overtime']
      ot_under = ot_under * self.data['weights']['undertime']

      ot_util = ot_over + ot_under
      
      operational_cost = ot_util.sum() + admission_delays

      # ---- Objective 2: Maximum Workload ----
      workload = np.zeros((self.n_wards, self.n_days))

      for i, patient in enumerate(self.data['patients']):
         ward = ward_assignments[i]
         start_day = day_assignments[i]

         for day in range(start_day, start_day + patient['length_of_stay']):
            if day < self.n_days:
               ward_data = self.data['wards'][ward]
               if patient['specialization'] in ward_data['minor_specializations']:
                  spec_idx = self.spec_to_idx[patient['specialization']]
                  scaling_factor = self.data['specializations'][spec_idx]['scaling_factor']
                  workload[ward, day] += patient['workload_per_day'][day - start_day] * scaling_factor
               else:
                  workload[ward, day] += patient['workload_per_day'][day - start_day]

      # Carryover workload
      for day in range(self.n_days):
         for ward in range(self.n_wards):
            ward_data = self.data['wards'][ward]
            workload[ward, day] += ward_data['carryover_workload'][day]

      # Normalize the workload
      normalized_workload = np.zeros((self.n_wards, self.n_days))

      for ward in range(self.n_wards):
         max_capacity = self.data['wards'][ward]['workload_capacity']  # βw in the formula
         if max_capacity > 0:
            normalized_workload[ward, :] = workload[ward, :] / max_capacity

      max_workload = np.max(normalized_workload)

      out["F"] = [operational_cost, max_workload]

      # ---- Constraints ----
      constraints = np.zeros(2)

      # Constraint 1: Feasibility of ward and day assignments
      check = 0
      for i, patient in enumerate(self.data['patients']):
         ward = ward_assignments[i]
         day = day_assignments[i]

         ward_data = self.data['wards'][ward]

         #print("Patient specialization: ", patient['specialization'])
         #print("Ward specialization: ", ward_data['major_specialization'])
         #print("Minor specializations: ", ward_data['minor_specializations'])

         is_feasible_ward = (
            patient['specialization'] == ward_data['major_specialization'] or
            patient['specialization'] in ward_data['minor_specializations']
         )

         #print ("Feasible Ward: ", is_feasible_ward)

         is_feasible_day = patient['earliest_admission'] <= day <= patient['latest_admission']         
         
         #print("Feasible Day: ", is_feasible_day)

         if not is_feasible_ward or not is_feasible_day:
            check = 1
            break
      
      constraints[0] = check

      # Bed capacity of wards not exceeded
      bed_capacity_violation = 0
      for ward in range(self.n_wards):
         for day in range(self.n_days):
            assigned_patients = sum(
               1 for i, patient in enumerate(self.data['patients'])
               if ward_assignments[i] == ward and day_assignments[i] <= day < day_assignments[i] + patient['length_of_stay']
            )
            
            total_patients = assigned_patients + self.data['wards'][ward]['carryover_patients'][day]

            #print("Total patients: ", total_patients)
            #print("Bed capacity: ", self.data['wards'][ward]['bed_capacity'])

            if total_patients > self.data['wards'][ward]['bed_capacity']:
               bed_capacity_violation = 1
               break
         if bed_capacity_violation:
            break

      constraints[1] = bed_capacity_violation

      #print("Ward violation: ", constraints[0])
      #print("Bed violation: ", constraints[1])

      out["H"] = constraints
      

data = parse_data("dataset/s0m0.dat")

matplotlib.use('TkAgg')
print("Active matplotlib backend:", matplotlib.get_backend())

algorithm = NSGA2(pop_size=100, 
                  sampling=IntegerRandomSampling(),
                  crossover=TuplePointCrossover(),
                  mutation=AdmissionDayMutation(),
                  eliminate_duplicates=True,
                  repair=RepairOperator())

result = minimize(BalancedWorkload(data),
                  algorithm,
                  termination=('n_gen', 200),
                  seed=1,
                  verbose=True)


# Print the solutions (ward and day assignments)
print("Solutions (ward and day assignments):")
for i, solution in enumerate(result.X):
    print(f"Solution {i + 1}:")
    n_patients = len(solution) // 2
    ward_assignments = solution[:n_patients].astype(int)
    day_assignments = solution[n_patients:].astype(int)
    for j in range(n_patients):
        print(f"  Patient {j + 1}: Ward {ward_assignments[j]}, Day {day_assignments[j]}")

plot = Scatter()
plot.add(result.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(result.F, facecolor="none", edgecolor="red")
plot.show()
