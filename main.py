import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

class BalancedWorkload(ElementwiseProblem):
   def __init__(self, data):
      n_patients = len(data['patients'])
      n_wards = len(data['wards'])
      n_days = data['days']
      patients = data['patients']

      super().__init__(n_var=2 * n_patients,  # 2 variables per patient: ward and day
                        n_obj=2,  # 2 objectives: operational cost and max workload
                        n_constr=10,  # 10 constraints
                        xl=np.zeros(2 * n_patients),  # Ward indices + Days
                        xu=np.concatenate((np.full(n_patients, n_wards - 1), 
                                    np.full(n_patients, n_days - 1))))  # Max bounds
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

      for day in range(self.n_days):
         for ward in range(self.n_wards):
            ward_data = self.data['wards'][ward]
            workload[ward, day] += ward_data['carryover_workload'][day]
      
      max_workload = np.max(workload)

      out["F"] = [operational_cost, max_workload]

data = ...

algorithm = NSGA2(pop_size=100, 
                  sampling=FloatRandomSampling(),
                  crossover=PointCrossover(0.9),
                  mutation=PolynomialMutation(),
                  eliminate_duplicates=True)

result = minimize(BalancedWorkload(data),
                  algorithm,
                  termination=('n_gen', 200),
                  seed=1,
                  verbose=True)
