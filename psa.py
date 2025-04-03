import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple
from parse import parse_data


class PatientSchedulingProblem:
    """Problem class for the patient scheduling problem (similar to BalancedWorkload but for PSA)"""
    def __init__(self, data):
        self.data = data
        self.n_patients = len(data['patients'])
        self.n_wards = len(data['wards'])
        self.n_days = data['days']
        
        # Convert specialization IDs to indices
        self.spec_to_idx = {spec['id']: idx for idx, spec in enumerate(data['specializations'])}
    
    def _evaluate(self, x, out):
        """Evaluate a solution, calculating objectives and constraint violations"""
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
        # Bed capacity of wards not exceeded
        bed_capacity_violation = 0
        for ward in range(self.n_wards):
            for day in range(self.n_days):
                assigned_patients = sum(
                    1 for i, patient in enumerate(self.data['patients'])
                    if ward_assignments[i] == ward and day_assignments[i] <= day < day_assignments[i] + patient['length_of_stay']
                )
                
                total_patients = assigned_patients + self.data['wards'][ward]['carryover_patients'][day]

                if total_patients > self.data['wards'][ward]['bed_capacity']:
                    bed_capacity_violation = 1
                    break
            if bed_capacity_violation:
                break

        out["H"] = bed_capacity_violation
        return out


class ParetoSimulatedAnnealing:
    def __init__(self, 
                 problem,
                 initial_solution=None,
                 temperature: float = 1000.0,
                 cooling_rate: float = 0.95,
                 n_iterations: int = 1000,
                 step_size: float = 0.1):
        self.problem = problem
        # Create initial solution if not provided
        if initial_solution is None:
            self.current_solution = np.concatenate((
                np.random.randint(0, len(problem.data['wards']), problem.n_patients),
                np.array([random.randint(
                    patient['earliest_admission'], 
                    patient['latest_admission']
                ) for patient in problem.data['patients']])
            ))
        else:
            self.current_solution = initial_solution
            
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.n_iterations = n_iterations
        self.step_size = step_size
        self.pareto_front = []
        self.pareto_front_objectives = []
        self.update_pareto_front(self.current_solution)
        
    def dominates(self, sol1_objectives: List[float], sol2_objectives: List[float]) -> bool:
        """Check if solution 1 dominates solution 2 (assuming minimization)"""
        better_in_any = False
        for f1, f2 in zip(sol1_objectives, sol2_objectives):
            if f1 > f2:  # For minimization
                return False
            if f1 < f2:
                better_in_any = True
        return better_in_any
    
    def evaluate_solution(self, solution):
        """Evaluate solution using the problem's _evaluate method"""
        out = {"F": None, "H": None}
        self.problem._evaluate(solution, out)
        return out["F"], out["H"]
    
    def generate_neighbor(self, solution):
        """Generate a more diverse neighboring solution"""
        neighbor = solution.copy()
        n_patients = self.problem.n_patients
        
        # Sometimes create a completely new solution (5% chance)
        if random.random() < 0.05:
            return np.concatenate((
                np.random.randint(0, len(self.problem.data['wards']), n_patients),
                np.array([random.randint(
                    patient['earliest_admission'], 
                    patient['latest_admission']
                ) for patient in self.problem.data['patients']])
            ))
        
        # Make larger changes 15% of the time
        make_large_change = random.random() < 0.15
        
        # Number of patients to modify
        num_to_change = random.randint(3, 8) if make_large_change else random.randint(1, 2)
        
        # Choose random patients to change
        patient_indices = random.sample(range(n_patients), num_to_change)
        
        for patient_idx in patient_indices:
            # Modify both ward and day with 30% probability
            if random.random() < 0.3:
                # Modify both
                self._modify_ward(neighbor, patient_idx)
                self._modify_day(neighbor, patient_idx)
            # Otherwise modify just one
            elif random.random() < 0.5:
                self._modify_ward(neighbor, patient_idx)
            else:
                self._modify_day(neighbor, patient_idx)
        
        return neighbor

    def _modify_ward(self, solution, patient_idx):
        """Helper to modify ward assignment"""
        patient = self.problem.data['patients'][patient_idx]
        valid_wards = []
        
        for ward_idx, ward in enumerate(self.problem.data['wards']):
            if (patient['specialization'] == ward['major_specialization'] or 
                patient['specialization'] in ward['minor_specializations']):
                valid_wards.append(ward_idx)
        
        if valid_wards:
            # Ensure we choose a different ward when possible
            current_ward = solution[patient_idx]
            if len(valid_wards) > 1 and current_ward in valid_wards:
                valid_wards.remove(current_ward)
            solution[patient_idx] = random.choice(valid_wards)

    def _modify_day(self, solution, patient_idx):
        """Helper to modify day assignment"""
        n_patients = self.problem.n_patients
        patient = self.problem.data['patients'][patient_idx]
        earliest = patient['earliest_admission']
        latest = patient['latest_admission']
        
        current_day = solution[n_patients + patient_idx]
        
        # Create a day range that excludes the current day
        possible_days = list(range(earliest, latest + 1))
        if len(possible_days) > 1 and current_day in possible_days:
            possible_days.remove(current_day)
        
        if possible_days:
            solution[n_patients + patient_idx] = random.choice(possible_days)
    
    def update_pareto_front(self, new_solution):
        """Update the Pareto front with a new solution"""
        new_objectives, violation = self.evaluate_solution(new_solution)
        
        # Skip solutions that violate constraints
        if violation > 0:
            return
            
        # Check for duplicates in objective space (IMPORTANT NEW CHECK)
        for objectives in self.pareto_front_objectives:
            if np.allclose(objectives, new_objectives, rtol=1e-5, atol=1e-8):
                return  # Skip if this objective vector already exists
                
        # Check if the new solution is dominated by any solution in the Pareto front
        dominated = False
        solutions_to_remove = []
        
        for i, (solution, objectives) in enumerate(zip(self.pareto_front, self.pareto_front_objectives)):
            if self.dominates(objectives, new_objectives):
                dominated = True
                break
            elif self.dominates(new_objectives, objectives):
                solutions_to_remove.append(i)
                
        # If not dominated, add to Pareto front and remove dominated solutions
        if not dominated:
            # Remove dominated solutions (in reverse order to avoid index issues)
            for i in sorted(solutions_to_remove, reverse=True):
                self.pareto_front.pop(i)
                self.pareto_front_objectives.pop(i)
                
            self.pareto_front.append(new_solution.copy())
            self.pareto_front_objectives.append(new_objectives)
    
    def acceptance_probability(self, current_objectives, neighbor_objectives):
        """Modified acceptance probability for multiple objectives"""
        # Equal weight to each objective
        norm_current = [o/100 for o in current_objectives]  # Normalize to similar scale
        norm_neighbor = [o/100 for o in neighbor_objectives]
        
        # Calculate energy difference using Euclidean distance
        curr_vec = np.array(norm_current)
        neig_vec = np.array(norm_neighbor)
        
        # For minimization, negative energy diff means improvement
        energy_diff = np.linalg.norm(neig_vec) - np.linalg.norm(curr_vec)
        
        # Always accept improvements
        if energy_diff < 0:
            return 1.0
            
        # Calculate acceptance probability with appropriate scaling
        if self.temperature < 1e-10:
            return 0.0
            
        # Use scaled energy difference
        try:
            return min(1.0, np.exp(-energy_diff / self.temperature))
        except OverflowError:
            return 0.0
    
    def optimize(self):
        """Run the Pareto Simulated Annealing algorithm"""
        print("Starting PSA optimization...")
        
        for i in range(self.n_iterations):
            if i % 100 == 0:
                print(f"Iteration {i}/{self.n_iterations}, Temperature: {self.temperature:.2f}")
                print(f"Current Pareto front size: {len(self.pareto_front)}")
            
            # Generate neighbor
            neighbor = self.generate_neighbor(self.current_solution)
            
            # Evaluate current and neighbor
            current_objectives, current_violation = self.evaluate_solution(self.current_solution)
            neighbor_objectives, neighbor_violation = self.evaluate_solution(neighbor)
            
            # Skip invalid neighbors
            if neighbor_violation > 0:
                continue
                
            # Accept new solution based on dominance or probability
            accept = False
            if self.dominates(neighbor_objectives, current_objectives):
                accept = True
            elif not self.dominates(current_objectives, neighbor_objectives):
                # Neither dominates - use acceptance probability
                probability = self.acceptance_probability(current_objectives, neighbor_objectives)
                if random.random() < probability:
                    accept = True
            
            if accept:
                self.current_solution = neighbor.copy()
                self.update_pareto_front(neighbor)
            
            # Cool down temperature
            self.temperature *= self.cooling_rate
        
        print("Optimization complete!")
        print(f"Final Pareto front size: {len(self.pareto_front)}")
        return self.pareto_front, self.pareto_front_objectives

    def plot_pareto_front(self):
        """Plot the Pareto front"""
        if not self.pareto_front_objectives:
            print("No solutions in Pareto front. Run optimize() first.")
            return
            
        objectives = np.array(self.pareto_front_objectives)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(objectives[:, 0], objectives[:, 1], c='red', s=50, edgecolors='none', label='PSA Solutions')
        plt.xlabel('Operational Cost')
        plt.ylabel('Maximum Workload')
        plt.title('Pareto Front - Patient Scheduling Problem')
        plt.grid(True)
        plt.legend()
        plt.show()


def run():
    """Run the PSA algorithm on the patient scheduling problem"""
    # Set matplotlib backend
    matplotlib.use('TkAgg')
    print("Active matplotlib backend:", matplotlib.get_backend())
    
    # Parse data and create problem instance
    data = parse_data("dataset/s13m0.dat")
    problem = PatientSchedulingProblem(data)
    
    print("Problem created successfully.")
    print(f"Days in planning period: {data['days']}")
    print(f"Number of wards: {len(data['wards'])}")
    print(f"Number of patients: {len(data['patients'])}")
    
    # Initialize the PSA algorithm
    psa = ParetoSimulatedAnnealing(
        problem=problem,
        temperature=100.0,         # Lower initial temperature 
        cooling_rate=0.995,        # Much slower cooling rate
        n_iterations=10000         # More iterations
    )
    
    # Run optimization
    pareto_front, objectives = psa.optimize()
    
    # Print results
    print("\nResults:")
    print(f"Found {len(pareto_front)} Pareto-optimal solutions")
    
    # Group solutions by objective values
    obj_groups = {}
    for i, obj in enumerate(objectives):
        key = tuple(np.round(obj, 2))  # Round to 2 decimal places for grouping
        if key not in obj_groups:
            obj_groups[key] = []
        obj_groups[key].append(i)

    print(f"Found {len(obj_groups)} distinct objective value groups")

    # Print unique objective values
    unique_objs = sorted(obj_groups.keys())
    print("\nDistinct objective values:")
    for i, obj in enumerate(unique_objs[:10]):  # Show top 10
        print(f"Group {i+1}: Cost={obj[0]}, Workload={obj[1]} ({len(obj_groups[obj])} solutions)")

    # Print a few representative solutions
    max_to_print = min(5, len(pareto_front))
    for i in range(max_to_print):
        print(f"\nSolution {i+1}:")
        print(f"Objectives: Operational Cost = {objectives[i][0]:.2f}, Max Workload = {objectives[i][1]:.2f}")
        
        # Extract ward and day assignments
        n_patients = problem.n_patients
        ward_assignments = pareto_front[i][:n_patients].astype(int)
        day_assignments = pareto_front[i][n_patients:].astype(int)
        
        print("Sample assignments (first 5 patients):")
        for j in range(min(5, n_patients)):
            print(f"  Patient {j+1}: Ward {ward_assignments[j]}, Day {day_assignments[j]}")
    
    # Visualize the Pareto front
    psa.plot_pareto_front()
    
    return pareto_front, objectives

if __name__ == "__main__":
    result, objectives = run()