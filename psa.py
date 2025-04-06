import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple
from parse import parse_data
import sys


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

        bed_capacity_violations = 0
        for ward in range(self.n_wards):
            for day in range(self.n_days):
                assigned_patients = sum(
                    1 for i, patient in enumerate(self.data['patients'])
                    if ward_assignments[i] == ward and day_assignments[i] <= day < day_assignments[i] + patient['length_of_stay']
                )
                
                total_patients = assigned_patients + self.data['wards'][ward]['carryover_patients'][day]

                if total_patients > self.data['wards'][ward]['bed_capacity']:
                    bed_capacity_violations += total_patients - self.data['wards'][ward]['bed_capacity']
        
        operational_cost = ot_util.sum() + admission_delays + bed_capacity_violations

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
            max_capacity = self.data['wards'][ward]['workload_capacity']
            if max_capacity > 0:
                normalized_workload[ward, :] = workload[ward, :] / max_capacity

        max_workload = np.max(normalized_workload)

        out["F"] = [operational_cost, max_workload]

        # Constraint: Feasibility of ward and day assignments
        unfeasible = 0
        for i, patient in enumerate(self.data['patients']):
            ward = ward_assignments[i]
            day = day_assignments[i]

            ward_data = self.data['wards'][ward]

            is_feasible_ward = (
                patient['specialization'] == ward_data['major_specialization'] or
                patient['specialization'] in ward_data['minor_specializations']
            )

            is_feasible_day = patient['earliest_admission'] <= day <= patient['latest_admission']         

            if not is_feasible_ward or not is_feasible_day:
                unfeasible = 1
                break

        out["H"] = unfeasible


class ParetoSimulatedAnnealing:
    def __init__(self, 
                 problem,
                 initial_solutions=None,
                 n_initial_solutions=10,
                 temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 max_iterations: int = 1000,
                 inner_iterations: int = 10):
        self.problem = problem
        self.temperature = temperature
        self.initial_temperature = temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.inner_iterations = inner_iterations
        
        # Generate initial solutions
        if initial_solutions is None:
            self.solutions = []
            for _ in range(n_initial_solutions):
                solution = np.concatenate((
                    np.random.randint(0, len(problem.data['wards']), problem.n_patients),
                    np.array([random.randint(
                        patient['earliest_admission'], 
                        patient['latest_admission']
                    ) for patient in problem.data['patients']])
                ))
                self.repair_solution(solution)
                self.solutions.append(solution)
        else:
            self.solutions = initial_solutions
            for solution in self.solutions:
                self.repair_solution(solution)
        
        # Initialize archive with non-dominated solutions from initial set
        self.archive = []
        self.archive_objectives = []
        
        # Initialize objective weights (equal weight initially)
        self.objective_weights = np.ones(2) / 2
        
        # Initialize archive with non-dominated solutions from initial population
        for solution in self.solutions:
            objectives, violation = self.evaluate_solution(solution)
            if violation == 0:
                self.archive.append(solution.copy())
                self.archive_objectives.append(objectives)
        
        self.progress_callback = None
    
    def dominates(self, sol1_objectives: List[float], sol2_objectives: List[float]) -> bool:
        """Check if solution 1 dominates solution 2 (assuming minimization)"""
        better_in_any = False
        for f1, f2 in zip(sol1_objectives, sol2_objectives):
            if f1 > f2:
                return False
            if f1 < f2:
                better_in_any = True
        return better_in_any
    
    def repair_solution(self, solution):
        """Repair a solution to ensure it respects ward and day feasibility constraints."""
        n_patients = self.problem.n_patients
        ward_assignments = solution[:n_patients]
        day_assignments = solution[n_patients:]

        for i, patient in enumerate(self.problem.data['patients']):
            # Repair ward assignment
            valid_wards = [
                ward_idx for ward_idx, ward in enumerate(self.problem.data['wards'])
                if patient['specialization'] == ward['major_specialization'] or
                   patient['specialization'] in ward['minor_specializations']
            ]
            if ward_assignments[i] not in valid_wards:
                ward_assignments[i] = random.choice(valid_wards)

            # Repair day assignment
            earliest = patient['earliest_admission']
            latest = patient['latest_admission']
            if not (earliest <= day_assignments[i] <= latest):
                day_assignments[i] = random.randint(earliest, latest)

        solution[:n_patients] = ward_assignments
        solution[n_patients:] = day_assignments
    
    def evaluate_solution(self, solution):
        """Evaluate solution using the problem's _evaluate method"""
        out = {"F": None, "H": None}
        self.problem._evaluate(solution, out)
        return out["F"], out["H"]
    
    def generate_neighbor(self, solution):
        """Generate a neighboring solution"""
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
    
    def update_archive(self, new_solution):
        """Update the archive with a new solution"""
        new_objectives, violation = self.evaluate_solution(new_solution)
        
        # Skip solutions that violate constraints
        if violation > 0:
            return False
            
        # Check for duplicates in objective space
        for objectives in self.archive_objectives:
            if np.allclose(objectives, new_objectives, rtol=1e-5, atol=1e-8):
                return False  # Skip if this objective vector already exists
                
        # Check if the new solution is dominated by any solution in the archive
        dominated = False
        solutions_to_remove = []
        
        for i, (solution, objectives) in enumerate(zip(self.archive, self.archive_objectives)):
            if self.dominates(objectives, new_objectives):
                dominated = True
                break
            elif self.dominates(new_objectives, objectives):
                solutions_to_remove.append(i)
                
        # If not dominated, add to archive and remove dominated solutions
        if not dominated:
            # Remove dominated solutions
            for i in sorted(solutions_to_remove, reverse=True):
                self.archive.pop(i)
                self.archive_objectives.pop(i)
                
            self.archive.append(new_solution.copy())
            self.archive_objectives.append(new_objectives)
            return True
            
        return False
    
    def get_closest_solution(self, solution, objectives):
        """Find closest solution in archive to current solution"""
        if not self.archive:
            return None, None
            
        curr_objectives, _ = self.evaluate_solution(solution)
        
        # Calculate Euclidean distances in normalized objective space
        min_dist = float('inf')
        closest_solution = None
        closest_objectives = None
        
        for archive_sol, archive_obj in zip(self.archive, self.archive_objectives):
            # Skip if it's the same solution
            if np.array_equal(solution, archive_sol):
                continue
                
            # Calculate distance in normalized objective space
            dist = np.linalg.norm(np.array(curr_objectives) - np.array(archive_obj))
            
            if dist < min_dist:
                min_dist = dist
                closest_solution = archive_sol
                closest_objectives = archive_obj
                
        return closest_solution, closest_objectives
    
    def update_objective_weights(self, current_objectives, neighbor_objectives):
        """Update weights based on partial dominance"""
        # Count objectives where each solution is better
        current_better = 0
        neighbor_better = 0
        
        for i, (curr, neigh) in enumerate(zip(current_objectives, neighbor_objectives)):
            if curr < neigh:  # Current is better (minimization)
                current_better += 1
            elif neigh < curr:  # Neighbor is better
                neighbor_better += 1
        
        # Update weights if there's partial dominance
        if current_better > 0 and neighbor_better > 0:
            # Increase weights for objectives where current solution is worse
            for i, (curr, neigh) in enumerate(zip(current_objectives, neighbor_objectives)):
                if neigh < curr:  # Current is worse in this objective
                    self.objective_weights[i] *= 1.05  # Increase weight
                    
            # Normalize weights
            self.objective_weights = self.objective_weights / np.sum(self.objective_weights)
    
    def acceptance_probability(self, current_objectives, neighbor_objectives):
        """Calculate acceptance probability for dominated solutions"""
        # Use weighted sum of objectives
        weighted_current = np.sum(np.array(current_objectives) * self.objective_weights)
        weighted_neighbor = np.sum(np.array(neighbor_objectives) * self.objective_weights)
        
        # Calculate energy difference (for minimization)
        energy_diff = weighted_neighbor - weighted_current
        
        # Always accept improvements
        if energy_diff < 0:
            return 1.0
            
        # Calculate acceptance probability
        if self.temperature < 1e-10:
            return 0.0
            
        return np.exp(-energy_diff / self.temperature)
    
    def check_cooling_condition(self, outer_iter):
        """Check if cooling should stop"""
        # Stop if temperature is very low or max iterations reached
        return self.temperature < 1e-6 or outer_iter >= self.max_iterations
    
    def optimize(self):
        """Run the Pareto Simulated Annealing algorithm following the pseudocode"""
        print("Starting PSA optimization...")
        
        outer_iter = 0
        
        # Outer loop
        while not self.check_cooling_condition(outer_iter):
            outer_iter += 1
            
            if outer_iter % 5 == 0 or outer_iter == 1:
                print(f"Outer iteration {outer_iter}/{self.max_iterations}, Temperature: {self.temperature:.2f}")
                print(f"Current archive size: {len(self.archive)}")
                
                # Call progress callback if defined
                if self.progress_callback and not self.progress_callback(outer_iter, self.temperature, len(self.archive)):
                    print("Optimization stopped by user")
                    break
            
            # For each solution in current population
            for solution_idx, current_solution in enumerate(self.solutions):
                current_objectives, current_violation = self.evaluate_solution(current_solution)
                
                # Skip invalid solutions
                if current_violation > 0:
                    continue
                    
                
                # Inner loop
                for inner_iter in range(self.inner_iterations):
                    # Generate neighbor S_new
                    neighbor = self.generate_neighbor(current_solution)
                    self.repair_solution(neighbor)
                    
                    neighbor_objectives, neighbor_violation = self.evaluate_solution(neighbor)
                    
                    # Skip invalid neighbors
                    if neighbor_violation > 0:
                        continue
                        
                    # Check dominance
                    if not self.dominates(current_objectives, neighbor_objectives):
                        # Update archive with S_new
                        archive_updated = self.update_archive(neighbor)
                        
                        # Find closest solution
                        closest_solution, closest_objectives = self.get_closest_solution(current_solution, current_objectives)
                        
                        # Update weights
                        if closest_objectives is not None:
                            self.update_objective_weights(current_objectives, neighbor_objectives)
                            
                        # Accept the new solution
                        current_solution = neighbor.copy()
                        current_objectives = neighbor_objectives.copy()
                        self.solutions[solution_idx] = current_solution
                    else:
                        # Dominated case
                        probability = self.acceptance_probability(current_objectives, neighbor_objectives)
                        if random.random() < probability:
                            current_solution = neighbor.copy()
                            current_objectives = neighbor_objectives.copy()
                            self.solutions[solution_idx] = current_solution
            
            # Decrease temperature
            self.temperature *= self.cooling_rate
        
        # Final progress callback
        if self.progress_callback:
            self.progress_callback(outer_iter, self.temperature, len(self.archive))
        
        print("Optimization complete!")
        print(f"Final archive size: {len(self.archive)}")
        return self.archive, self.archive_objectives
