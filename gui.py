import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import threading
import time
import os
from psa import ParetoSimulatedAnnealing, PatientSchedulingProblem
from nsga import BalancedWorkload, NSGA2, AdmissionDayMutation, RepairOperator, minimize
from parse import parse_data

class MetaheuristicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Metaheuristics for Patient Scheduling")
        self.root.geometry("1200x800")
        
        # Problem data
        self.problem_data = None
        self.current_problem = None
        self.results = {}
        self.running = False
        self.comparison_mode = False
        self.selected_files = []
        
        # Setup UI
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Problem selection
        ttk.Label(control_frame, text="Problem File:").pack(anchor=tk.W)
        self.problem_file = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.problem_file, width=30).pack(fill=tk.X)
        ttk.Button(control_frame, text="Browse...", command=self.browse_problem).pack(pady=5)
        
        # Comparison mode checkbox
        self.comparison_var = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Comparison Mode", variable=self.comparison_var,
                       command=self.toggle_comparison_mode).pack(pady=5)
        
        # Comparison file selection (hidden by default)
        self.comparison_frame = ttk.Frame(control_frame)
        ttk.Label(self.comparison_frame, text="Select Files for Comparison:").pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(self.comparison_frame, selectmode=tk.MULTIPLE, height=5)
        self.file_listbox.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.comparison_frame, text="Load Files", command=self.load_instances).pack(pady=5)
        
        # Algorithm selection
        ttk.Label(control_frame, text="Algorithm:").pack(anchor=tk.W, pady=(10,0))
        self.algorithm = tk.StringVar(value="PSA")
        algo_frame = ttk.Frame(control_frame)
        algo_frame.pack(fill=tk.X)
        ttk.Radiobutton(algo_frame, text="Pareto SA", variable=self.algorithm, value="PSA").pack(side=tk.LEFT)
        ttk.Radiobutton(algo_frame, text="NSGA-II", variable=self.algorithm, value="NSGA2").pack(side=tk.LEFT)
        
        # Parameters frame
        param_frame = ttk.LabelFrame(control_frame, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        # Common parameters
        ttk.Label(param_frame, text="Iterations:").grid(row=0, column=0, sticky=tk.W)
        self.iterations = tk.IntVar(value=1000)
        ttk.Entry(param_frame, textvariable=self.iterations, width=10).grid(row=0, column=1, sticky=tk.E)
        
        # PSA specific parameters
        self.psa_frame = ttk.Frame(param_frame)
        ttk.Label(self.psa_frame, text="Initial Temp:").grid(row=0, column=0, sticky=tk.W)
        self.initial_temp = tk.DoubleVar(value=100.0)
        ttk.Entry(self.psa_frame, textvariable=self.initial_temp, width=10).grid(row=0, column=1, sticky=tk.E)
        
        ttk.Label(self.psa_frame, text="Cooling Rate:").grid(row=1, column=0, sticky=tk.W)
        self.cooling_rate = tk.DoubleVar(value=0.95)
        ttk.Entry(self.psa_frame, textvariable=self.cooling_rate, width=10).grid(row=1, column=1, sticky=tk.E)
        
        # NSGA-II specific parameters
        self.nsga_frame = ttk.Frame(param_frame)
        ttk.Label(self.nsga_frame, text="Population:").grid(row=0, column=0, sticky=tk.W)
        self.population = tk.IntVar(value=100)
        ttk.Entry(self.nsga_frame, textvariable=self.population, width=10).grid(row=0, column=1, sticky=tk.E)
        
        ttk.Label(self.nsga_frame, text="Mutation Prob:").grid(row=1, column=0, sticky=tk.W)
        self.mutation_prob = tk.DoubleVar(value=0.1)
        ttk.Entry(self.nsga_frame, textvariable=self.mutation_prob, width=10).grid(row=1, column=1, sticky=tk.E)
        
        # Show appropriate parameter frame based on algorithm
        self.algorithm.trace_add('write', self.update_parameter_visibility)
        self.update_parameter_visibility()
        
        # Run button
        ttk.Button(control_frame, text="Run Optimization", command=self.run_optimization).pack(pady=10)
        
        # Compare button (only visible in comparison mode)
        self.compare_button = ttk.Button(control_frame, text="Compare Algorithms", 
                                      command=self.compare_algorithms, state=tk.DISABLED)
        
        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status).pack(side=tk.BOTTOM, fill=tk.X)
        
        # Right panel - visualization
        vis_frame = ttk.Frame(main_frame)
        vis_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook for multiple tabs
        self.notebook = ttk.Notebook(vis_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Pareto front tab
        self.pareto_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.pareto_tab, text="Pareto Front")
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.pareto_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.pareto_tab)
        self.toolbar.update()
        
        # Solution details tab
        self.solution_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.solution_tab, text="Solution Details")
        
        # Text widget for solution details
        self.solution_text = tk.Text(self.solution_tab, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(self.solution_tab, command=self.solution_text.yview)
        self.solution_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.solution_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress tab
        self.progress_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.progress_tab, text="Progress")
        
        # Progress text widget
        self.progress_text = tk.Text(self.progress_tab, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(self.progress_tab, command=self.progress_text.yview)
        self.progress_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.progress_text.pack(fill=tk.BOTH, expand=True)
        
        # Results comparison tab
        self.comparison_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_tab, text="Algorithm Comparison")
        
        # Treeview for results comparison
        columns = ("File", "Algorithm", "Operational Cost", "Max Workload", "Time (s)", "Iterations")
        self.comparison_tree = ttk.Treeview(self.comparison_tab, columns=columns, show="headings")
        for col in columns:
            self.comparison_tree.heading(col, text=col)
            self.comparison_tree.column(col, width=120)
        scrollbar = ttk.Scrollbar(self.comparison_tab, command=self.comparison_tree.yview)
        self.comparison_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comparison_tree.pack(fill=tk.BOTH, expand=True)
        
    def toggle_comparison_mode(self):
        self.comparison_mode = self.comparison_var.get()
        if self.comparison_mode:
            self.comparison_frame.pack(fill=tk.X, pady=5)
            self.compare_button.pack(pady=10)
        else:
            self.comparison_frame.pack_forget()
            self.compare_button.pack_forget()
    
    def load_instances(self):
        instances_dir = os.path.join(os.getcwd(), "data", "instances")
        if not os.path.exists(instances_dir):
            messagebox.showerror("Error", f"Directory not found: {instances_dir}")
            return
            
        dat_files = [f for f in os.listdir(instances_dir) if f.endswith('.dat')]
        if not dat_files:
            messagebox.showinfo("Info", "No .dat files found in the instances directory")
            return
            
        self.file_listbox.delete(0, tk.END)
        for file in sorted(dat_files):
            self.file_listbox.insert(tk.END, file)
        
        self.compare_button['state'] = tk.NORMAL
    
    def update_parameter_visibility(self, *args):
        # Hide all parameter frames
        self.psa_frame.grid_forget()
        self.nsga_frame.grid_forget()
        
        # Show appropriate parameter frame
        if self.algorithm.get() == "PSA":
            self.psa_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5,0))
        else:
            self.nsga_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5,0))
    
    def browse_problem(self):
        filename = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "data", "instances"),
            title="Select Problem File",
            filetypes=(("DAT files", "*.dat"), ("All files", "*.*")))
        if filename:
            self.problem_file.set(filename)
    
    def run_optimization(self):
        if self.running:
            messagebox.showwarning("Warning", "An optimization is already running!")
            return
            
        if not self.problem_file.get():
            messagebox.showerror("Error", "Please select a problem file first!")
            return
            
        try:
            # Parse the problem data
            self.problem_data = parse_data(self.problem_file.get())
            self.current_problem = PatientSchedulingProblem(self.problem_data)
            
            # Clear previous results
            self.ax.clear()
            self.solution_text.delete(1.0, tk.END)
            self.progress_text.delete(1.0, tk.END)
            
            # Update status
            self.status.set("Running optimization...")
            self.running = True
            
            # Run in a separate thread to keep GUI responsive
            thread = threading.Thread(target=self.execute_algorithm, daemon=True)
            thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load problem file:\n{str(e)}")
            self.status.set("Error loading problem")
    
    def compare_algorithms(self):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "Please select at least one file to compare!")
            return
            
        self.selected_files = [self.file_listbox.get(i) for i in selected_indices]
        
        # Clear previous results
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        
        # Run comparison for each selected file
        for file in self.selected_files:
            file_path = os.path.join(os.getcwd(), "data", "instances", file)
            self.problem_file.set(file_path)
            
            # Run both algorithms
            for algorithm in ["PSA", "NSGA2"]:
                self.algorithm.set(algorithm)
                try:
                    self.run_optimization()
                    
                    # Wait until optimization completes
                    while self.running:
                        time.sleep(0.1)
                        self.root.update()
                    
                    # Add results to comparison tree
                    if algorithm in self.results and len(self.results[algorithm]['objectives']) > 0:
                        best_idx = np.argmin([obj[0] for obj in self.results[algorithm]['objectives']])
                        objectives = self.results[algorithm]['objectives'][best_idx]
                        
                        self.comparison_tree.insert("", tk.END, values=(
                            os.path.basename(file),
                            algorithm,
                            f"{objectives[0]:.2f}",
                            f"{objectives[1]:.2f}",
                            f"{self.results[algorithm]['time']:.2f}",
                            self.results[algorithm]['iterations']
                        ))
                    
                except Exception as e:
                    self.comparison_tree.insert("", tk.END, values=(
                        os.path.basename(file),
                        algorithm,
                        "ERROR",
                        "ERROR",
                        "ERROR",
                        "ERROR"
                    ))
                    messagebox.showerror("Error", f"Failed to process {file} with {algorithm}:\n{str(e)}")
        
        # Switch to comparison tab
        self.notebook.select(self.comparison_tab)
        self.status.set("Comparison complete")
    
    def execute_algorithm(self):
        algorithm = self.algorithm.get()
        iterations = self.iterations.get()
        
        start_time = time.time()
        
        if algorithm == "PSA":
            # Run Pareto Simulated Annealing
            psa = ParetoSimulatedAnnealing(
                problem=self.current_problem,
                temperature=self.initial_temp.get(),
                cooling_rate=self.cooling_rate.get(),
                n_iterations=iterations
            )
            
            # Add a callback to update progress
            def psa_callback(iteration, temp, front_size):
                self.update_progress(f"Iteration {iteration}/{iterations}, Temp: {temp:.2f}, Front size: {front_size}")
                self.update_pareto_front(psa.pareto_front_objectives)
                return not self.running  # Return True to stop if self.running is False
                
            psa.optimize()
            solutions = psa.pareto_front
            objectives = psa.pareto_front_objectives
            
        else:  # NSGA-II
            # Run NSGA-II
            algorithm = NSGA2(
                pop_size=self.population.get(),
                n_offsprings=self.population.get(),
                mutation=AdmissionDayMutation(prob=self.mutation_prob.get()),
                eliminate_duplicates=True,
                repair=RepairOperator()
            )
            
            result = minimize(
                BalancedWorkload(self.problem_data),
                algorithm,
                termination=('n_gen', iterations // self.population.get()),
                verbose=True,
                callback=self.nsga_callback
            )
            
            solutions = result.X
            objectives = result.F
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Store results
        self.results[algorithm] = {
            "solutions": solutions,
            "objectives": objectives,
            "time": execution_time,
            "iterations": iterations
        }
        
        # Update UI with final results
        self.update_comparison()
        self.update_pareto_front(objectives)
        self.display_solution_details(solutions[0], objectives[0])
        
        self.status.set(f"Optimization complete in {execution_time:.2f} seconds")
        self.running = False
    
    def nsga_callback(self, algorithm):
        # Callback for NSGA-II to update progress
        gen = algorithm.n_gen
        pop_size = algorithm.pop_size
        front_size = len(algorithm.opt)
        
        self.update_progress(f"Generation {gen}, Population: {pop_size}, Front size: {front_size}")
        
        # Update Pareto front visualization
        if hasattr(algorithm, 'opt'):
            self.update_pareto_front(algorithm.opt.get("F"))
        
        return not self.running  # Return True to stop if self.running is False
    
    def update_progress(self, message):
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update()
    
    def update_pareto_front(self, objectives):
        if not objectives:
            return
            
        self.ax.clear()
        
        # Convert objectives to numpy array if not already
        objectives = np.array(objectives)
        
        # Plot Pareto front
        self.ax.scatter(objectives[:, 0], objectives[:, 1], c='red', s=50, edgecolors='none', label='Solutions')
        
        # Add labels and title
        self.ax.set_xlabel('Operational Cost')
        self.ax.set_ylabel('Maximum Workload')
        self.ax.set_title('Pareto Front')
        self.ax.grid(True)
        self.ax.legend()
        
        # Redraw canvas
        self.canvas.draw()
    
    def display_solution_details(self, solution, objectives):
        self.solution_text.delete(1.0, tk.END)
        
        # Display objectives
        self.solution_text.insert(tk.END, f"Operational Cost: {objectives[0]:.2f}\n")
        self.solution_text.insert(tk.END, f"Maximum Workload: {objectives[1]:.2f}\n\n")
        
        # Display ward assignments
        n_patients = len(solution) // 2
        ward_assignments = solution[:n_patients].astype(int)
        day_assignments = solution[n_patients:].astype(int)
        
        self.solution_text.insert(tk.END, "Patient Assignments:\n")
        for i in range(min(20, n_patients)):  # Show first 20 patients
            patient = self.problem_data['patients'][i]
            ward = self.problem_data['wards'][ward_assignments[i]]
            self.solution_text.insert(tk.END, 
                f"Patient {i+1}: Ward {ward['id']} ({ward['major_specialization']}), "
                f"Day {day_assignments[i]} (LOS: {patient['length_of_stay']})\n")
        
        # Display bed capacity violations
        violations = self.calculate_violations(solution)
        if violations:
            self.solution_text.insert(tk.END, "\nBed Capacity Violations:\n")
            for ward_idx, day, excess in violations:
                ward = self.problem_data['wards'][ward_idx]
                self.solution_text.insert(tk.END, 
                    f"Ward {ward['id']}, Day {day}: {excess} patients over capacity\n")
    
    def calculate_violations(self, solution):
        violations = []
        n_patients = len(solution) // 2
        ward_assignments = solution[:n_patients].astype(int)
        day_assignments = solution[n_patients:].astype(int)
        
        for ward in range(self.current_problem.n_wards):
            for day in range(self.current_problem.n_days):
                assigned_patients = sum(
                    1 for i, patient in enumerate(self.problem_data['patients'])
                    if ward_assignments[i] == ward and day_assignments[i] <= day < day_assignments[i] + patient['length_of_stay']
                )
                
                total_patients = assigned_patients + self.problem_data['wards'][ward]['carryover_patients'][day]
                capacity = self.problem_data['wards'][ward]['bed_capacity']
                
                if total_patients > capacity:
                    violations.append((ward, day, total_patients - capacity))
        
        return violations
    
    def update_comparison(self):
        # Clear existing data
        for item in self.comparison_tree.get_children():
            self.comparison_tree.delete(item)
        
        # Add new results
        for algo, result in self.results.items():
            if len(result['objectives']) > 0:
                # Get the best solution (minimum operational cost)
                best_idx = np.argmin([obj[0] for obj in result['objectives']])
                objectives = result['objectives'][best_idx]
                
                self.comparison_tree.insert("", tk.END, values=(
                    os.path.basename(self.problem_file.get()),
                    algo,
                    f"{objectives[0]:.2f}",
                    f"{objectives[1]:.2f}",
                    f"{result['time']:.2f}",
                    result['iterations']
                ))
    
def main():
    root = tk.Tk()
    app = MetaheuristicsApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()