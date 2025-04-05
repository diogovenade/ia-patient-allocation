import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import os
import threading
import time
import json
from datetime import datetime
from parse import parse_data
from nsga import BalancedWorkload, NSGA2, IntegerRandomSampling, TuplePointCrossover, AdmissionDayMutation, RepairOperator, DefaultMultiObjectiveTermination
from psa import ParetoSimulatedAnnealing, PatientSchedulingProblem
from pymoo.optimize import minimize

class OptimizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NSGA-II Patient Scheduling Optimization")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Variables
        self.dataset_path = tk.StringVar(value="dataset/s0m0.dat")
        self.population_size = tk.IntVar(value=100)
        self.max_generations = tk.IntVar(value=100)
        self.mutation_prob = tk.DoubleVar(value=0.1)
        self.initial_temp = tk.DoubleVar(value=100.0)
        self.cooling_rate = tk.DoubleVar(value=0.995)
        self.iterations = tk.IntVar(value=3000)
        self.running = False
        self.thread = None
        self.datasets = self.find_datasets()
        self.solution_history = []
        
        # Create GUI
        self.create_gui()
        
    def find_datasets(self):
        """Find all available datasets in the dataset folder"""
        datasets = []
        dataset_dir = "dataset"
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                if file.endswith(".dat"):
                    datasets.append(os.path.join(dataset_dir, file))
        return sorted(datasets)
    
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Split into left panel (controls) and right panel (visualization)
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Algorithm Selection
        algo_select_frame = ttk.LabelFrame(left_panel, text="Algorithm Selection")
        algo_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Algorithm selection with radio buttons
        self.algorithm = tk.StringVar(value="NSGA2")
        ttk.Radiobutton(algo_select_frame, text="NSGA-II", variable=self.algorithm, 
                        value="NSGA2", command=self.update_algorithm_controls).pack(side=tk.LEFT, padx=20, pady=5)
        ttk.Radiobutton(algo_select_frame, text="Pareto Simulated Annealing", variable=self.algorithm, 
                        value="PSA", command=self.update_algorithm_controls).pack(side=tk.LEFT, padx=20, pady=5)
        
        # Left panel - Controls
        self.control_frame = ttk.LabelFrame(left_panel, text="Algorithm Controls")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Dataset selection
        ttk.Label(self.control_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        dataset_combo = ttk.Combobox(self.control_frame, textvariable=self.dataset_path, values=self.datasets)
        dataset_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create frames for each algorithm's specific controls
        self.nsga_params_frame = ttk.Frame(self.control_frame)
        self.psa_params_frame = ttk.Frame(self.control_frame)
        
        # NSGA-II specific parameters
        ttk.Label(self.nsga_params_frame, text="Population Size:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        pop_size_entry = ttk.Entry(self.nsga_params_frame, textvariable=self.population_size)
        pop_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(self.nsga_params_frame, text="Max Generations:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        max_gen_entry = ttk.Entry(self.nsga_params_frame, textvariable=self.max_generations)
        max_gen_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(self.nsga_params_frame, text="Mutation Prob:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        mut_prob_entry = ttk.Entry(self.nsga_params_frame, textvariable=self.mutation_prob)
        mut_prob_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(self.psa_params_frame, text="Initial Temperature:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        temp_entry = ttk.Entry(self.psa_params_frame, textvariable=self.initial_temp)
        temp_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(self.psa_params_frame, text="Cooling Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        cooling_entry = ttk.Entry(self.psa_params_frame, textvariable=self.cooling_rate)
        cooling_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(self.psa_params_frame, text="Iterations:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        iter_entry = ttk.Entry(self.psa_params_frame, textvariable=self.iterations)
        iter_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Initialize with NSGA-II selected
        self.update_algorithm_controls()
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.control_frame)
        buttons_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=10)
        
        # Run button
        self.run_button = ttk.Button(buttons_frame, text="Run Algorithm", command=self.run_algorithm)
        self.run_button.pack(side=tk.LEFT, padx=5)

        # Save results button
        self.save_button = ttk.Button(buttons_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(left_panel, text="Progress")
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        self.status_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a notebook for tabbed display in the left panel
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Solution info tab
        solution_frame = ttk.Frame(self.notebook)
        self.notebook.add(solution_frame, text="Summary")
        
        # Create a Text widget for the solution details
        self.solution_text = tk.Text(solution_frame, wrap=tk.WORD, height=10, width=30)
        solution_scroll = ttk.Scrollbar(solution_frame, command=self.solution_text.yview)
        self.solution_text.config(yscrollcommand=solution_scroll.set)
        self.solution_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        solution_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Solution viewer tab for browsing solutions
        viewer_frame = ttk.Frame(self.notebook)
        self.notebook.add(viewer_frame, text="Solutions")
        
        # Solution selector frame
        selector_frame = ttk.Frame(viewer_frame)
        selector_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Solution selector dropdown
        ttk.Label(selector_frame, text="Solution:").pack(side=tk.LEFT, padx=5)
        self.solution_idx = tk.IntVar(value=0)
        self.solution_selector = ttk.Combobox(selector_frame, textvariable=self.solution_idx, state='readonly')
        self.solution_selector.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.solution_selector.bind('<<ComboboxSelected>>', self.on_solution_selected)
        
        # Solution details text widget
        self.solution_details = tk.Text(viewer_frame, wrap=tk.WORD, height=20)
        details_scroll = ttk.Scrollbar(viewer_frame, command=self.solution_details.yview)
        self.solution_details.config(yscrollcommand=details_scroll.set)
        self.solution_details.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        details_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)
        
        # Right panel - Visualization
        viz_frame = ttk.LabelFrame(right_panel, text="Visualization")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for visualization
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Operational Cost')
        self.ax.set_ylabel('Maximum Workload')
        self.ax.set_title('Pareto Front Evolution')
        self.ax.grid(True)
        
        # Canvas for displaying the plot
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
    
    def update_algorithm_controls(self):
        """Update the displayed controls based on the selected algorithm"""
        # Remove both frames first
        self.nsga_params_frame.grid_forget()
        self.psa_params_frame.grid_forget()
        
        # Show the appropriate frame based on selected algorithm
        if self.algorithm.get() == "NSGA2":
            self.nsga_params_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
            self.control_frame.config(text="NSGA-II Controls")
        else:  # PSA
            self.psa_params_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
            self.control_frame.config(text="PSA Controls")
    
    def on_solution_selected(self, event=None):
        """Handle the selection of a solution from the dropdown"""
        if not hasattr(self, 'last_result') or self.last_result is None:
            return
        
        # Get the selected solution index
        try:
            # Get the index from the combobox directly, not from the IntVar
            dropdown_idx = self.solution_selector.current()
            
            # If a valid selection is made
            if dropdown_idx >= 0:
                # Get the actual index in the result array (sorted by cost)
                sorted_idx = np.argsort(self.last_result.F[:, 0])
                solution_idx = sorted_idx[dropdown_idx]
                
                # Display the selected solution
                self.display_solution_details(solution_idx)
        except Exception as e:
            self.solution_details.delete(1.0, tk.END)
            self.solution_details.insert(tk.END, f"Error displaying solution: {str(e)}")
    
    def display_solution_details(self, solution_idx):
        """Display detailed information about the selected solution"""
        if not hasattr(self, 'last_result') or self.last_result is None:
            return
        
        # Get the solution data
        solution = self.last_result.X[solution_idx]
        objectives = self.last_result.F[solution_idx]
        
        # Clear the text widget
        self.solution_details.delete(1.0, tk.END)
        
        # Add solution header
        self.solution_details.insert(tk.END, f"Solution {solution_idx+1} Details\n")
        self.solution_details.insert(tk.END, f"{'='*40}\n\n")
        
        # Objective values
        self.solution_details.insert(tk.END, f"Operational Cost: {objectives[0]:.2f}\n")
        self.solution_details.insert(tk.END, f"Maximum Workload: {objectives[1]:.4f}\n\n")
        
        # Extract ward and day assignments
        n_patients = len(self.last_result.problem.data['patients'])
        ward_assignments = solution[:n_patients].astype(int)
        day_assignments = solution[n_patients:].astype(int)
        
        # Add patient assignments table header
        self.solution_details.insert(tk.END, "Patient Assignments:\n")
        self.solution_details.insert(tk.END, f"{'='*40}\n")
        self.solution_details.insert(tk.END, f"{'PatID':<6} {'Spec':<6} {'Ward':<6} {'Day':<4} {'LOS':<4} {'Delay':<5}\n")
        self.solution_details.insert(tk.END, f"{'-'*40}\n")
        
        # Add each patient's assignments
        for i, patient in enumerate(self.last_result.problem.data['patients']):
            ward_idx = ward_assignments[i]
            ward = self.last_result.problem.data['wards'][ward_idx]
            day = day_assignments[i]
            
            # Calculate delay
            delay = max(0, day - patient['earliest_admission'])
            
            # Only show first 20 patients to avoid flooding the UI
            if i < 20:
                self.solution_details.insert(
                    tk.END, 
                    f"{patient['id']:<6} {patient['specialization']:<6} {ward['id']:<6} "
                    f"{day:<4} {patient['length_of_stay']:<4} {delay:<5}\n"
                )
        
        if n_patients > 20:
            self.solution_details.insert(tk.END, "... (more patients not shown - save for complete results)\n")

    def run_algorithm(self):
        """Start the algorithm in a separate thread"""
        if self.running:
            messagebox.showerror("Error", "Algorithm is already running!")
            return
        
        # Validate input parameters
        try:
            pop_size = self.population_size.get()
            max_gen = self.max_generations.get()
            mut_prob = self.mutation_prob.get()
            
            if pop_size <= 0 or max_gen <= 0 or mut_prob < 0 or mut_prob > 1:
                raise ValueError("Invalid parameters")
            
            dataset = self.dataset_path.get()
            if not os.path.exists(dataset):
                raise FileNotFoundError(f"Dataset {dataset} not found")
                
        except (ValueError, FileNotFoundError) as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return
            
        # Update UI state
        self.running = True
        self.run_button.config(state=tk.DISABLED)
        self.solution_history = []
        self.solution_text.delete(1.0, tk.END)
        
        # Start the algorithm in a separate thread
        self.thread = threading.Thread(target=self._run_algorithm_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def _run_algorithm_thread(self):
        """Run the selected algorithm in a separate thread"""
        try:
            # Reset progress
            self.progress['value'] = 0
            self.solution_text.delete(1.0, tk.END)
            self.status_var.set("Loading dataset...")
            
            # Parse data
            dataset = self.dataset_path.get()
            data = parse_data(dataset)
            
            # Update status
            self.status_var.set("Setting up algorithm...")
            
            # Get the selected algorithm
            selected_algorithm = self.algorithm.get()
            
            start_time = time.time()
            
            if selected_algorithm == "NSGA2":
                # Create a local copy of callback state that won't be affected by reset operations
                callback_state = {
                    'running': True,
                    'max_generations': self.max_generations.get(),
                    'solution_history': [],
                }
                
                # Store the reference in the instance for other methods to access
                self._callback_state = callback_state
                
                # Define a callback that uses the local state rather than accessing the class directly
                def external_callback(algorithm):
                    # Check the local copy of running state
                    if not callback_state['running']:
                        return False
                        
                    # Get current generation
                    current_gen = algorithm.n_gen
                    max_gen = callback_state['max_generations']
                    
                    # Store data for later UI updates
                    objectives = algorithm.pop.get("F")
                    callback_state['solution_history'].append({
                        'generation': current_gen,
                        'objectives': objectives.tolist() if objectives is not None else []
                    })
                    
                    # Using local variables to communicate with main thread
                    callback_state['current_gen'] = current_gen
                    callback_state['progress'] = int((current_gen / max_gen) * 100)
                    
                    return True
                
                # Create algorithm with external callback
                algorithm = NSGA2(
                    pop_size=self.population_size.get(),
                    sampling=IntegerRandomSampling(),
                    crossover=TuplePointCrossover(n_points=2),
                    mutation=AdmissionDayMutation(prob=self.mutation_prob.get()),
                    eliminate_duplicates=True,
                    repair=RepairOperator()
                )
                
                # Create termination criterion
                termination = DefaultMultiObjectiveTermination(
                    n_max_gen=self.max_generations.get(),
                )
                
                # Update status
                self.status_var.set("Running NSGA-II optimization...")
                
                # Start a separate timer thread for UI updates
                stop_ui_updates = threading.Event()
                ui_update_thread = threading.Thread(
                    target=self._ui_updater, 
                    args=(stop_ui_updates, callback_state)
                )
                ui_update_thread.daemon = True
                ui_update_thread.start()
                
                try:
                    # Run the optimization
                    result = minimize(
                        BalancedWorkload(data),
                        algorithm,
                        termination=termination,
                        seed=1,
                        verbose=True,
                        save_history=True,
                        callback=external_callback
                    )
                    
                    # Calculate execution time
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Update status only if still running (not reset)
                    if callback_state['running']:
                        self.status_var.set(f"Completed in {execution_time:.2f} seconds")
                        
                        # Get solution history from callback state
                        self.solution_history = callback_state['solution_history']
                        
                        # Store the result and execution time for use in save_results
                        self.last_result = result
                        self.last_exec_time = execution_time
                        
                        # Update solution text with results
                        self.root.after(0, lambda: self.update_solution_text(result, execution_time))
                        
                        # Set progress to 100%
                        self.progress['value'] = 100
                        
                        # Update plot with final results
                        self.root.after(0, lambda: self.plot_final_results(result))
                finally:
                    # Stop UI updates
                    stop_ui_updates.set()
                    
            else:  # PSA algorithm
                # Create problem object
                problem = PatientSchedulingProblem(data)
                
                # Create PSA algorithm instance
                psa = ParetoSimulatedAnnealing(
                    problem=problem,
                    temperature=self.initial_temp.get(),
                    cooling_rate=self.cooling_rate.get(),
                    n_iterations=self.iterations.get()
                )
                
                # Initialize progress tracking
                total_iterations = self.iterations.get()
                self.solution_history = []
                
                # Create a custom progress tracker
                def psa_progress_monitor(current_iteration, temperature, front_size):
                    if not self.running:
                        return False  # Signal to stop if no longer running
                        
                    # Calculate progress percentage
                    progress = int((current_iteration / total_iterations) * 100)
                    
                    # Update UI from main thread
                    self.root.after(0, lambda: self.progress.config(value=progress))
                    self.root.after(0, lambda: self.status_var.set(
                        f"Iteration {current_iteration}/{total_iterations}, Temp: {temperature:.2f}, Front: {front_size}"
                    ))
                    
                    # Store history for visualization periodically
                    if current_iteration % 50 == 0 or current_iteration == total_iterations:
                        objectives = np.array(psa.pareto_front_objectives) if psa.pareto_front_objectives else np.array([])
                        history_entry = {
                            'generation': current_iteration,
                            'objectives': objectives.tolist() if len(objectives) > 0 else []
                        }
                        self.solution_history.append(history_entry)
                        
                        # Update plot from main thread
                        if len(objectives) > 0:
                            self.root.after(0, lambda h=history_entry: self.update_plot_with_history(h))
                    
                    return True  # Continue running
                
                # Set PSA progress monitor
                psa.progress_callback = psa_progress_monitor
                
                # Run optimization
                self.status_var.set("Running PSA optimization...")
                pareto_front, objectives = psa.optimize()
                
                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Create a result object similar to NSGA-II for compatibility
                class PSAResult:
                    def __init__(self, X, F, problem, exec_time):
                        self.X = np.array(X)
                        self.F = np.array(F)
                        self.problem = problem
                        self.success = True
                        self.algorithm = type('', (), {'n_gen': total_iterations})
                        
                result = PSAResult(pareto_front, objectives, problem, execution_time)
                
                # Store the result
                self.last_result = result
                self.last_exec_time = execution_time
                
                # Update the UI
                self.status_var.set(f"Completed in {execution_time:.2f} seconds")
                self.progress['value'] = 100
                
                # Update the interface with results
                self.root.after(0, lambda: self.update_solution_text(result, execution_time))
                self.root.after(0, lambda: self.plot_final_results(result))
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.status_var.set(f"Error: {str(e)}")
        finally:
            # Reset UI state
            self.running = False
            self.root.after(0, lambda: self.run_button.config(state=tk.NORMAL))
    
    def _ui_updater(self, stop_event, callback_state):
        """Update UI elements based on algorithm progress"""
        while not stop_event.is_set():
            # Get current progress data
            current_gen = callback_state.get('current_gen')
            progress = callback_state.get('progress')
            
            if current_gen is not None:
                # Update progress bar
                self.root.after(0, lambda p=progress: self.progress.config(value=p))
                
                # Update status text
                self.root.after(0, lambda c=current_gen, m=callback_state['max_generations']: 
                    self.status_var.set(f"Generation {c}/{m}")
                )
                
                # Update plot periodically
                if current_gen % 5 == 0 and callback_state['solution_history']:
                    # Create a local copy of the current state to avoid race conditions
                    current_history = callback_state['solution_history'][-1].copy()
                    self.root.after(0, lambda h=current_history: self.update_plot_with_history(h))
            
            time.sleep(0.1)  # Small delay to prevent high CPU usage
    
    def update_plot_with_history(self, history_entry):
        """Update the plot with a specific history entry"""
        # Clear the current plot
        self.ax.clear()
        
        objectives = history_entry.get('objectives', [])
        generation = history_entry.get('generation', 0)
        
        if objectives:
            # Convert to numpy array for plotting
            obj_array = np.array(objectives)
            
            # Plot the objectives
            self.ax.scatter(obj_array[:, 0], obj_array[:, 1], c='red', label=f'Gen {generation}')
            
            # Plot best front line (convex hull approximation)
            if len(obj_array) > 2:
                try:
                    # Sort points by first objective
                    sorted_idx = np.argsort(obj_array[:, 0])
                    sorted_obj = obj_array[sorted_idx]
                    
                    # Plot line connecting points (simple front visualization)
                    self.ax.plot(sorted_obj[:, 0], sorted_obj[:, 1], 'b--', alpha=0.5)
                except Exception:
                    pass
        
        # Set labels and title
        self.ax.set_xlabel('Operational Cost')
        self.ax.set_ylabel('Maximum Workload')
        self.ax.set_title(f'Pareto Front - Generation {generation}')
        self.ax.grid(True)
        self.ax.legend()
        
        # Redraw the canvas
        self.canvas.draw()
        
    
    def algorithm_callback(self, algorithm):
        """Callback function called by the algorithm at each generation"""
        if not self.running:
            return False  # Stop the algorithm if running is False
            
        # Get current generation
        current_gen = algorithm.n_gen
        max_gen = self.max_generations.get()
        
        # Update progress bar
        progress_value = int((current_gen / max_gen) * 100)
        self.root.after(0, lambda: self.progress.config(value=progress_value))
        
        # Update status
        self.root.after(0, lambda: self.status_var.set(f"Generation {current_gen}/{max_gen}"))
        
        # Get current Pareto front
        pop = algorithm.pop
        objectives = pop.get("F")
        
        # Store history for visualization
        self.solution_history.append({
            'generation': current_gen,
            'objectives': objectives.tolist() if objectives is not None else []
        })
        
        # Update plot every few generations
        if current_gen % 5 == 0 or current_gen == max_gen:
            self.root.after(0, lambda: self.update_plot())
            
        return True  # Continue the algorithm
    
    def update_plot(self):
        """Update the plot with the latest solution history"""
        if not self.solution_history:
            return
            
        # Clear the current plot
        self.ax.clear()
        
        # Get the latest generation data
        latest = self.solution_history[-1]
        objectives = latest['objectives']
        
        if objectives:
            # Convert to numpy array for plotting
            obj_array = np.array(objectives)
            
            # Plot the objectives
            self.ax.scatter(obj_array[:, 0], obj_array[:, 1], c='red', label=f'Gen {latest["generation"]}')
            
            # Plot best front line (convex hull approximation)
            if len(obj_array) > 2:
                try:
                    # Sort points by first objective
                    sorted_idx = np.argsort(obj_array[:, 0])
                    sorted_obj = obj_array[sorted_idx]
                    
                    # Plot line connecting points (simple front visualization)
                    self.ax.plot(sorted_obj[:, 0], sorted_obj[:, 1], 'b--', alpha=0.5)
                except Exception:
                    pass
        
        # Set labels and title
        self.ax.set_xlabel('Operational Cost')
        self.ax.set_ylabel('Maximum Workload')
        self.ax.set_title(f'Pareto Front - Generation {latest["generation"]}')
        self.ax.grid(True)
        self.ax.legend()
        
        # Redraw the canvas
        self.canvas.draw()
    
    def plot_final_results(self, result):
        """Plot the final optimization results"""
        # Clear the current plot
        self.ax.clear()
        
        # Plot the final Pareto front
        if result.F is not None and len(result.F) > 0:
            self.ax.scatter(result.F[:, 0], result.F[:, 1], c='red', s=50, label='Final Pareto Front')
            
            # Plot line connecting Pareto front points
            if len(result.F) > 1:
                # Sort by first objective
                sorted_idx = np.argsort(result.F[:, 0])
                sorted_F = result.F[sorted_idx]
                self.ax.plot(sorted_F[:, 0], sorted_F[:, 1], 'b-', alpha=0.7)
        
        # Set labels and title
        self.ax.set_xlabel('Operational Cost')
        self.ax.set_ylabel('Maximum Workload')
        self.ax.set_title('Final Pareto Front')
        self.ax.grid(True)
        self.ax.legend()
        
        # Redraw the canvas
        self.canvas.draw()
        
        # Update the solution selector
        if result.F is not None and len(result.F) > 0:
            # Create solution options for the dropdown
            sorted_idx = np.argsort(result.F[:, 0])
            options = []
            for i, idx in enumerate(sorted_idx):
                options.append(f"#{i+1}: Cost={result.F[idx, 0]:.2f}, WL={result.F[idx, 1]:.4f}")
            
            self.solution_selector['values'] = options
            
            # Select the first (best) solution
            if options:
                self.solution_selector.current(0)
                self.on_solution_selected()
    
    def update_solution_text(self, result, execution_time):
        """Update the solution text widget with results"""
        self.solution_text.delete(1.0, tk.END)
        
        # Add summary information
        self.solution_text.insert(tk.END, f"Dataset: {self.dataset_path.get().split('/')[-1]}\n")
        self.solution_text.insert(tk.END, f"Execution Time: {execution_time:.2f} seconds\n")
        self.solution_text.insert(tk.END, f"Generations: {result.algorithm.n_gen}\n")
        self.solution_text.insert(tk.END, f"Population Size: {self.population_size.get()}\n")
        self.solution_text.insert(tk.END, f"Number of Solutions: {len(result.X)}\n\n")
        
        # Display top solutions
        self.solution_text.insert(tk.END, "Top Solutions (Objective Values):\n")
        
        # Sort by first objective (operational cost)
        sorted_idx = np.argsort(result.F[:, 0])
        
        # Display top 5 solutions or fewer if there are less
        for i in range(min(5, len(sorted_idx))):
            idx = sorted_idx[i]
            obj = result.F[idx]
            self.solution_text.insert(tk.END, f"{i+1}. Cost: {obj[0]:.2f}, Workload: {obj[1]:.4f}\n")
    
    def save_results(self):
        """Save the current results to a file (text format with detailed information)"""
        if not self.solution_history:
            messagebox.showinfo("No Results", "No results to save!")
            return
            
        try:
            # Ask for file location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Results As"
            )
            
            if not file_path:  # User cancelled
                return
                
            # Create detailed text report
            with open(file_path, 'w') as f:
                # Header information
                f.write("=================================================\n")
                f.write("NSGA-II Patient Scheduling Optimization Results\n")
                f.write("=================================================\n\n")
                
                # Date and time
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Dataset info
                dataset_name = self.dataset_path.get().split('/')[-1]
                f.write(f"Dataset: {dataset_name}\n")
                
                # Algorithm parameters
                f.write("\nAlgorithm Parameters:\n")
                f.write(f"- Population Size: {self.population_size.get()}\n")
                f.write(f"- Max Generations: {self.max_generations.get()}\n")
                f.write(f"- Mutation Probability: {self.mutation_prob.get()}\n")
                
                # Get the latest optimization result from solution history
                if hasattr(self, 'last_result') and self.last_result is not None:
                    result = self.last_result
                    exec_time = self.last_exec_time
                    
                    # Execution information
                    f.write("\nExecution Information:\n")
                    f.write(f"- Total Execution Time: {exec_time:.2f} seconds\n")
                    f.write(f"- Completed Generations: {result.algorithm.n_gen}\n")
                    f.write(f"- Number of Solutions: {len(result.X)}\n")
                    f.write(f"- Success: {result.success}\n")
                    
                    # Algorithm convergence information
                    f.write("\nConvergence Information:\n")
                    if hasattr(result, "message"):
                        f.write(f"- Termination message: {result.message}\n")
                    
                    # Pareto front details
                    f.write("\nPareto Front Solutions:\n")
                    f.write("---------------------------------------------------\n")
                    f.write("     Operational Cost      Maximum Workload\n")
                    f.write("---------------------------------------------------\n")
                    
                    # Sort solutions by operational cost
                    sorted_idx = np.argsort(result.F[:, 0])
                    for i, idx in enumerate(sorted_idx):
                        f.write(f"{i+1:3d}. {result.F[idx, 0]:20.4f} {result.F[idx, 1]:20.4f}\n")
                    
                    # Evolution history summary
                    f.write("\nEvolution History:\n")
                    f.write("---------------------------------------------------\n")
                    f.write("Gen    Min Cost    Max Cost    Min Workload    Max Workload\n")
                    f.write("---------------------------------------------------\n")
                    
                    # Extract data from the solution history
                    for entry in self.solution_history:
                        gen = entry['generation']
                        objs = np.array(entry['objectives'])
                        if len(objs) > 0:
                            min_cost = np.min(objs[:, 0])
                            max_cost = np.max(objs[:, 0])
                            min_workload = np.min(objs[:, 1])
                            max_workload = np.max(objs[:, 1])
                            
                            f.write(f"{gen:3d}  {min_cost:10.2f}  {max_cost:10.2f}  {min_workload:13.4f}  {max_workload:13.4f}\n")

                    
                    # Loop through all Pareto front solutions
                    f.write("\n=================================================\n")
                    f.write("DETAILED SOLUTIONS INFORMATION\n")
                    f.write("=================================================\n")
                    
                    # Get number of patients for extraction
                    n_patients = len(result.problem.data['patients'])
                    
                    # Loop through all solutions
                    for i, sol_idx in enumerate(sorted_idx):
                        solution = result.X[sol_idx]
                        objectives = result.F[sol_idx]
                        
                        f.write(f"\n\n--------------------- SOLUTION #{i+1} ---------------------\n")
                        f.write(f"Operational Cost: {objectives[0]:.2f}\n")
                        f.write(f"Maximum Workload: {objectives[1]:.4f}\n\n")
                        
                        # Extract ward and day assignments for this solution
                        ward_assignments = solution[:n_patients].astype(int)
                        day_assignments = solution[n_patients:].astype(int)
                        
                        # Add patient assignments table header
                        f.write("Patient Assignments:\n")
                        f.write(f"{'='*40}\n")
                        f.write(f"{'PatID':<6} {'Spec':<6} {'Ward':<6} {'Day':<4} {'LOS':<4} {'Delay':<5}\n")
                        f.write(f"{'-'*40}\n")
                        
                        # Variables to track statistics
                        total_delay = 0
                        ward_usage = np.zeros(len(result.problem.data['wards']))
                        
                        # Add each patient's assignments
                        for j, patient in enumerate(result.problem.data['patients']):
                            ward_idx = ward_assignments[j]
                            ward = result.problem.data['wards'][ward_idx]
                            day = day_assignments[j]
                            
                            # Calculate delay
                            delay = max(0, day - patient['earliest_admission'])
                            total_delay += delay
                            
                            # Update ward usage count
                            ward_usage[ward_idx] += 1
                            
                            # Format the assignment row
                            f.write(f"{patient['id']:<6} {patient['specialization']:<6} {ward['id']:<6} "
                                    f"{day:<4} {patient['length_of_stay']:<4} {delay:<5}\n")
                        
                else:
                    # Basic information from solution history
                    latest = self.solution_history[-1]
                    gen = latest['generation']
                    f.write(f"\nReached generation {gen}\n")
                    
                    if 'objectives' in latest and latest['objectives']:
                        objectives = np.array(latest['objectives'])
                        f.write(f"Final generation solutions: {len(objectives)}\n")
                        f.write(f"Operational cost range: {np.min(objectives[:, 0]):.2f} - {np.max(objectives[:, 0]):.2f}\n")
                        f.write(f"Maximum workload range: {np.min(objectives[:, 1]):.4f} - {np.max(objectives[:, 1]):.4f}\n")
                
            messagebox.showinfo("Success", f"Results saved to {file_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")

    def on_close(self):
        self.root.destroy()

def main():
    # Set matplotlib backend for Tkinter
    plt.style.use('ggplot')
    
    # Create the root window
    root = tk.Tk()
    
    # Create the application
    app = OptimizationApp(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()