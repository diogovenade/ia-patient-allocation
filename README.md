# IA Project 1 - Generating balanced workload allocations in hospitals (Grade: 20/20)

Based on https://www.sciencedirect.com/science/article/abs/pii/S2211692323000139 and https://data.mendeley.com/datasets/3mv4rtxtfs/1

To run this program, it is necessary to install the pymoo package with the following command.

```pip install -U pymoo```

It may be necessary to install aditional packages as well (such as numpy and matplotlib), depending on your current setup.

You can launch the program by running the main.py file:

```python main.py```

The program will launch a graphical user interface (GUI) for patient scheduling optimization.

How to use:

- Select Algorithm: Choose between "NSGA-II" or "Pareto Simulated Annealing" in the "Algorithm Selection" section.
- Set Parameters: Adjust the algorithm-specific parameters in the "Algorithm Controls" section.
- Select Dataset: Choose a dataset file from the dropdown menu.
- Run Algorithm: Click the "Run Algorithm" button to start the optimization process.
- View Results:
    - Check the "Summary" tab for a high-level overview of the results.
    - Use the "Solutions" tab to browse detailed information about individual solutions.
    - View the Pareto front evolution in the "Visualization" panel.
- Save Results: Click the "Save Results" button to export the detailed results to a file.
- Compare Algorithms: Use the "Compare Algorithms" button to run both algorithms (with your chosen parameters) on the same dataset and compare their performance.

This project was developed by:
- Daniel Basílio
- Diogo Venade
- Margarida Fonseca
