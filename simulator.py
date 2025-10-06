import numpy as np
import matplotlib.pyplot as plt
import time

class AgentSimulator:
    """
    Simulates a classical agent on a 2D triangular lattice searching for resources.
    The task is to determine the phase transition of task-solvability.
    """

    def __init__(self, L, resource_density):
        """
        Initializes the simulation environment.

        Args:
            L (int): The side length of the square grid representing the lattice.
            resource_density (float): The probability 'p' for a site to have a resource.
        """
        self.L = L
        self.p = resource_density
        # For a triangular lattice, each node has 6 neighbors.
        # We use a square grid and define neighbors accordingly.
        # Neighbors (dr, dc) for a triangular lattice mapped to a grid:
        # On even rows (r): (0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)
        # On odd rows (r):  (0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)
        self.neighbors_even = np.array([[-1, 0], [-1, 1], [0, -1], [0, 1], [1, 0], [1, 1]])
        self.neighbors_odd = np.array([[-1, -1], [-1, 0], [0, -1], [0, 1], [1, -1], [1, 0]])
        
        # This grid will be regenerated for each Monte Carlo run.
        self.grid = None

    def _generate_grid(self):
        """Creates a new random resource grid."""
        self.grid = (np.random.rand(self.L, self.L) < self.p).astype(int)

    def run_single_agent(self, max_steps):
        """
        Simulates a single agent's journey for one configuration.

        Args:
            max_steps (int): The number of steps required for a task to be successful.

        Returns:
            bool: True if the agent succeeds, False if it gets trapped.
        """
        self._generate_grid()
        
        # Find all sites with resources to select a starting point
        resource_sites = np.argwhere(self.grid == 1)
        if len(resource_sites) == 0:
            return False  # No resources, impossible to start

        # Start at a random resource site
        start_pos_idx = np.random.randint(0, len(resource_sites))
        current_pos = resource_sites[start_pos_idx]

        path_length = 0
        while path_length < max_steps:
            # Consume resource at the current position
            self.grid[current_pos[0], current_pos[1]] = 0
            
            # Get the correct neighbor offsets based on row parity
            row, col = current_pos
            if row % 2 == 0:
                neighbor_offsets = self.neighbors_even
            else:
                neighbor_offsets = self.neighbors_odd

            # Find available neighbors with resources
            possible_next_steps = []
            for offset in neighbor_offsets:
                # Apply periodic boundary conditions
                next_pos = (current_pos + offset) % self.L
                if self.grid[next_pos[0], next_pos[1]] == 1:
                    possible_next_steps.append(next_pos)
            
            # Check for trapping condition (Local Resource Prison)
            if not possible_next_steps:
                return False  # Trapped, task failed

            # Move to a randomly chosen valid neighbor
            next_step_idx = np.random.randint(0, len(possible_next_steps))
            current_pos = possible_next_steps[next_step_idx]
            path_length += 1
            
        return True # Survived for max_steps, task succeeded

    def run_monte_carlo(self, num_runs, max_steps_factor=1.0):
        """
        Runs the full Monte Carlo simulation for a given resource density.

        Args:
            num_runs (int): The number of independent simulations to average over.
            max_steps_factor (float): The task length as a factor of L.

        Returns:
            float: The success rate (Phi).
        """
        success_count = 0
        max_steps = int(self.L * max_steps_factor)
        
        for _ in range(num_runs):
            if self.run_single_agent(max_steps):
                success_count += 1
        
        return success_count / num_runs

def plot_phase_transition(densities, success_rates):
    """Plots the data for Figure 1."""
    plt.figure(figsize=(8, 6))
    plt.plot(densities, success_rates, 'o-', color='red', label='Simulated Success Rate ($\Phi$)')
    plt.axvline(x=0.3015, color='gray', linestyle='--', label='Theoretical $p_c \\approx 0.3015$')
    plt.title('Figure 1: Phase Transition of Task-Solvability', fontsize=16)
    plt.xlabel('Resource Density ($\\rho_s$)', fontsize=12)
    plt.ylabel('Success Rate ($\Phi$)', fontsize=12)
    plt.xlim(0.2, 0.4)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_finite_size_scaling(fss_data):
    """Plots the data for Figure 2."""
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'orange']
    markers = ['o', 's', '^']
    
    for i, L in enumerate(fss_data.keys()):
        densities = sorted(fss_data[L].keys())
        rates = [fss_data[L][p] for p in densities]
        plt.plot(densities, rates, marker=markers[i], color=colors[i], label=f'L = {L}')
        
    plt.title('Figure 2: Finite-Size Scaling Analysis', fontsize=16)
    plt.xlabel('Resource Density ($p$)', fontsize=12)
    plt.ylabel('Success Rate ($\Phi$)', fontsize=12)
    plt.xlim(0.28, 0.32)
    plt.ylim(0.2, 0.8)
    plt.grid(True)
    plt.legend()
    plt.show()


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # --- Part 1: Generate data for the main phase transition curve (Fig 1) ---
    print("--- Generating data for Figure 1: Phase Transition Curve ---")
    sim_L = 50
    num_mc_runs = 2000 # Increase for smoother curve, but will be slower
    
    densities_fig1 = np.linspace(0.22, 0.38, 17)
    success_rates_fig1 = []
    
    start_time = time.time()
    for p in densities_fig1:
        simulator = AgentSimulator(L=sim_L, resource_density=p)
        rate = simulator.run_monte_carlo(num_runs=num_mc_runs, max_steps_factor=1.0)
        success_rates_fig1.append(rate)
        print(f"L={sim_L}, p={p:.4f}, Success Rate={rate:.4f}")
    
    print(f"Figure 1 simulation took {time.time() - start_time:.2f} seconds.")
    plot_phase_transition(densities_fig1, success_rates_fig1)

    # --- Part 2: Generate data for Finite-Size Scaling analysis (Fig 2) ---
    print("\n--- Generating data for Figure 2: Finite-Size Scaling ---")
    system_sizes = [30, 50, 70]
    densities_fss = np.linspace(0.28, 0.32, 5) # Focus around the critical point
    num_mc_runs_fss = 1000 # Can be lower since we focus on the crossing
    
    fss_results = {L: {} for L in system_sizes}
    
    start_time = time.time()
    for L in system_sizes:
        for p in densities_fss:
            simulator = AgentSimulator(L=L, resource_density=p)
            rate = simulator.run_monte_carlo(num_runs=num_mc_runs_fss, max_steps_factor=1.0)
            fss_results[L][p] = rate
            print(f"L={L}, p={p:.4f}, Success Rate={rate:.4f}")
            
    print(f"Figure 2 simulation took {time.time() - start_time:.2f} seconds.")
    plot_finite_size_scaling(fss_results)
