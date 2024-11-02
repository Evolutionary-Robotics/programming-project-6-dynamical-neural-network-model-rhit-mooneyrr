# simulate_lif.py

import numpy as np
import matplotlib.pyplot as plt
from lif_model import LIFNetwork

def lif_simulation(proportion_exc, I_ext, repetitions, total_neurons=100, time_steps=1000, dt=0.1):

    avg_activities = []

    for _ in range(repetitions):
        network = LIFNetwork(total_neurons=total_neurons, proportion_exc=proportion_exc, I_ext=I_ext, dt=dt)
        avg_activity = network.run(time_steps)
        avg_activities.append(avg_activity)
    
    return avg_activities

def plot_avg_activity(proportion_exc, I_ext_values, repetitions=10, total_neurons=100, time_steps=1000, dt=0.1):

    avg_activity_values = []

    for I_ext in I_ext_values:
        avg_activity = lif_simulation(proportion_exc, I_ext, repetitions, total_neurons, time_steps, dt)
        avg_activity_values.append(sum(avg_activity) / repetitions)

    plt.figure(figsize=(10, 6))
    plt.plot(I_ext_values, avg_activity_values, marker='o', linestyle='-', color='b')
    plt.xlabel("External Current (I_ext)")
    plt.ylabel("Average Activity Level")
    plt.title(f"Average Activity vs. External Current (Proportion of Excitatory Neurons: {proportion_exc})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    I_ext_values = np.linspace(0, 20, 10) 
    proportion_exc = 0.5 
    repetitions = 10 
    plot_avg_activity(proportion_exc, I_ext_values, repetitions)
