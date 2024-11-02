import numpy as np

class LIFNetwork:
    def __init__(self, total_neurons, proportion_exc, V_rest=-65.0, V_th=-50.0, V_reset=-70.0, tau_m=10.0, I_ext=5.0, dt=0.1, synaptic_scaling=0.1):
        self.total_neurons = total_neurons  
        self.V_rest = V_rest 
        self.V_th = V_th  
        self.V_reset = V_reset  
        self.tau_m = tau_m  
        self.I_ext = I_ext 
        self.dt = dt 
        self.synaptic_scaling = synaptic_scaling  

        self.V = np.full(total_neurons, V_rest, dtype=float)

        excitatory_neurons = int(proportion_exc * total_neurons)
        inhibitory_neurons = total_neurons - excitatory_neurons
        self.W = self._initialize_weights(excitatory_neurons, inhibitory_neurons)

    def _initialize_weights(self, excitatory_neurons, inhibitory_neurons):
        W = np.zeros((self.total_neurons, self.total_neurons), dtype=float)
        
        # Assign positive weights for excitatory neurons and negative weights for inhibitory neurons
        for i in range(self.total_neurons):
            if i < excitatory_neurons:
                W[i, :] = np.random.uniform(0.1, 0.5, size=self.total_neurons)  
            else:
                W[i, :] = np.random.uniform(-0.5, -0.1, size=self.total_neurons)  
        
        # Remove self-connections by setting the diagonal to zero
        np.fill_diagonal(W, 0)
        
        return W

    def run(self, time_steps):
        active_count = 0

        for t in range(time_steps):
            synaptic_input = self.synaptic_scaling * np.dot(self.W, (self.V >= self.V_th).astype(float))

            total_input = self.I_ext + synaptic_input
            dV = (-(self.V - self.V_rest) + total_input) / self.tau_m * self.dt
            self.V += dV

            spikes = self.V >= self.V_th  
            active_count += np.sum(spikes)  
            self.V[spikes] = self.V_reset  

        avg_activity = active_count / (self.total_neurons * time_steps)
        return avg_activity
