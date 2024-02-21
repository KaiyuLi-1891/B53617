import numpy as np
import pyyeti.era as era


class Actuator_Selection:
    def __init__(self, N, pulse_responses, baseline_response, dt):
        self.N = N
        self.pulse_responses = pulse_responses
        self.baseline_response = baseline_response
        self.dt = dt
        pass

    def collect_data(self, N, pulse_responses, baseline_response):
        C0 = np.mean(baseline_response)
        return [pulse_response - C0 for pulse_response in pulse_responses]

    def identify_system_realizations(self, markov_parameters, dt=self.dt):
        system_realizations = []
        for hi in markov_parameters:
            # ERA
            sys = era.sspa(hi, 10, 10)

            # Tustin's approximation for conversion to continuous-time
            A = sys.A
            B = sys.B
            C = sys.C
            Ad = np.eye(A.shape[0]) + A * dt
            Bd = B * dt
            Cd = C

            system_realizations.append((Ad, Bd, Cd))
        return system_realizations

    def compute_H2_optimality_measures(self, system_realizations):
        H2_norms = []
        for (Ad, Bd, Cd) in system_realizations:
            P = np.dot(Cd, np.dot(np.linalg.inv(np.eye(Ad.shape[0]) - Ad), Bd))
            H2_norms.append(np.linalg.norm(P, 'fro'))
        return H2_norms

    def select_optimal_actuator(self, H2_norms):
        sorted_indices = np.argsort(H2_norms)[::-1]
        optimal_actuator_index = sorted_indices[0]
        return optimal_actuator_index

    def visualize_H2_norms(self, H2_norms):
        plt.figure()
        plt.bar(range(len(H2_norms)), H2_norms)
        plt.xlabel('Actuator Location')
        plt.ylabel('H2 Norm')
        plt.title('H2 Norm of Each Actuator')
        plt.show()

    def single_run(self):
        markov_parameters = self.collect_data(self.N, self.pulse_responses, self.baseline_response)
        system_realizations = self.identify_system_realizations(markov_parameters, self.dt)
        H2_norms = self.compute_H2_optimality_measures(system_realizations)
        optimal_actuator_index = self.select_optimal_actuator(H2_norms)
        print("Optimal actuator location:", optimal_actuator_index)
        self.visualize_H2_norms(H2_norms)
