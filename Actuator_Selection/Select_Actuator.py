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


import numpy as np
from scipy.linalg import solve_continuous_are
import control


class Location_selection():
    def __init__(self,pulse_responses, base_response, rank=200,r=8):
        self.pulse_responses = pulse_responses
        self.base_response = base_response
        self.rank = rank
        self.U = np.zeros((self.rank,self.rank))
        self.V_T = np.zeros((self.rank,self.rank))
        self.S =[]
        self.U_r = np.zeros((self.rank,self.rank))
        self.V_T_r = np.zeros((self.rank,self.rank))
        self.Sigma_r = np.zeros((self.rank,self.rank))
        self.r = r
        self.Sigma = np.zeros((self.rank,self.rank))
        
        
        self.hankel_matrix0 = self.construct_hankel_matrix(self.pulse_responses[0:],self.base_response)
        self.U,self.S,self.V_T,self.Sigma_r,self.U_r,self.V_T_r = self.SVD(self.hankel_matrix0)
        for i in range(200):
            print(self.S[i])
            
        self.hankel_matrix1 = self.construct_hankel_matrix(self.pulse_responses[1:],self.base_response)
        self.A_d, self.B_d, self.C_d = self.Minimal_realization(self.U_r,self.Sigma_r,self.V_T_r,self.hankel_matrix1)
        self.A_c, self.B_c, self.C_c = self.Tustin_approximation(self.A_d,self.B_d,self.C_d)
        print(self.A_c.shape,"A_c")
        print(self.B_c.shape,"B_c")
        self.X = self.Solve_for_X(self.A_c,self.B_c)
        self.P = self.Solve_for_P(self.A_c,self.B_c,self.X)
        norm = self.Calculate_norm(self.C_c,self.P)
        print(norm)
        
        
    def construct_hankel_matrix(self,pulse_responses, base_response):
#         assert pulse_responses.shape==base_response.shape,"The pulse_responses and the base_response should be in the same shape!"
#         C0 = np.mean(base_response)
        C0 = 1.636
        self.pulse_responses = [pulse_response-C0 for pulse_response in pulse_responses ]
        hankel_matrix = np.array([self.pulse_responses[i:i+self.rank] for i in range(self.rank)])
        return hankel_matrix
    
    def SVD(self,hankel_matrix):
        U,S,V_T = np.linalg.svd(hankel_matrix)
        for i in range(self.r):
            self.Sigma_r[i,i] = S[i]
        self.U_r[:,0:self.r] = U[:,0:self.r]
        self.V_T_r[0:self.r,:] = V_T[0:self.r,:]
        return U,S,V_T,self.Sigma_r,self.U_r,self.V_T_r
    
    def Minimal_realization(self,U_r,Sigma_r,V_T_r,H_prime):
        A_d = self.Matrix_power(Sigma_r,-0.5).dot(np.transpose(U_r)).dot(H_prime).dot(V_T_r).dot(self.Matrix_power(Sigma_r,-0.5))
        B_d = self.Matrix_power(Sigma_r,0.5).dot(V_T_r)[:,0]
        C_d = U_r.dot(self.Matrix_power(Sigma_r,0.5))[0,:]
        return A_d,B_d,C_d
    
    def Tustin_approximation(self,A_d,B_d,C_d):
        # Get the shape of the matrix A_discrete
        n, m = A_d.shape

        # Identity matrix of appropriate size
        I = np.eye(n)
        
        T = 0.01

        # Apply Tustin's approximation to obtain continuous-time matrices
        A_c = np.dot(2/T, np.dot(np.linalg.inv(I - 0.5 * T * A_d), (I + 0.5 * T * A_d)))
        B_c = np.dot(2/T, np.dot(np.linalg.inv(I - 0.5 * T * A_d), B_d)).reshape(-1,1)
        C_c = C_d
        return A_c,B_c,C_c
    
    def Matrix_power(self,matrix,power_times):
        rank = np.linalg.matrix_rank(matrix)+1
        for i in range(rank):
            if matrix[i,i] != 0:
                matrix[i,i] = matrix[i,i]**power_times
        return matrix
            
        
    def Solve_for_X(self,A,B):
        q = np.zeros((A.shape[0],A.shape[0]))
#         X = solve_continuous_are(A,B,q,r=1)
        X,L,G = control.care(A,B,q)
#         X = np.eye(A.shape[0])

#         # Set the maximum number of iterations and convergence threshold
#         max_iterations = 100
#         tolerance = 1e-6
#         print("X:",X.shape,"A:",A.shape,"B:",B.shape)
#         # Perform Riccati iteration
#         for i in range(max_iterations):
#             # Update X using the Riccati iteration formula
#             X_new = X - X @ B @ np.linalg.inv(np.eye(A.shape[0]) + B.T @ X @ B) @ B.T @ X

#             # Check for convergence
#             if np.linalg.norm(X_new - X) < tolerance:
#                 X = X_new
#                 break

#             X = X_new
        return X
    
    def Solve_for_P(self,A,B,X):
        F = -np.transpose(B).dot(X)
        P = solve_continuous_are(A + np.dot(B, F.T), B, np.dot(B, B.T))
        return P
    
    def Calculate_norm(self,C_c,P):
        norm = np.sqrt(C_c.dot(P).dot(C_c.T))
        return norm
            
    
    def reconstruct_hankel_matrix(self,U_r,Sigma_r,V_T_r):
        reconstructed_hankel_matrix = U_r.dot(Sigma_r).dot(T_r)
        return reconstructed_hankel_matrix
        
        
