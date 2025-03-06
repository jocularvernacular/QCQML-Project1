import numpy as np
from math import sqrt

# Set up the basis vectors
basis_0, basis_1 = np.array([1, 0]), np.array([0, 1])
Pauli_X = np.array([[0, 1], [1, 0]])
Pauli_Y = np.array([[0, -1j], [1j, 0]])
Pauli_Z = np.array([[1, 0], [0, -1]])
Id = np.array([[1, 0], [0, 1]])

def apply_pauli_matrices(basis_state):

    results = {
        "X": np.dot(Pauli_X, basis_state),
        "Y": np.dot(Pauli_Y, basis_state),
        "Z": np.dot(Pauli_Z, basis_state)
    }
    return results

def print_pauli_results():
    print("Basis state |0>:")
    print(basis_0)
    print("\nApplying Pauli matrices to |0>:")
    results_0 = apply_pauli_matrices(basis_0)
    for matrix, result in results_0.items():
        print(f"{matrix} |0> = {result}")

    print("\nBasis state |1>:")
    print(basis_1)
    print("\nApplying Pauli matrices to |1>:")
    results_1 = apply_pauli_matrices(basis_1)
    for matrix, result in results_1.items():
        print(f"{matrix} |1> = {result}")

print_pauli_results()

def print_section_divider():
    print("-----------------------------------")

print_section_divider()

# Moving on to Hadamard and phase gates:
Hadamard = (1 / sqrt(2)) * np.array([[1, 1], [1, -1]])

def apply_phase(phase, state):
    gate = np.array([[1, 0], [0, np.exp(1j * phase)]])
    return np.dot(gate, state)

def apply_and_print(func, name, basis_0, basis_1):
    result_0 = func(basis_0)
    result_1 = func(basis_1)
    print(name + " :")
    print(f"|0> = {result_0}")
    print(f"|1> = {result_1}")

apply_and_print(lambda state: Hadamard * state, "Hadamard", basis_0, basis_1)
apply_and_print(lambda state: apply_phase(np.pi, state), "Phase", basis_0, basis_1)
print_section_divider()

# Now the bell states and CNOT stuff:
bell_phi_p = (1 / sqrt(2)) * np.array([1, 0, 0, 1])
bell_phi_m = (1 / sqrt(2)) * np.array([1, 0, 0, -1])
bell_psi_p = (1 / sqrt(2)) * np.array([0, 1, 1, 0])
bell_psi_m = (1 / sqrt(2)) * np.array([0, 1, -1, 0])

CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])

hadamard_4D = np.kron(Hadamard, np.identity(2))

def apply_hadamard_and_cnot():
    bell_hadamard = np.dot(hadamard_4D, bell_phi_p)
    print("Hadamard applied to Bell state Phi+: ")
    print(bell_hadamard)
    bell_hadamard_cnot = np.dot(CNOT, bell_hadamard)
    print("CNOT + Hadamard applied to Bell state Phi+: ")
    print(bell_hadamard_cnot)
    return bell_hadamard_cnot

def measure(state, basis, shots):
    probabilities = np.abs(state.flatten())**2
    results = np.random.choice(basis, size=shots, p=probabilities)
    counts = {}
    for base in basis:
        counts[base] = (results == base).sum()
        """
        if(counts[base] == 0):
            print("ZEEEERO")
            print("State:")
            print(state)
            print("PROBS")
            print(probabilities)"
        """
    return counts

bell_hadamard_cnot = apply_hadamard_and_cnot()
results = measure(bell_hadamard_cnot, ["00", "01", "10", "11"], 10000)
print(results)
print_section_divider()

import qiskit as qk
from qiskit_aer import Aer

def measure_bell_state(repetitions=10000):
    backend = Aer.get_backend('aer_simulator')
    qc = qk.QuantumCircuit(2, 2)
    
    # Create a bell state
    qc.h(0)
    qc.cx(0, 1)
    
    # Apply Hadamard and CNOT
    qc.h(0)
    qc.cx(0, 1)
    
    qc.measure([0, 1], [0, 1])
    compiled_circuit = qk.transpile(qc, backend)
    job = backend.run(compiled_circuit, shots=repetitions)
    results = job.result().get_counts()
    return results

### Part b)
import matplotlib.pyplot as plt

def EW_1(lambd):
    return 2 + sqrt((3 * lambd - 2)**2 + (0.2 * lambd) ** 2)

def EW_2(lambd):
    return 2 - sqrt((3 * lambd - 2)**2 + (0.2 * lambd) ** 2)

x = [i / 100 for i in range(100)]
y_e1 = [EW_1(i) for i in x]
y_e2 = [EW_2(i) for i in x]

plt.plot(x, y_e1)
plt.plot(x, y_e2)
plt.legend(["E1", "E2"])
#plt.show()

### Part c)

# Constants
E_1 = 0
E_2 = 4
V_11 = 3
V_22 = -3
V_12 = 0.2
V_21 = V_12
eps = (E_1 + E_2) / 2
omega = (E_1 - E_2) / 2
c = (V_11 + V_22) / 2
omega_z = (V_11 - V_22) / 2
omega_x = V_12

### Qiskit version
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter

ansatz = QuantumCircuit(1)
ansatz.rx(Parameter("a"), 0)
ansatz.ry(Parameter("b"), 0)

def create_Hamiltonian(x=0, y=0, z=0, I=0):
    pauli_terms = []
    if x != 0:
        pauli_terms.append(("X", x))
    if y != 0:
        pauli_terms.append(("Y", y))
    if z != 0:
        pauli_terms.append(("Z", z))
    if I != 0:
        pauli_terms.append(("I", z))
    return SparsePauliOp.from_list(pauli_terms)


H_0 = create_Hamiltonian(z=omega, I=eps)
H_1 = create_Hamiltonian(x=omega_x, z=omega_z, I=c)

# Define the Hamiltonian as a function of lambda
def H(lambd):
    return H_0 + lambd * H_1

#from qiskit.primitives import StatevectorEstimator
from qiskit.primitives import StatevectorEstimator

def expectation_value(hamiltonian, params):
    bound_ansatz = ansatz.assign_parameters(params)
    estimator = StatevectorEstimator()
    job = estimator.run([(bound_ansatz, hamiltonian)])
    result = job.result()
    expectation_value = result[0].data.evs
    return expectation_value

from qiskit_algorithms.optimizers import COBYLA
optimizer = COBYLA(maxiter=100)
initial_params = np.random.rand(2)

def VQE():
    vals = []
    for var in x:
        hamiltonian = H(var)
        def objective_function(params):
            return expectation_value(hamiltonian, params)
        result = optimizer.minimize(objective_function, initial_params)
        optimized_params = result.x
        optimized_energy = result.fun

        print("Optimized parameters:", optimized_params)
        print("Optimized energy (ground state energy):", optimized_energy)
        vals.append(optimized_energy)

    plt.plot(x, vals)

### Raw version
def prepare_state(theta, phi, target = None):
    state = np.array([1, 0])
    Rx = np.cos(theta/2) * Id - 1j * np.sin(theta/2) * Pauli_X
    Ry = np.cos(phi/2) * Id - 1j * np.sin(phi/2) * Pauli_Y
    state = Ry @ Rx @ state
    if target is not None:
        state = target
    return state

def prepare_Hamiltonian(x=0, y=0, z=0, I=0):
    return (x * Pauli_X) + (y * Pauli_Y) + (z * Pauli_Z) + (I * Id)

H_0 = prepare_Hamiltonian(z=omega, I=eps)
H_1 = prepare_Hamiltonian(x=omega_x, z=omega_z, I=c)

def Hamiltonian(lambd):
    return H_0 + lambd * H_1

def get_energy(angles, lmb, number_shots, target = None):
    theta, phi = angles[0], angles[1]
    init_state = prepare_state(theta, phi, target)
    
    measure_z = measure(init_state, ["0", "1"], number_shots)
    # expected value of Z = (number of 0 measurements - number of 1 measurements)/ number of shots
    # number of 1 measurements = sum(measure_z)
    exp_val_z = (omega + lmb*omega_z)*(number_shots - 2*measure_z["1"]) / number_shots

    measure_x = measure(np.matmul(init_state, Hadamard), ["0", "1"], number_shots)
    exp_val_x = lmb*omega_x*(number_shots - 2*measure_x["1"]) / number_shots
    
    exp_val_i = (eps + c*lmb)
    exp_val = (exp_val_z + exp_val_x + exp_val_i)
    return exp_val

def minimize_energy(lmb, number_shots, angles_0, learning_rate, max_epochs):
    # angles = np.random.uniform(low = 0, high = np.pi, size = 2)
    angles = angles_0 #lmb*np.array([np.pi, np.pi])
    epoch = 0
    delta_energy = 1
    energy = get_energy(angles, lmb, number_shots)
    while (epoch < max_epochs) and (delta_energy > 1e-4):
        grad = np.zeros_like(angles)
        for idx in range(angles.shape[0]):
            angles_temp = angles.copy()
            angles_temp[idx] += np.pi/2 
            E_plus = get_energy(angles_temp, lmb, number_shots)
            angles_temp[idx] -= np.pi
            E_minus = get_energy(angles_temp, lmb, number_shots)
            grad[idx] = (E_plus - E_minus)/2 
        angles -= learning_rate*grad 
        new_energy = get_energy(angles, lmb, number_shots)
        delta_energy = np.abs(new_energy - energy)
        energy = new_energy
        epoch += 1
    return angles, epoch, (epoch < max_epochs), energy, delta_energy

number_shots_search = 10_000
number_shots = 10_000
learning_rate = 0.3
max_epochs = 400
lmbvalues = np.linspace(0.0, 1.0, 30)
min_energy = np.zeros(len(lmbvalues))
epochs = np.zeros(len(lmbvalues))
for index, lmb in enumerate(lmbvalues):
    memory = 0
    angles_0 = np.random.uniform(low = 0, high = np.pi, size = 2)
    angles, epochs[index], converged, energy, delta_energy = minimize_energy(lmb, number_shots_search, angles_0, learning_rate, max_epochs)
    if epochs[index] < (epochs[index-1] - 5):
        angles_0 = np.random.uniform(low = 0, high = np.pi, size = 2)
        angles, epochs[index], converged, energy, delta_energy = minimize_energy(lmb, number_shots_search, angles_0, learning_rate, max_epochs)
    min_energy[index] = get_energy(angles, lmb, number_shots)

from scipy.optimize import minimize
number_shots = 10_000
lmbvalues_scipy = np.linspace(0.0, 1.0, 50)
min_energy_scipy = np.zeros(len(lmbvalues_scipy))
for index, lmb in enumerate(lmbvalues_scipy):
    angles_start = np.random.uniform(low = 0, high = np.pi, size = 4)
    res = minimize(get_energy, angles_start, args = (lmb, number_shots), method = 'Powell', options = {'maxiter': 1000}, tol = 1e-5)
    min_energy_scipy[index] = res.fun

lmbvalues_ana = np.arange(0, 1, 0.01)
eigvals_ana = np.zeros((len(lmbvalues_ana), 2))
for index, lmb in enumerate(lmbvalues_ana):
    H = Hamiltonian(lmb)
    eigen, eigvecs = np.linalg.eig(H)
    permute = eigen.argsort()
    eigvals_ana[index] = eigen[permute]
    eigvecs = eigvecs[:,permute]

fig, axs = plt.subplots(1, 1, figsize=(10, 10))
for i in range(2):
    axs.plot(lmbvalues_ana, eigvals_ana[:,i], label=f'$E_{i+1}$', color = '#4c72b0')
axs.scatter(lmbvalues, min_energy, label = 'VQE eigenvalues', color = '#dd8452')
axs.scatter(lmbvalues_scipy, min_energy_scipy, label = 'VQE Scipy', color = '#55a868')
axs.set_xlabel(r'$\lambda$')
axs.set_ylabel('Energy')
plt.legend()
plt.show()