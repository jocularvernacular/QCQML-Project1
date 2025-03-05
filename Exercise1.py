import numpy as np
from math import sqrt

# Set up the basis vectors
basis_0, basis_1 = np.array([1, 0]), np.array([0, 1])

def apply_pauli_matrices(basis_state):
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    results = {
        "X": np.dot(X, basis_state),
        "Y": np.dot(Y, basis_state),
        "Z": np.dot(Z, basis_state)
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
H = (1 / sqrt(2)) * np.array([[1, 1], [1, -1]])

def apply_phase(phase, state):
    gate = np.array([[1, 0], [0, np.exp(1j * phase)]])
    return np.dot(gate, state)

def apply_and_print(func, name, basis_0, basis_1):
    result_0 = func(basis_0)
    result_1 = func(basis_1)
    print(name + " :")
    print(f"|0> = {result_0}")
    print(f"|1> = {result_1}")

apply_and_print(lambda state: H * state, "Hadamard", basis_0, basis_1)
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

hadamard_4D = np.kron(H, np.identity(2))

def apply_hadamard_and_cnot():
    bell_hadamard = np.dot(hadamard_4D, bell_phi_p)
    print("Hadamard applied to Bell state Phi+: ")
    print(bell_hadamard)
    bell_hadamard_cnot = np.dot(CNOT, bell_hadamard)
    print("CNOT + Hadamard applied to Bell state Phi+: ")
    print(bell_hadamard_cnot)
    return bell_hadamard_cnot

def simulate_measurements(state):
    probabilities = np.abs(state.flatten())**2
    basis_states = ["00", "01", "10", "11"]
    n_measurements = 10000
    results = np.random.choice(basis_states, size=n_measurements, p=probabilities)
    unique, counts = np.unique(results, return_counts=True)
    measurement_results = dict(zip(unique, counts / n_measurements))
    return measurement_results

bell_hadamard_cnot = apply_hadamard_and_cnot()
results = simulate_measurements(bell_hadamard_cnot)
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
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter

# Constants
E_1 = 0
E_2 = 4
V_11 = 3
V_22 = -3
V_12 = 0.2
V_21 = V_12

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

# H_0
eps = (E_1 + E_2) / 2
omega = (E_1 - E_2) / 2
H_0 = create_Hamiltonian(z=omega, I=eps)

# H_1
c = (V_11 + V_22) / 2
omega_z = (V_11 - V_22) / 2
omega_x = V_12
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
    print(expectation_value)
    return expectation_value

from qiskit_algorithms.optimizers import COBYLA
optimizer = COBYLA(maxiter=100)
initial_params = np.random.rand(2)

def objective_function(params):
    return expectation_value(hamiltonian, params)

vals = []
for var in x:
    hamiltonian = H(var)
    result = optimizer.minimize(objective_function, initial_params)
    optimized_params = result.x
    optimized_energy = result.fun

    print("Optimized parameters:", optimized_params)
    print("Optimized energy (ground state energy):", optimized_energy)
    vals.append(optimized_energy)

plt.plot(x, vals)
plt.show()