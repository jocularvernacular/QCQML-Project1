import numpy as np
from math import sqrt

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

lambda_range = np.linspace(0.0, 1.0, 30)
def plot_eigenvalues():
    y_e1 = [EW_1(i) for i in lambda_range]
    y_e2 = [EW_2(i) for i in lambda_range]

    plt.plot(lambda_range, y_e1)
    plt.plot(lambda_range, y_e2)
    plt.legend(["E1", "E2"])
    plt.show()

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

def H(lambd):
    return H_0 + lambd * H_1

from scipy.optimize import minimize

def Rx(theta):
    return np.cos(theta/2) * Id - 1j * np.sin(theta/2) * Pauli_X

def Ry(phi):
    return np.cos(phi/2) * Id - 1j * np.sin(phi/2) * Pauli_Y

def prepare_state(args):
    state = np.array([1, 0])
    theta = args[0]
    phi = args[1]
    rx = Rx(theta)
    ry = Ry(phi)
    state = ry @ rx @ state
    return state

def prepare_Hamiltonian(x=0, y=0, z=0, I=0):
    return (x * Pauli_X) + (y * Pauli_Y) + (z * Pauli_Z) + (I * Id)

H_0 = prepare_Hamiltonian(z=omega, I=eps)
H_1 = prepare_Hamiltonian(x=omega_x, z=omega_z, I=c)

def Hamiltonian(H_0, H_1, lambd):
    return H_0 + lambd * H_1

def get_energy(angles, lmb, number_shots):
    theta, phi = angles[0], angles[1]
    init_state = prepare_state([theta, phi])
    
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

def minimize_energy_scipy(fnc):
    number_shots = 10_000
    min_energy = np.zeros(len(lambda_range))
    for index, lmb in enumerate(lambda_range):
        angles_start = np.random.uniform(low = 0, high = np.pi, size = 4)
        res = minimize(fnc, angles_start, args = (lmb, number_shots), method = 'Powell', options = {'maxiter': 1000}, tol = 1e-5)
        min_energy[index] = res.fun
    return min_energy

def find_eigenvalues_scipy(n_angles):
    eigvals = np.zeros((len(lambda_range), n_angles))
    for index, lmb in enumerate(lambda_range):
        H = Hamiltonian(H_0, H_1, lmb)
        eigen, eigvecs = np.linalg.eigh(H)
        eigvals[index] = eigen
    return eigvals

def VQE(n_angles):
    number_shots_search = 10_000
    learning_rate = 0.3
    max_epochs = 400
    min_energy = np.zeros(len(lambda_range))
    epochs = np.zeros(len(lambda_range))
    for index, lmb in enumerate(lambda_range):
        memory = 0
        angles_0 = np.random.uniform(low = 0, high = np.pi, size = n_angles)
        angles, epochs[index], converged, min_energy[index], delta_energy = minimize_energy(lmb, number_shots_search, angles_0, learning_rate, max_epochs)
        if epochs[index] < (epochs[index-1] - 5):
            angles_0 = np.random.uniform(low = 0, high = np.pi, size = n_angles)
            angles, epochs[index], converged, min_energy[index], delta_energy = minimize_energy(lmb, number_shots_search, angles_0, learning_rate, max_epochs)
        print(f'Lambda = {lmb}, Energy = {min_energy[index]}, Epochs = {epochs[index]}, Converged = {converged}, Delta Energy = {delta_energy}')

    min_energy_scipy = minimize_energy_scipy(get_energy)

    eigvals_ana = find_eigenvalues_scipy(n_angles)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    for i in range(n_angles):
        axs.plot(lambda_range, eigvals_ana[:,i], label=f'$E_{i+1}$')#color = '#4c72b0')
    axs.scatter(lambda_range, min_energy, label = 'VQE eigenvalues', color = '#dd8452')
    axs.scatter(lambda_range, min_energy_scipy, label = 'VQE Scipy', color = '#55a868')
    axs.set_xlabel(r'$\lambda$')
    axs.set_ylabel('Energy')
    plt.legend()
    plt.show()

# Part d)
Hx = 2.0
Hz = 3.0
# H_0
Energiesnoninteracting = [0.0, 2.5, 6.5, 7.0]
H_0 = np.diag(Energiesnoninteracting)
H_1 = Hx * np.kron(Pauli_X, Pauli_X) + Hz * np.kron(Pauli_Z, Pauli_Z)

def trace_out(state, index):
    density = np.outer(state, np.conj(state))
    if index == 0:
        op0 = np.kron(basis_0, Id)
        op1 = np.kron(basis_1, Id)
    elif index == 1:
        op0 = np.kron(Id, basis_0)
        op1 = np.kron(Id, basis_1)
    return op0.conj() @ density @ op0.T + op1.conj() @ density @ op1.T # need to take conj() on first and .T on second since np.arrays are 

eigenvalues = []
entropy = np.zeros((len(lambda_range), 4))
for index, lmb in enumerate(lambda_range):
    Hamilt = Hamiltonian(H_0, H_1, lmb)
    eigvals, eigvecs = np.linalg.eigh(Hamilt)
    eigenvalues.append(eigvals)
    for i in range(4):
        sub_density = trace_out(eigvecs[:, i], 0) # trace out qubit 0 from the ground state
        lmb_density = np.linalg.eigvalsh(sub_density)
        lmb_density = np.ma.masked_equal(lmb_density, 0).compressed() # remove zeros to avoid log(0)
        entropy[index, i] = -np.sum(lmb_density*np.log2(lmb_density))
eigenvalues = np.array(eigenvalues)

def plot_energies_and_entropy():
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(4):
        axs.plot(lambda_range, eigenvalues[:, i], label=f'$E_{i}$')
    axs.set_xlabel(r'$\lambda$')
    axs.set_ylabel('Energy')
    axs.legend()
    plt.show()

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(1):
        axs.plot(lambda_range, entropy[:, i], label=f'$H_{i}$')
    axs.set_xlabel(r'$\lambda$')
    axs.set_ylabel('Entropy')
    axs.legend()
    plt.show()

# Part e)
def apply_to_qubit(operator, qubit_index):
    if qubit_index == 0:
        return np.kron(operator, Id)
    elif qubit_index == 1:
        return np.kron(Id, operator)
    
CNOT_10 = np.array([[1,0,0,0],
                 [0,0,0,1],
                 [0,0,1,0],
                 [0,1,0,0]])
SWAP = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]])
S_dag = np.array([[1, 0], [0, -1j]])

def transform_to_Z_basis(pauli_string):
    gates = {
        "X": Hadamard,
        "ZI": np.kron(Id, Id),
        "IZ": SWAP,
        "XI": np.kron(Hadamard, Id),
        "IX": np.kron(Hadamard, Id) @ SWAP,
        "YI": np.kron(S_dag @ Hadamard, Id),
        "IY": np.kron(S_dag @ Hadamard, Id) @ SWAP,
        "ZZ": CNOT_10,
        "XX": CNOT_10 @ np.kron(Hadamard, Hadamard),
        "YY": CNOT_10 @ np.kron(S_dag @ Hadamard, S_dag @ Hadamard),
        "ZX": CNOT_10 @ np.kron(Id, Hadamard),
        "XZ": CNOT_10 @ np.kron(Id, Hadamard) @ SWAP
    }
    return gates.get(pauli_string, None)

def prepare_state(args):
    theta0, phi0, theta1, phi1 = args[0], args[1], args[2], args[3]
    qubit = np.array([1, 0, 0, 0])
    Rx0 = Rx(theta0)
    Ry0 = Ry(phi0)
    Rx1 = Rx(theta1)
    Ry1 = Ry(phi1)
    state = apply_to_qubit(Rx0, 0) @ qubit
    state = apply_to_qubit(Ry0, 0) @ state
    state = apply_to_qubit(Rx1, 1) @ state
    state = apply_to_qubit(Ry1, 1) @ state
    return CNOT @ state # CNOT ensures entanglement

def get_energy(angles, lmb, number_shots):
    Hx = 2.0 
    Hz = 3.0
    eps00, eps01, eps10, eps11 = np.array([0.0, 2.5, 6.5, 7.0])
    A = (eps00 + eps01 + eps10 + eps11) / 4.0
    B = (eps00 - eps01 + eps10 - eps11) / 4.0
    C = (eps00 + eps01 - eps10 - eps11) / 4.0
    D = (eps00 - eps01 - eps10 + eps11) / 4.0
    
    init_state = prepare_state(angles)

    ZI = np.kron(Pauli_Z, Id)

    qubit = SWAP @ init_state # rotate measurement basis
    measure_iz = measure(transform_to_Z_basis("IZ") @ init_state, ["00", "01", "10", "11"], number_shots)

    measure_zi = measure(init_state, ["00", "01", "10", "11"], number_shots)
    
    qubit = CNOT_10 @ init_state
    measure_zz = measure(transform_to_Z_basis("ZZ") @ init_state, ["00", "01", "10", "11"], number_shots)
    
    qubit = CNOT_10 @ apply_to_qubit(Hadamard, 1) @ apply_to_qubit(Hadamard, 0) @ init_state
    measure_xx = measure(transform_to_Z_basis("XX") @ init_state, ["00", "01", "10", "11"], number_shots)
    
    # expected value of ZI = (#00 + #01 - #10 - #11)/ number of shots
    exp_vals = np.zeros(4) # do not include the expectation value of II
    measures = np.array([measure_iz, measure_zi, measure_zz, measure_xx])
    constants = np.array([B, C, D + lmb*Hz, lmb*Hx])
    for index in range(len(exp_vals)):
        exp_vals[index] = measures[index]["00"] + measures[index]["01"] - measures[index]["10"] - measures[index]["11"]
    exp_val = A + np.sum(constants * exp_vals) / number_shots
    return exp_val

#VQE(4)

# Part f)
epsilon = 0.5
H_0 = epsilon * np.diag([-2, -1, 0, 1, 2])

H_1 = np.zeros((5, 5))
H_1[2, 0] = np.sqrt(6)
H_1[3, 1] = 3
H_1[4, 2] = H_1[2, 0]
H_1 = H_1 + H_1.T

eigvals = find_eigenvalues_scipy(5)
fig, axs = plt.subplots(1, 1, figsize=(10, 10))
for i in range(5):
    axs.plot(lambda_range, eigvals[:,i], label=f'$E_{i+1}$')#color = '#4c72b0')
plt.show()

# Part g)

def get_energy(angles, lmb, number_shots):
    E, V, W = 1, lmb, 0
    term1 = W * (Id ^ Id)
    term2 = (W-E) * (Pauli_Z ^ Id) #ZI -> IxI with x being the kroncker product
    term3 = -(W+E) * (Id ^ Pauli_Z) #IZ -> SWAP
    term4 = -W * (Pauli_Z ^ Pauli_Z) # can stay this way
    term5 = np.sqrt(6)/2*V * (Id ^ Pauli_X) # IX -> (HxI)Swap
    term6 = np.sqrt(6)/2*V * (Pauli_X ^ Id) # XI -> HxI
    term7 = np.sqrt(6)/2*V * (Pauli_Z ^ Pauli_X) # ZX -> CNOT10 (IxH)
    term8 = -np.sqrt(6)/2*V * (Pauli_X ^ Pauli_Z) # XZ -> CNOT10 (IxH) SWAP

    H_135= term1+term2+term3+term4+term5+term6+term7+term8
    H_24=3*W*Id-E*Pauli_Z+3*V*Pauli_X
    
    init_state = prepare_state(angles)

    pauli_strings = ["ZI", "IZ", "ZZ", "IX", "XI", "ZX", "XZ"]
    measurements = []
    for p_str in pauli_strings:
        qubit = transform_to_Z_basis(p_str) @ init_state
        measurements.append(measure(qubit, ["00", "01", "10", "11"], number_shots))
    
    # expected value of ZI = (#00 + #01 - #10 - #11)/ number of shots
    exp_vals = np.zeros(len(measurements)) # do not include the expectation value of II
    constants = np.array([(W-E), -(W+E), -W, np.sqrt(6)/2*V, np.sqrt(6)/2*V, np.sqrt(6)/2*V, -np.sqrt(6)/2*V])
    for index in range(len(exp_vals)):
        exp_vals[index] = measurements[index]["00"] + measurements[index]["01"] - measurements[index]["10"] - measurements[index]["11"]
    exp_val = W + np.sum(constants * exp_vals) / number_shots
    return exp_val

VQE(5)