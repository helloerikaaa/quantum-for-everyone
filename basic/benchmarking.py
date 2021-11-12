import timeit

import pennylane as qml
import pennylane.numpy as np

# ===================== Benchmarking ================= #

dev = qml.device('default.qubit', wires=4)


# The StronglyEntanglingLayers helps to make a more complicated qNode
@qml.qnode(dev, diff_method='parameter-shift', mutable=False)
def circuit(params):
    qml.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


# Initialize params of the circuit. Containing 180 parameters
param_shape = qml.templates.StronglyEntanglingLayers.shape(
    n_wires=4, n_layers=15)
params = np.random.normal(scale=0.1, size=param_shape)

print(params.size)
print(circuit(params))


# Benchmarking time of performing a forward pass of the circuit
reps = 3
num = 10
times = timeit.repeat("circuit(params)", globals=globals(),
                      number=num, repeat=reps)
forward_time = min(times) / num

print(f'Forward pass (best of {reps}): {forward_time} sec per loop')


# Benchmarking time of computation of full gradient vector
grad_fn = qml.grad(circuit)
circuit.qtape = None

times = timeit.repeat("grad_fn(params)", globals=globals(),
                      number=num, repeat=reps)
backward_time = min(times) / num

print(f"Gradient computation (best of {reps}): {backward_time} sec per loop")
