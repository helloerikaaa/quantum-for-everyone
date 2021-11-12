import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)

# device creation
dev = qml.device('default.qubit', wires=3)


# ===================== Paramater-Shift Rule ============================ #

# Manual parameter-shift rule implementation
@qml.qnode(dev, diff_method='parameter-shift')
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern='ring')

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern='ring')

    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))


# initial parameters
params = np.random.random([6])

print(f'Parameters: {params}')
print(f'Expectation value: {circuit(params)}')
print('Circuit')
print(qml.draw(circuit)(params))


# function to compute the gradient of the ith parameter using the parameter-shift rule
def parameter_shift_term(qnode, params, i):
    shifted = params.copy()
    shifted[i] += np.pi/2
    # forward evaluation
    forward = qnode(shifted)

    shifted[i] -= np.pi
    # backward evaluation
    backward = qnode(shifted)

    return 0.5 * (forward - backward)


# function to compute the gradient of all parameters
def parameter_shift(qnode, params):
    gradients = np.zeros([len(params)])

    for i in range(len(params)):
        gradients[i] = parameter_shift_term(qnode, params, i)

    return gradients


print(parameter_shift(circuit, params))
