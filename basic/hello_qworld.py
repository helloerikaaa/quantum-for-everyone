import pennylane as qml
# hybrid numpy with classical function but with quantum ones as well
from pennylane import numpy as np

# A quantum device is any computational object that can apply quantum ops
# and return a measurement value

# qml.device(name, wires)
# name = name of the device to be loaded
# wires = number of subsystems to initialize the device with
dev1 = qml.device('default.qubit', wires=1)

# parmeters to be used in the circuit
p = [0.54, 0.12]

# ==================== QNode Construction ====================== #
# A QNode is an abstract encapsulation of a quantum function, described by a quantum circuit
# QNodes are bounded to a particular quantum devivde, used to evaluate expectation and variance
# of the circuit.

# Characteristics of a quantum function
# - A quantum function must contain quantum operations, one per line, in the order in which they are to be applied
# - A quantum function must return a single or a tuple of measured observables

# the parameters of the circuit can be a tuple, a list or an array
# uses the individual elements for gate parameters.

# the decorator qnode converts the function into a QNode, this QNode runs on device1


@qml.qnode(dev1)
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))


# Multiple positional arguments
# Since quantum circuit functions are a restricted subset of python functions,
# can also make use of multiple positional arguments and keywords arguments


@qml.qnode(dev1)
def circuit2(phil1, phil2):
    qml.RX(phil1, wires=0)
    qml.RY(phil2, wires=0)
    return qml.expval(qml.PauliZ(0))


print(f'Result from the qNode with array params is {circuit(p)}')

print(f'Result from the qNode with multiple params is {circuit2(0.54, 0.12)}')

# ========================= Gradient Calculation ====================== #

# qml.grad(qnode, argnum)
# this function returns a function representing the derivative of the qnode with respect to the argnum

dcircuit = qml.grad(circuit, argnum=0)
print(f'Gradient of the qNode with array params is {dcircuit(p)}')

dcircuit2 = qml.grad(circuit2, argnum=[0, 1])
print(f'Gradient of the qNode with multiple params is {dcircuit2(0.54, 0.12)}')


# ======================== Optimization ================================ #
# Optimize the two parameters of the circuit so the qubit originally in state 0
# is rotated to be in state 1.

# Cost function
# This function is necessary because when minimizing the cost function,
# the optimizer will determine the values of the circuit parameters
# that produce the desired outcome.

def cost(x):
    return circuit(x)


# Begin optimization with small parameters
init_parameters = np.array([0.011, 0.012])
# the output should be close to 1
print(cost(init_parameters))

# Using the gradient descent optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 100
params = init_parameters

for i in range(steps):
    # update the circuit parameters
    params = opt.step(cost, params)

    if (i + 1) % 5 == 0:
        print(f'Cost after step {i + 1} is {cost(params)}')

print(f'Optimized rotation angles {params}')
