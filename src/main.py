from micrograd.engine import Value
from micrograd.nn import MLP

# test for autograd engine
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a)
d += 3 * d + (b - a)
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print("--------------------------------------------------------------")
print("gradient calculation")
print(f'{g.data:.4f}') 
g.backward()
print(f'{a.grad:.4f}') 
print(f'{b.grad:.4f}')

print("--------------------------------------------------------------")
# test for builidng a multi layer perceptron based neural network
n = MLP(2, [2,1])
print("initial weights values")
print(*n.parameters(), sep='\n')
print("--------------------------------------------------------------")

# test for a traning loop
xs = [
  [2.0, 3.0],
  [3.0, -1.0],
  [0.5, 1.0],
  [1.0, 1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
epochs = 5
for epoch in range(epochs):
      
  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()
  
  # update
  for p in n.parameters():
    p.data += -0.1 * p.grad
  print(f"iteration{epoch+1} loss")
  print(loss.data)
# general neuron loss
print("---------------------------")
print("weight of layer1 1st neuron 1st weight")
print(n.layers[0].neurons[0].weights[0])

print("---------------------------")
print("gradient of layer1 1st neuron 1st weight")
print(n.layers[0].neurons[0].weights[0].grad)

print("---------------------------")
print("bias of layer1 1st neuron 1st weight")
print(n.layers[0].neurons[0].bias)
  