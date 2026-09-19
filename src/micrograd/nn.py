import random
from .engine import Value

class Neuron:

    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1,1))
        
    def __call__(self,x):
        trf = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        out = trf.tanh()
        return out
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, nin_each, nn_layer):
        self.neurons = [Neuron(nin_each) for _ in range(nn_layer)]
    def __call__(self, x):
        out = [node(x) for node in self.neurons]
        return out
    def parameters(self):
        p = [p for n in self.neurons for p in n.parameters()]
        return p
    
class MLP:
      
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x[0] if len(x)==1 else x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]