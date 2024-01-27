import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hiding INFO and WARNING debugging information

from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt
from graphviz import Digraph

#All object in computational graph is instance of class Value
class Value:
    def __init__(self, data, _children=[], _op='', label=''):
        # data is a numpy array
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
      
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._forward = lambda: None
        self._prev = _children
        self._op = _op
    
        self.label = label
    
    # Operator + 
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data + other.data, [self, other], '+')

        # In backward process, we have to notice that we can add to numpy with different shape,
        # So before using normal formula, with operand having smaller ndim, we have to use sum on axises that being extended or have value 1
        def _backward():
            if(self.data.shape == out.data.shape):
                self.grad = self.grad + (1.0 * out.grad)
                i = 0
                tup = ()
                prvshape = other.data.shape
                if(other.data.ndim < self.data.ndim):
                    st = other.data.ndim
                    ed = self.data.ndim
                    for i in range (st,ed):
                        other.data = np.expand_dims(other.data, axis = 0)
                
                for i in range(0, other.data.ndim):
                    if( self.data.shape[i] != other.data.shape[i]):
                        tup = tup + (i,)
                
                
                other.data = other.data.reshape(prvshape)
                other.grad = other.grad +  (1.0 * out.grad.sum(axis=tup, keepdims=1)).reshape(other.data.shape)
                
            else:
                other.grad = other.grad + (1.0 * out.grad)
                i = 0
                tup = ()
                prvshape = self.data.shape
                if(self.data.ndim < other.data.ndim):
                    st = self.data.ndim
                    ed = other.data.ndim
                    for i in range (st,ed):
                        self.data = np.expand_dims(self.data, axis = 0)
                
                for i in range(0, self.data.ndim):
                    if( self.data.shape[i] != other.data.shape[i]):
                        tup = tup + (i,)
                        total = total * out.data.shape[i]
                
                self.data = self.data.reshape(prvshape)
                self.grad = self.grad + (1.0 * out.grad.sum(axis=tup, keepdims=1)).reshape(self.data.shape)
            
        def _forward():
            out.data = self.data + other.data

        out._backward = _backward
        out._forward = _forward
        return out

    #Operator '*' , pretty like '+'
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        
        out = Value(self.data * other.data, [self, other], '*')

        def _backward():
            # self.grad = self.grad + (other.data * out.grad)
            # other.grad = other.grad + (self.data * out.grad)
            
            if(self.data.shape == out.data.shape):
                self.grad = self.grad + (other.data * out.grad)
                i = 0
                tup = ()
                prvshape = other.data.shape
                if(other.data.ndim < self.data.ndim):
                    st = other.data.ndim
                    ed = self.data.ndim
                    for i in range (st,ed):
                        other.data = np.expand_dims(other.data, axis = 0)
                
                for i in range(0, other.data.ndim):
                    if( self.data.shape[i] != other.data.shape[i]):
                        tup = tup + (i,)
                
                other.data = other.data.reshape(prvshape)
                other.grad = other.grad +  ( (self.data * out.grad).sum(axis=tup, keepdims=1)).reshape(other.data.shape)
                
            else:
                other.grad = other.grad + (self.data * out.grad)
                i = 0
                tup = ()
                prvshape = self.data.shape
                if(self.data.ndim < other.data.ndim):
                    st = self.data.ndim
                    ed = other.data.ndim
                    for i in range (st,ed):
                        self.data = np.expand_dims(self.data, axis = 0)
                
                for i in range(0, self.data.ndim):
                    if( self.data.shape[i] != other.data.shape[i]):
                        tup = tup + (i,)
                        total = total * out.data.shape[i]
                
                self.data = self.data.reshape(prvshape)
                self.grad = self.grad + ((other.data * out.grad).sum(axis=tup, keepdims=1)).reshape(self.data.shape)
            
        def _forward():
            out.data = self.data * other.data

        out._backward = _backward
        out._forward = _forward
        return out

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            # If the exponent is a scalar, perform element-wise power operation
            out_data = np.power(self.data, other)
            out = Value(out_data, [self], f'**{other}')

            def _backward():
                # Calculate the gradients using the chain rule and update the gradients of the operand
                self.grad = self.grad + ((other * np.power(self.data, other - 1)) * out.grad)
                
            def _forward():
                out.data = np.power(self.data, other)

            out._backward = _backward
            out._forward = _forward
            return out
        elif isinstance(other, Value):
      # If the exponent is a Value instance, perform element-wise power operation
            out_data = np.power(self.data, other.data)
            out = Value(out_data, (self, other), f'**')

            def _backward():
                # Calculate the gradients using the chain rule and update the gradients of the operands
                self.grad = self.grad + ((other.data * np.power(self.data * other.data - 1)) * out.grad)
                other.grad = other.grad + (np.log(self.data) * out.grad)

            def _forward():
                out.data = np.power(self.data, other.data)

            out._backward = _backward
            out._forward = _forward
            return out
        else:
            raise TypeError("Unsupported operand type(s) for **: 'Value' and '{}'".format(type(other).__name__))
  
    def __radd__(self, other):
        # Perform right addition by the Value instance
        return self + other


    def __rmul__(self, other):
        # Perform right multiplication by the Value instance
        return self * other


    def __truediv__(self, other):
        # Perform true division by the Value instance
        return self * other**-1.0


    def __neg__(self):
        # Perform negation of the Value instance
        return self * -1


    def __sub__(self, other):
        # Perform subtraction of a Value instance
        return self + (-other)

    def exp(self):
        # Compute the element-wise exponential of the Value instance
        x = self.data
        out = Value(np.exp(x), [self], 'exp')
        
        def _backward():
            # Calculate the gradients using the chain rule
            self.grad = self.grad + (out.data * out.grad)
            
        def _forward():
            out.data = np.exp(self.data)

        # Assign the backward function to the new Value instance
        out._backward = _backward
        out._forward = _forward
        return out

    def tanh(self):
        # Compute the element-wise hyperbolic tangent of the Value instance
        x = self.data
        t = np.tanh(x)
        out = Value(t, [self], 'tanh')

        def _backward():
            # Calculate the gradients using the chain rule
            self.grad = self.grad + ((1 - t**2) * out.grad)

        def _forward():
            out.data = np.tanh(self.data)

        # Assign the backward function to the new Value instance
        out._backward = _backward
        out._forward = _forward
        return out
    
    def relu(self):
        t = np.maximum(0,self.data)
        out = Value(t, [self], 'relu')
        
        def _backward():
            self.grad = (self.data > 0).astype(float) * out.grad

        def _forward():
            out.data = np.maximum(0,self.data)
            
        out._backward = _backward
        out._forward = _forward
        return out
    
    def sum(self, Axis = None):
        out = Value(self.data.sum(Axis), [self], 'sum')
        if( Axis == None):
            Axis = ()
            for i in range (0, self.data.ndim):
                Axis = Axis + (i,)
        elif(type(Axis) is not tuple):
            Axis = (Axis,)
        def _backward():
            temp = np.expand_dims(out.grad, axis = Axis)
            for i in Axis:
                temp = np.repeat( temp, self.data.shape[i], axis = i )
            
            self.grad = self.grad + temp
            
        def _forward():
            out.data = self.data.sum(Axis)
            
        out._backward = _backward
        out._forward = _forward
        return out
    
    def log(self):
        out = Value( np.log(self.data), [self], 'ln')
        
        def _backward():
            self.grad = self.grad + 1.0 / self.data * out.grad
            
        def _forward():
            out.data = np.log(self.data)
            
        out._backward = _backward
        out._forward = _forward
        return out
    
    def max(self):
        return self.data.max()
    
    def dot(self, other):
        out = Value( np.dot( self.data, other.data), [self, other], 'dot')
        
        def _backward():
            self.grad = self.grad + np.dot(out.grad, other.data.T)
            other.grad = other.grad + np.dot(self.data.T, out.grad)
            
        def _forward():
            out.data = np.dot( self.data, other.data)
            
        out._backward = _backward
        out._forward = _forward
        return out

    def backwardFrom(self):
        # Perform backpropagation to compute gradients
        topo = []
        visited = set()

        def build_topo(v):
            visited.add(v)
            for child in v._prev:
                if child not in visited:
                    build_topo(child)
            topo.append(v)

        # Build the topological order of nodes
        build_topo(self)

        # Set the gradient of the output to ones (assuming scalar loss)
        self.grad = np.ones_like(self.data)

        # Perform backward pass through the nodes in reverse topological order
        for node in reversed(topo):
            node._backward()
            
    def forwardTo(self):
        visited = set()
        
        def dfs(v):
            visited.add(v)
            for child in v._prev:
                if child not in visited:
                    dfs(child)
            v._forward()
            
        dfs(self)
        
    def clearGrad(self):
        visited = set()
        
        def dfs(v):
            visited.add(v)
            for child in v._prev:
                if child not in visited:
                    dfs(child)
            v.grad = np.zeros_like(v.grad)
            
        dfs(self)
                
    def __repr__(self):
        # String representation of the Value instance
        return f"{self.data}"

def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)

            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph( 'Computational_Graph' , graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    # Iterate over the nodes
    for n in nodes:
        uid = str(id(n))
        # Convert the data array to a string representation
        data_str = np.array2string(n.data, precision=4, separator=',', suppress_small=True)
        # Convert the gradient array to a string representation
        grad_str = np.array2string(n.grad, precision=4, separator=',', suppress_small=True)
        # Create the label for the node using the node's label, data, and gradient strings
        if(n._op):
            label = "{ %s | %s | data %s | grad %s }" % (n._op, n.label, data_str, grad_str)
        else:
            label = "{ %s | data %s | grad %s }" % (n.label, data_str, grad_str)
        # Add a node to the graph with a unique ID, label, and shape 'record'
        dot.node(name=uid, label=label, shape='record')
        

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)))

    return dot

##########################################################
##########################################################

a = Value([1,2,3], label='a')
b = Value([2,3,4], label='b')
c = a * b
d = c + a
e = d.sum()
c.label = 'c'
d.label = 'd'
e.label = 'e'
e.backwardFrom()
dot = draw_dot(e)
dot.render(directory='', view=False)
