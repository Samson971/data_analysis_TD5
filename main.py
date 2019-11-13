#%%
import os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
%matplotlib inline

random_values = np.random.randint( 0,100, (200, 1))
X, y = random_values, 2*random_values + np.random.normal(0, 30, size=(200,1))  # y = 2*x + epsilon with epsilon ~ N(0,1)
plt.scatter(x=X,y=y)
plt.show()

#%%
X.reshape((200,))

#%%
pd.DataFrame({"X":  X.reshape((200,)), 'y': y.reshape((200,))})

# %%
np.random.randint(-2,2, (1,))

# %%
weight = np.random.randint(-2,2, (X.shape[1],))
b= np.random.randint(0,10)
w =np.random.randint(-2,2,(1,1))
y

# %%
print(y.size)
print(b)
print(w)

# %%
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Neural_Network_1neuron:
    
    def __init__(self, X, y, nb_epochs=100, fixed_bias=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20)
        self.X_train, self.X_test = self.scale_values(self.X_train, self.X_test)
        self.y_train, self.y_test = \
            np.reshape(self.y_train, (len(self.y_train),1)),\
            np.reshape(self.y_test, (len(self.y_test),1))
        
        self.weights = [] * X.shape[1]  # as many weights as features, here 1
        self.weights = self.init_weights(X)
        self.bias, self.fixed_bias    = 0, fixed_bias
        self.training_predictions, self.mse = 0, 0
        self.learning_rate = 0.01
        self.nb_epochs = nb_epochs
        self.weights_update, self.bias_update = 0, 0
        self.records = pd.DataFrame([[self.weights, self.bias,0]], columns=['weights', 'bias', 'mse'])
    
    def scale_values(self, X_train, X_test):
        scale   = StandardScaler()
        X_train = scale.fit_transform(X_train)
        X_test  = scale.transform(X_test)
        return X_train, X_test
        
    def init_weights(self, X):
        nb_of_features_input = X.shape[1]
        return np.random.randint(-2,2, (nb_of_features_input, 1))
    
    def activation(self):
        return x
    def derivate_activation(self):
        return 1
    
    def forward_pass(self):
        self.training_predictions = np.dot(self.X_train, self.weights) + self.bias
    
    def compute_mse(self):
        self.mse = 1/(2*len(self.X_train)) * sum( ( self.training_predictions - self.y_train)**2 )
    
    def backpropagation(self):
        # derivative of error by weights so to update them along with bias
        # dE/dw = dE/da * da/dz * dz/dw
        # w = w - n*dE/dw
        # E = mse = (1/2n)* sum(( predictions(==activations) - target)**2) ON ALL TRAINING EXAMPLES
        
        self.dE_da = self.training_predictions - self.y_train #for all training examples
        
        self.da_dz = self.derivate_activation() # derivate of activation
        
        self.dz_dw = self.X_train     # z = W*X + bias => dz_dw = X    
        
        self.weights_update, self.bias_update = \
            ( 1 / self.X_train.shape[0] ) * np.reshape( sum( self.dE_da * self.da_dz * self.dz_dw), (self.X_train.shape[1], 1) ),\
            ( 1 / self.X_train.shape[0] ) * sum( self.dE_da * self.da_dz ),  # weights updates, bias update
    
    def update(self):
        self.weights = self.weights - self.learning_rate * self.weights_update
        self.bias    = self.bias    - self.learning_rate * self.bias_update if not self.fixed_bias else self.bias
        
    def predict(self):
        return np.dot(self.X_test, self.weights) + self.bias
    
    def run(self):
        for i in range(1, self.nb_epochs):
            self.forward_pass()
            self.compute_mse()
            self.backpropagation()
            self.update()
            self.records.loc[i] = [self.weights, self.bias, self.mse]  
        return self.records

# %%
unReseauDeNeurone = Neural_Network_1neuron(X, y, nb_epochs=2000)
records= unReseauDeNeurone.run()
plt.plot(records['weights'])
plt.plot(records['bias'])


# %%
from sklearn.preprocessing import StandardScaler
plt.scatter(x=unReseauDeNeurone.X_test, y= unReseauDeNeurone.y_test, color='green')
x_ = np.linspace(-2, 2, 100).reshape((100,1))
y_ = float(unReseauDeNeurone.weights)*x_ + float(unReseauDeNeurone.bias)
plt.plot(x_, y_, color='red')


#%%
%matplotlib inline

import matplotlib.animation as animation

fig, ax = plt.subplots()
# Initial plot
x_ = np.linspace(-2, 2, 100).reshape((100,1))
y_ = float(records.loc[0, "weights"])*x_ + float(records.loc[0, "bias"])

line, = ax.plot(x_, y_, label="Fit from the perceptron")

plt.rcParams["figure.figsize"] = (8,6)
plt.ylabel("y")
plt.xlabel("X")
plt.scatter(x=unReseauDeNeurone.X_train, y= unReseauDeNeurone.y_train, color='red', label="Training data")
plt.scatter(x=unReseauDeNeurone.X_test, y= unReseauDeNeurone.y_test, color='green', label="Test data")
plt.xlim(-2, 2)
plt.legend()
plt.title("Linear regression training fit using a single neuron | perceptron")

def animate(i):
    line.set_label("Fit from the perceptron : epoch {}".format(i))
    plt.legend()
    x_ = np.linspace(-2, 2, 100).reshape((100,1))
    line.set_xdata(x_)  # update the data
    line.set_ydata( float(records.loc[i, "weights"])*x_ + float(records.loc[i, "bias"]))# update the data
    return line,


ani = animation.FuncAnimation(fig, animate, frames=np.arange(1, len(records)), interval=100)
plt.show()

# %%
