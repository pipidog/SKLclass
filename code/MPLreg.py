from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import timeit

# Parameters ========================
# input data (y=x**x_power+x_shift+noise)
dataset_size=1000
x_power=5          
x_shift=1.0        
noise_std=0.05     # noise standard deviation
steps=1000
# generate data ======================
x_data = np.linspace(-1, 1, dataset_size)[:, np.newaxis] # dataset_size x 1
noise = np.random.normal(0, noise_std, x_data.shape)
y_data = x_data**x_power - x_shift + noise

# Run NN =============================
reg=MLPRegressor(max_iter=steps,tol=1e-6)  # define neural network
reg.fit(x_data,y_data)                     # run neural network

# show results =======================
y_pred=reg.predict(x_data)
loss=reg.loss_

# plot prediction and real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.cla()
lines = ax.plot(x_data, y_pred, 'r-', lw=5)
ax.scatter(x_data, y_data)
plt.title('loss={0}'.format(loss))
plt.show()
