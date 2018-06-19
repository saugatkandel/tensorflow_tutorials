#Author - Saugat Kandel
# coding: utf-8


from autograd import numpy as np, grad
import matplotlib.pyplot as plt



# Sin
gradfn = grad(np.sin)



arr = np.linspace(-np.pi, np.pi, 100)
fnvals = np.array([np.sin(x) for x in arr])
gradfnvals = np.array([gradfn(x) for x in arr])
cosvals = np.array([np.cos(x) for x in arr])



plt.plot(fnvals, c='red', label='sin')
plt.plot(gradfnvals, c='blue', label='grad', ls = '-.')
plt.plot(cosvals - gradfnvals, c = 'green', label='cos - gradfn', ls = ':')
plt.legend(loc='best')
plt.show()



def f(x):
    return (x-np.pi)**2



gradf = grad(f)



x_guess = 3.0
loss_init = f(x_guess)
print(f'Initial loss is {loss_init}')

step_size = 0.1
lossvals = []
x_guesses = []
for i in range(50):
    lossval = f(x_guess)
    x_guess = x_guess - step_size * gradf(x_guess)
    lossvals.append(lossval)
    x_guesses.append(x_guess)
print(f'Loss: {lossval}   Updated x: {x_guess}')

plt.subplot(1,2,1)
plt.plot(lossvals)
plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(x_guesses, ls = ':', c='green')
plt.hlines(np.pi, xmin=0, xmax = 50, color='red')
plt.show()

