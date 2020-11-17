import numpy as np
import matplotlib.pyplot as plt


class GradientDescent:
    def __init__(self, loss_function, loss_function_gradient, x_range):
        """
        loss_function_gradient: The first derivative of the loss function.
        x_range: Used for plotting the function.
        """
        self.loss_function = loss_function
        self.loss_function_gradient = loss_function_gradient
        self.x_range = x_range

    def get_y_values_for_x(self):
        return [self.loss_function(x) for x in self.x_range]

    def gradient_descent(self, initial_x, learning_rate=0.01, iterations=100):
        """
        Starting from an initial x, apply gradient descent to minimise the loss function.

        Learning rate and iterations should be adjusted depenting on the loss function.
        """
        iteration = 0
        current_x = initial_x

        # Store all the (x, y) points obtained through gradient descent for later plot scatter.
        all_x_obtained = [current_x]
        all_y_obtained = [self.loss_function(current_x)]

        for i in range(iterations):
            iteration += 1
            print("\nIteration {}: \nx is {}.".format(iteration, current_x))

            current_x -= learning_rate * self.loss_function_gradient(current_x)

            all_x_obtained.append(current_x)
            all_y_obtained.append(self.loss_function(current_x))

        return all_x_obtained, all_y_obtained

    def plot_loss_function(self, all_x_obtained, all_y_obtained):
        # plot the function
        plt.plot(self.x_range, self.get_y_values_for_x())
        # scatter the (x, y) points obtained from gradient descent
        plt.scatter(all_x_obtained, all_y_obtained, c="r", marker="x")
        plt.title("f(x) = $x^2$")
        plt.show()


def loss_function(x):
    return x ** 2


def loss_function_gradient(x):
    return 2 * x


x_range = np.linspace(-1, 1, 100)

gradient_descent = GradientDescent(loss_function,
                                   loss_function_gradient,
                                   x_range)
all_x_obtained, all_y_obtained = gradient_descent.gradient_descent(initial_x=-1,
                                                                   learning_rate=0.1)
gradient_descent.plot_loss_function(all_x_obtained, all_y_obtained)
