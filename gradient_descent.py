import numpy as np
import matplotlib.pyplot as plt


class GradientDescent:
    def __init__(self, loss_function, loss_function_gradient, x_range):
        """
        loss_function_gradient: The first derivative of the loss function.
        x_range: The interval of x's.
        """
        self.loss_function = loss_function
        self.loss_function_gradient = loss_function_gradient
        self.x_range = x_range

    def get_y_values_for_x(self):
        """
        Return the values for x based on the loss function.
        """
        return [self.loss_function(x) for x in self.x_range]

    def gradient_descent(self, initial_x, learning_rate=0.01, iterations=10000):
        current_x = initial_x

        chosen_x = [current_x]
        chosen_y = [self.loss_function(current_x)]

        for i in range(iterations):
            current_x -= learning_rate * self.loss_function_gradient(current_x)
            chosen_x.append(current_x)
            chosen_y.append(self.loss_function(current_x))

        return chosen_x, chosen_y

    def plot_loss_function(self, chosen_x, chosen_y):
        plt.plot(self.x_range, self.get_y_values_for_x())
        plt.scatter(chosen_x, chosen_y)
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
chosen_x, chosen_y = gradient_descent.gradient_descent(-1)
gradient_descent.plot_loss_function(chosen_x, chosen_y)
