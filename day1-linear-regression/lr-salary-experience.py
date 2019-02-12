import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linear_equation(m, x, c):
    return m * x + c


def plot(x, y, m, c):
    plt.scatter(x, y)
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.plot(x, linear_equation(m, x, c), label="fit for line y={0}x + {1}".format(m, c))
    plt.legend()
    plt.show()


def compute_error(c, m, x, y):
    total_error = 0
    for i in range(0, len(x)):
        total_error += (y[i] - (m * x[i] + c)) ** 2
    return total_error / float(len(x))


def step_gradient(c_current, m_current, x, y, learning_rate):
    c_gradient = 0
    m_gradient = 0
    n = float(len(x))
    for i in range(0, len(x)):
        c_gradient += -(2 / n) * (y[i] - ((m_current * x[i]) + c_current))
        m_gradient += -(2 / n) * x[i] * (y[i] - ((m_current * x[i]) + c_current))
    new_c = c_current - (learning_rate * c_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_c, new_m]


def calculate_m_b_with_gradient_descent(x, y, starting_m, starting_c, learning_rate, iterations):
    c_best = starting_c
    m_best = starting_m
    err_best = compute_error(c_best, m_best, x, y)

    c_arr = np.array([c_best])
    m_arr = np.array([m_best])
    error_arr = np.array([err_best])
    iter_arr = np.array([0])

    fig = plt.figure(figsize=(20, 4))

    ax = fig.add_subplot(131)
    grad, = ax.plot(c_arr, m_arr)
    ax.set(xlabel="c", ylabel="m")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1000)

    bx = fig.add_subplot(132)
    bx.scatter(x, y)
    lr, = bx.plot(x, linear_equation(m_best, x, c_best))

    cx = fig.add_subplot(133)
    ep, = cx.plot(iter_arr, error_arr)
    cx.set(xlabel="iterations", ylabel="error")
    cx.set_xlim(0, iterations)
    cx.set_ylim(0, 100000000)

    plt.ion()
    plt.show()

    for i in range(iterations):
        c_best, m_best = step_gradient(c_best, m_best, x, y, learning_rate)
        err_best = compute_error(c_best, m_best, x, y)
        # print(m_best, c_best, err_best)

        c_arr = np.append(c_arr, c_best)
        m_arr = np.append(m_arr, m_best)
        error_arr = np.append(error_arr, err_best)
        iter_arr = np.append(iter_arr, i + 1)

        grad.set_data(c_arr, m_arr)
        lr.set_data(x, linear_equation(m_best, x, c_best))
        ep.set_data(iter_arr, error_arr)
        # print(np.array(range(i+2)).shape, error_arr.shape, c_arr.shape, m_arr.shape)

        plt.pause(0.0000000000000000001)

    plt.show()

    return m_best, c_best, error_arr


if __name__ == '__main__':
    initial_c = initial_m = 0
    data = pd.read_csv('experience-salary-datasets.csv')

    # plot(data.experience, data.salary, initial_m, initial_c)
    # print(np.array(range(14)))
    error = compute_error(initial_c, initial_m, data.experience, data.salary)
    print("Error at b = {0}, m = {1}, error = {2}".format(initial_c, initial_m, error))

    learning_rate = 0.00001
    iterations = 1000
    m, c, error_array = calculate_m_b_with_gradient_descent(data.experience, data.salary, initial_m, initial_c, learning_rate, iterations)

    plt.plot(error_array)
    plt.show()
