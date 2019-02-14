import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def linear_equation(m, x, c):
    return m * x + c


def plot(x, y, m, c):
    plt.scatter(x, y)
    plt.xlabel("Experience")
    plt.ylabel("Salary")
    plt.plot(x, linear_equation(m, x, c), label="fit for line y={0}x + {1}".format(m, c))
    plt.legend()
    plt.show()


def mse(c, m, x, y):
    total_error = 0
    for i in range(0, len(x)):
        total_error += (y[i] - (m * x[i] + c)) ** 2
    return total_error / float(len(x))


def sse(x, y, m, c):
    total = 0
    for i in range(0, len(x)):
        total += (y[i] - (m * x[i] + c)) ** 2

    return total


def sst(x, y):
    total = 0
    y_mean = np.mean(y)
    for i in range(0, len(x)):
        total += (y[i] - y_mean) ** 2

    return total


def r_square(x, y, m, c):
    return 1 - sse(x, y, m, c) / sst(x, y)


def r_square_adjusted(x, y, m, c):
    n = len(x)
    p = 1
    return 1 - (1 - r_square(x, y, m, c)) * ((n - 1) / (n - p - 1))


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


def calculate_m_b_with_gradient_descent(x, y, starting_m, starting_c, learning_rate, iterations, is_normalized):
    c_best = starting_c
    m_best = starting_m
    err_best = mse(c_best, m_best, x, y)
    r_sq = r_square(x, y, m_best, c_best)
    r_sq_ad = r_square_adjusted(x, y, m_best, c_best)

    c_arr = np.array([c_best])
    m_arr = np.array([m_best])
    error_arr = np.array([err_best])
    r_sq_arr = np.array([r_sq])
    r_sq_ad_arr = np.array([r_sq_ad])
    iter_arr = np.array([0])

    fig = plt.figure(figsize=(20, 6))

    ax = fig.add_subplot(221)
    grad, = ax.plot(c_arr, m_arr)
    ax.set(xlabel="c", ylabel="m")
    if is_normalized:
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.1)
    else:
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 2000)

    bx = fig.add_subplot(222)
    bx.scatter(x, y)
    lr, = bx.plot(x, linear_equation(m_best, x, c_best))

    cx = fig.add_subplot(223)
    ep, = cx.plot(iter_arr, error_arr)
    cx.set(xlabel="iterations", ylabel="error")
    cx.set_xlim(0, iterations)
    cx.set_ylim(0, err_best)

    dx = fig.add_subplot(224)
    r2, = dx.plot(iter_arr, r_sq_arr, label="r2")
    r2_ad, = dx.plot(iter_arr, r_sq_ad_arr, label="r2_ad")
    dx.set(xlabel="iterations", ylabel="goodness of fit (r-squared)")
    dx.set_xlim(0, iterations)
    dx.set_ylim(-1, 1)
    dx.legend()

    plt.ion()
    plt.show()

    for i in range(iterations):
        c_best, m_best = step_gradient(c_best, m_best, x, y, learning_rate)
        err_best = mse(c_best, m_best, x, y)
        r_sq = r_square(x, y, m_best, c_best)
        r_sq_ad = r_square_adjusted(x, y, m_best, c_best)
        print(m_best, c_best, err_best, r_sq, r_sq_ad)

        c_arr = np.append(c_arr, c_best)
        m_arr = np.append(m_arr, m_best)
        error_arr = np.append(error_arr, err_best)
        r_sq_arr = np.append(r_sq_arr, r_sq)
        r_sq_ad_arr = np.append(r_sq_ad_arr, r_sq_ad)
        iter_arr = np.append(iter_arr, i + 1)

        grad.set_data(c_arr, m_arr)
        lr.set_data(x, linear_equation(m_best, x, c_best))
        ep.set_data(iter_arr, error_arr)
        r2.set_data(iter_arr, r_sq_arr)
        r2_ad.set_data(iter_arr, r_sq_ad_arr)
        # print(np.array(range(i+2)).shape, error_arr.shape, c_arr.shape, m_arr.shape)

        plt.pause(0.0000000000000000001)

    plt.show(block=True)

    return m_best, c_best, err_best


def run(initial_c, initial_m, x, y, learning_rate, iterations, is_normalized):
    # plot(x, y, initial_m, initial_c)
    error = mse(initial_c, initial_m, x, y)
    print("Error at b = {0}, m = {1}, error = {2}".format(initial_c, initial_m, error))

    m, c, error = calculate_m_b_with_gradient_descent(x, y, initial_m, initial_c, learning_rate, iterations, is_normalized)

    # Model Evaluation - Coefficient of Determination (R-squared)
    # i.e. goodness of fit of our regression model.
    # https://towardsdatascience.com/coefficient-of-determination-r-squared-explained-db32700d924e
    # R^2 = 1 - SSE / SST
    # SSE = the sum of squared errors of our regression model
    # SST = the sum of squared errors of our baseline model (which is worst model).
    print("r square = {0} and r_squared_adjusted = {1}".format(r_square(x, y, m, c), r_square_adjusted(x, y, m, c)))


def run_with_normalization(data, initial_c, initial_m, learning_rate, iterations):
    x = data.experience

    y = data.salary
    y_min = np.min(y)
    y_max = np.max(y)

    # implementing min-max scaling
    y = data['salary'].apply(lambda salary: ((salary - y_min) / (y_max - y_min)))

    run(initial_c, initial_m, x, y, learning_rate, iterations, True)


class OutlierRemover:
    def __init__(self, df):
        # Apply replace() on each column of the dataframe
        df = df.apply(self.replace, axis=1)

        # remove the rows containing any outlier:
        df = df[~df.apply(self.is_outlier).any(axis=1)]

        self.df = df

    def is_outlier(self, x):
        # a number "a" from the vector "x" is an outlier if
        # a > median(x)+1.5*iqr(x) or a < median-1.5*iqr(x)
        # iqr: interquantile range = third interquantile - first interquantile
        # The function return a boolean vector: True if the element is an outlier. False, otherwise.
        return np.abs(x - x.median()) > 1.5 * (x.quantile(.75) - x.quantile(0.25))

    def replace(self, x):
        # Replace the upper outlier(s) with the 95th percentile and the lower one(s) with the 5th percentile
        out = x[self.is_outlier(x)]
        return x.replace(to_replace=[out.min(), out.max()], value=[np.percentile(x, 5), np.percentile(x, 95)])

    def get(self):
        return self.df


def distribution_plot(data):
    plt.figure(figsize=(10, 8))
    plt.subplot(221)
    plt.xlim(data.experience.min(), data.experience.max() * 1.1)

    # Plot kernel distribution
    ax = data.experience.plot(kind='kde')

    plt.subplot(223)
    plt.xlim(data.experience.min(), data.experience.max() * 1.1)
    sns.boxplot(x=data.experience)

    plt.subplot(222)
    plt.xlim(data.salary.min(), data.salary.max() * 1.1)

    # Plot kernel distribution
    bx = data.salary.plot(kind='kde')

    plt.subplot(224)
    plt.xlim(data.salary.min(), data.salary.max() * 1.1)
    sns.boxplot(x=data.salary)

    plt.show()


if __name__ == '__main__':
    plt.interactive(True)

    initial_c = initial_m = 0
    data = pd.read_csv('./salaries_clean.csv', encoding='utf-8')

    # Data Cleaning
    data = data[['total_experience_years', 'annual_base_pay']]
    data.columns = ['experience', 'salary']

    # ## Removing infinities and na form data
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # ## This data has outliers
    # plt.scatter(data.experience, data.salary)
    # plt.show()

    # ## To visualize outliers, box plot is better
    # distribution_plot(data)

    print("before removing outliers: ")
    print(data.describe())

    # ## Removing outliers
    # ## For each series in the dataframe, we could use between and quantile (or percentile) to remove outliers.
    data = OutlierRemover(data).get()

    # distribution_plot(data)

    print("after removing outliers: ")
    print(data.describe())

    plt.scatter(data.experience, data.salary)
    plt.show(block=True)

    data = data.reset_index()

    x = data.experience
    y = data.salary

    learning_rate = 0.01
    iterations = 1000

    # print(data.describe())
    # run(initial_c, initial_m, x, y, learning_rate, iterations, False)

    # Why, How and When to Scale (or normalize) Features
    # https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e
    run_with_normalization(data, initial_c, initial_m, learning_rate, iterations)
