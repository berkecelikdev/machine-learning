import numpy as np

def compute_cost_multi(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2

    total_cost = cost / (2 * m)
    return total_cost

def compute_gradient_multi(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] += error * X[i, j]

        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent_multi(X, y, w_init, b_init, alpha, num_iters):
    w = np.copy(w_init)
    b = b_init
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_multi(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost_multi(X, y, w, b)
        cost_history.append(cost)

    return w, b, cost_history

X_train = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0]
])

y_train = np.array([8.0, 9.0, 18.0])

w_init = np.array([0.0, 0.0])
b_init = 0.0

cost = compute_cost_multi(X_train, y_train, w_init, b_init)
dj_dw, dj_db = compute_gradient_multi(X_train, y_train, w_init, b_init)

print(f"Başlangıç Cost: {cost}")
print(f"dj_dw: {dj_dw}, dj_db: {dj_db}")

alpha = 0.05
iterations = 1000
w_final, b_final, _ = gradient_descent_multi(X_train, y_train, w_init, b_init, alpha, iterations)

print(f"Eğitilmiş w: {np.round(w_final, 2)}")
print(f"Eğitilmiş b: {np.round(b_final, 2)}")

