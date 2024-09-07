import mlx.core as mlx

num_features = 100
num_examples = 1000
num_iterations = 10000
lr = 0.03

# True weights
w_star = mlx.random.normal((num_features, ))
print("w_star shape = ", w_star.shape)

# Input data
X = mlx.random.normal((num_examples, num_features))
print("X shape = ", X.shape)

# Bias
eps = 1e-2 * mlx.random.normal((num_examples,))

# Output
y = X @ w_star + eps
print("y shape = ", y.shape)

print(y)

def loss_fn(w):
    return 0.5 * mlx.mean((X @ w - y) ** 2)

# SGD
grad_fn = mlx.grad(loss_fn)

# Random initialization of wieghts
w = 1e-2 * mlx.random.normal((num_features, ))
print("w shape = ", w.shape)

for _ in range(num_iterations):
    w = w - lr * grad_fn(w)
    mlx.eval(w)

loss = loss_fn(w)
print("loss = ", loss)

error_norm = mlx.sum(mlx.square(w - w_star)).item() ** 0.5

print(
     f"Loss {loss.item():.5f}, |w-w*| = {error_norm:.5f}, "
)
