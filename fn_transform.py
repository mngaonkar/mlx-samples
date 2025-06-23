import mlx.core as mx

def sin(x):
    return mx.sin(x)

dfdx = mx.grad(sin) # get the gradient function

print(f"dfdx(1.0) = {dfdx(mx.array(1.0))}") # d/dx of sin is cos
print(f"dfdx(1.0) = {mx.cos(mx.array(1.0))}")  

def fn(x):
    return 2*x**2 + 5*x + 3

dfdx_fn = mx.grad(fn) # get the gradient function
print(f"dfdx_fn(1.0) = {dfdx_fn(mx.array(1.0))}")