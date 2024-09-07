import mlx.core as mx

print("Default device = ", mx.default_device())
print("Default stream = ", mx.default_stream(mx.default_device()))

# vector addition
a = mx.random.normal((100,))
b = mx.random.normal((100,))

c = mx.add(a, b, stream=mx.gpu)
print(c)

# matrix multiplication
p = mx.random.uniform(shape=(4096, 512))
q = mx.random.uniform(shape=(512, 4))

r = mx.matmul(p, q, stream=mx.gpu)
print(r)

