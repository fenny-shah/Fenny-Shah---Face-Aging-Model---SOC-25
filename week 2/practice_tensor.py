import torch
x = torch.empty(1) 
print(x)
x = torch.empty(3) 
print(x)
x = torch.empty(2,3) 
print(x)
x = torch.empty(2,2,3) 
print(x)

x = torch.rand(5, 3)
print(x)
x = torch.zeros(5, 3)
print(x)

print(x.size())
print(x.dtype)
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)
print(x.dtype)
x = torch.tensor([5.5, 3])
print(x.size())
x = torch.tensor([5.5, 3], requires_grad=True)

y = torch.rand(2, 2)
x = torch.rand(2, 2)

z = x + y

z = x - y
z = torch.sub(x, y)

z = x * y
z = torch.mul(x,y)

z = x / y
z = torch.div(x,y)
x = torch.rand(5,3)
print(x)
print(x[:, 0]) 
print(x[1, :]) 
print(x[1,1]) 

print(x[1,1].item())

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  
print(x.size(), y.size(), z.size())
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))
a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

a += 1
print(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")        
    y = torch.ones_like(x, device=device)  
    x = x.to(device)                       
    z = x + y
    z.to("cpu")       
