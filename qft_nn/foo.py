import torch
from torch.autograd.functional import jacobian

model = torch.nn.Linear(5, 2, dtype=float)
foo = torch.tensor([4, 3, 2, 1, 8], dtype=None)

baz = jacobian(model, foo)


def hello_world() -> str:
    return "Hello World"


if __name__ == "__main__":
    print(baz)
    print(model(foo))
