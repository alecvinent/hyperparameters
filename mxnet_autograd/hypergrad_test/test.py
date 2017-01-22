
from core import grad


def loss(x, y):
    c = x + y
    c = c * c *c

    return c

if __name__ == '__main__':

    loss_grad = grad(loss)
    g = loss_grad(5.0, 3.0)
    print g
