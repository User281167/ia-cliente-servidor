## [Implementación del algoritmo #2 Rennala SGD](https://arxiv.org/pdf/2501.16168)

```
Input:
    initial point x^0 in R^d
    learning rate gamma > 0
    server accumulation size B

Workers start computing stochastic gradients at x^0

for k = 0 to K-1 do
    g_k = 0
    b = 0

    while b < B do
        Receive gradient grad(x^(k-delta), xi)
        from worker i

        if delta == 0 then
            g_k = g_k + grad(x^(k-delta), xi)
            b = b + 1
        end if

        Send current model x^k back to worker i
        Worker i starts computing a new gradient
    end while

    x^(k+1) = x^k - gamma * (g_k / B)
end for
```
