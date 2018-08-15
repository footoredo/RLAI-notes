### 9.3 Stochastic-gradient and Semi-gradient Methods 

$ \mathbf{w}\doteq(w_1,w_2,\dots,w_d)^\top $ and $\hat{v}(s,\mathbf{w})$ is a **differentiable** function of $\mathbf{w}$ for all $s$.

**SGD**:
$$
\begin{aligned}
\mathbf{w}_{t+1}&\doteq\mathbf{w}_{t}-\frac{1}{2}\alpha\nabla[v_\pi(S_t)-\hat{v}(S_t,\mathbf{w}_t)]^2\\
&=\mathbf{w}_{t}+\alpha[v_\pi(S_t)-\hat{v}(S_t,\mathbf{w}_t)]\nabla\hat{v}(S_t,\mathbf{w}_t)
\end{aligned}
$$
