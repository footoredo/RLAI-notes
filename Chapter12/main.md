## Chapter 12: Eligibility Traces

### 12.1 The $\lambda$-return

$$
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{v}(S_{t+n},\mathbf{w}_{t+n-1})
$$

- **$\lambda$-return**:
  $$
  \begin{aligned}
  G_t^\lambda&\doteq(1-\lambda)\sum_{n=1}^\infty\lambda^{n-1}G_{t:t+n}\\
  &\doteq(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n}+\lambda^{T-t-1}G_t
  \end{aligned}
  $$

- Offline $\lambda$-return algorithm 
  $$
  \mathbf{w}_{t+1}\doteq\mathbf{w}_t+\alpha[G_t^\lambda-\hat{v}(S_t,\mathbf{w}_t)]\nabla\hat{v}(S_t,\mathbf{w}_t)
  $$


### 12.2 TD($\lambda$)

- **Eligibility trace**

$$
\begin{aligned}
\mathbf{z}_{-1}&\doteq\mathbf{0}\\
\mathbf{z}_t&\doteq\gamma\lambda\mathbf{z}_{t-1}+\nabla\hat{v}(S_t,\mathbf{w}_t)
\end{aligned}
$$

- **TD($\lambda$)**

$$
\delta_t\doteq R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}_t)-\hat{v}(S_t,\mathbf{w}_t)
$$

$$
\mathbf{w}_{t+1}\doteq\mathbf{w}_t+\alpha\delta_t\mathbf{z}_t
$$

- Convergence

$$
\overline{\text{VE}}(\mathbf{w}_\infty)\leq\frac{1-\gamma\lambda}{1-\gamma}\min_{\mathbf{w}}\overline{\text{VE}}(\mathbf{w})
$$

### 12.3 $n$-step Truncated $\lambda$-return Methods

- Truncated $\lambda$-return

$$
G_{t:h}^\lambda\doteq(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_{t:t+n}+\lambda^{h-t-1}G_t
$$

