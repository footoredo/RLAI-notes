## Chapter 12: Eligibility Traces

### 12.1 The $\lambda$-return

$$
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\cdots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{v}(S_{t+n},\mathbf{w}_{t+n-1})
$$

- **$\lambda$-return**:
  $$
  \begin{aligned}
  G_t^\lambda&\doteq(1-\lambda)\sum_{n=1}^\infty\lambda^{n-1}G_{t:t+n}\\
  &=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n}+\lambda^{T-t-1}G_t
  \end{aligned}
  $$

- Offline $\lambda$-return algorithm 
  $$
  \mathbf{w}_{t+1}\doteq\mathbf{w}_t+\alpha[G_t^\lambda-\hat{v}(S_t,\mathbf{w}_t)]\nabla\hat{v}(S_t,\mathbf{w}_t)
  $$
  

