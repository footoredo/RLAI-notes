## Chapter 6: Temporal-Difference Learning 

### 6.1 TD Prediction 

- Constatnt-$\alpha$ MC

- $$
  V(S_t)\leftarrow V(S_t)+\alpha(G_t-V(S_t))
  $$

- TD(0), or one-step TD

- $$
  V(S_t)\leftarrow V(S_t)+\alpha(R_{t+1}+\gamma V(S_{t+1})-S_t)
  $$


TD(0) based on existing estimates, so it's a *bootstrapping* method.

### 6.2 Advantages of TD Prediction Methods 

- **Compared to DP**: Do not need a model of the environment
- **Compared to MC**: Online, fully incremental
- TD(0) has been proved to converge to $v_\pi(s)$.
- In practice, TD methods usually converge faster than constant-$\alpha$ MC methods on stochastic tasks.