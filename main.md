## Chapter 9: On-policy Prediction with Approximation

$$
\hat{v}(s,\mathbf{w})\approx v_\pi(s)
$$

### 9.1 Value-function Approximation

- **Update target**: $s\mapsto u$

  - Monte Carlo: $S_t\mapsto G_t$
  - TD(0): $S_t\mapsto R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}_t)$
  - DP: $s\mapsto\mathbb{E}_\pi[R_{t+1}+\gamma\hat{v}(S_{t+1}, \mathbf{w}_t)\mid S_t=s]$

  To mimic the input-output ($s$,$u$): **supervised learning**.

  When output is a number: **function approximation**.

- Methods that cannot easily handle such **non-stationarity** are less suitable for
  reinforcement learning.

### 9.2 The Prediction Objective ($\overline{\text{VE}}$)

- **State distribution** (How much we care about the error is each state): $\mu(s)\geq0, \sum_s \mu(s)=1$.

- **Mean Squared Value Error**:
  $$
  \overline{\text{VE}}(\mathbf{w})\doteq\sum_{s\in\mathcal{S}}\mu(s)[v_\pi(s)-\hat{v}(s,\mathbf{w})]^2
  $$

  - The root $\overline{\text{VE}}$ is often used in plots.

  - $\mu(s)$ is often chosen to be the fraction of time spent in $s$. **(On-policy distribution)**

    - In a continuing task, the on-policy distribution is the stationary distribution under policy $\pi$.

    - In a episodic task, first we denote the probability that an episode starts at $s$ as $h(s)$, and the average time spent on state $s$ in an episode as $\eta(s)$.
      $$
      \eta(s)=h(s)+\sum_{\overline{s}}\eta(\overline{s})\sum_{a}\pi(a|\overline{s})p(s|\overline{s},a)
      $$

    - We can solve for $\eta(s)$, then 
      $$
      \mu(s)=\frac{\eta(s)}{\sum_{s'}\eta(s')}
      $$

    - 

