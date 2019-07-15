---
layout:     post                    # 使用的布局（不需要改）
title:      A Tutorial on Thompson Sampling             # 标题 
subtitle:   HKUST Summer Research Task 1 #副标题
date:       2019-07-13             # 时间
author:     BY Zhipeng Liang                    # 作者
header-img: img/what-is-reinforcement-learning-the-complete-guide.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 强化学习
---

## A Tutorial on Thompson Sampling

Core Function:

```{python}
def reproduce_figure(figure_options, run_frac, data_path, plot_path):
  """Function to reproduce figures for TS tutorial.

  Args:
    figure_options: a FigureOptions namedtuple.
    run_frac: float in [0,1] of how many jobs to run vs paper.
    data_path: where to save intermediate experiment .csv.
    plot_path: where to save output plot.

  Returns:
    None, experiment results are written to data_path and plots to plot_path.
  """
  experiment_name, n_jobs = _logging(
      figure_options, run_frac, data_path, plot_path)

  # Running the jobs via command line (this can/should be parallelized)
  for i in range(n_jobs):
    print('Starting job {} out of {}'.format(i, n_jobs))
    os.system('python batch_runner.py --config {} --job_id {} --save_path {}'
              .format(figure_options.config, i, data_path))

  # Plotting output
  plot_dict = figure_options.plot_fun(experiment_name, data_path)
  _save_plot_to_file(plot_dict, plot_path, run_frac)


"""
Batch Runner: in charge of combining agent, experiment, environment, config and start experiment
"""
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run batch experiment')
  parser.add_argument('--config', help='config', type=str)
  parser.add_argument('--job_id', help='job_id', type=int)
  parser.add_argument('--save_path', help='save_path', type=str, default='./tmp/')
  args = parser.parse_args()

  # Loading in the experiment config.
  experiment_config = importlib.import_module(args.config)
  config = experiment_config.get_config()

  # Running the experiment.
  job_config = config_lib.get_job_config(config, args.job_id)
  experiment = job_config['experiment']
  experiment.run_experiment()

  # Saving results to csv.
  file_name = ('exp=' + config.name
               + '&id=' + str(args.job_id) + '.csv')
  file_path = os.path.join(args.save_path, file_name)
  with open(file_path, 'w') as f:
    experiment.results.to_csv(f, index=False)
  # Save the parameters if it is the first job.
  if args.job_id == 0:
    params_df = config_lib.get_params_df(config)
    file_name = 'exp=' + config.name + '&params.csv'
    file_path = os.path.join(args.save_path, file_name)
    with open(file_path, 'w') as f:
      params_df.to_csv(f, index=False)
    
    
"""
Agent
"""
class FiniteBernoulliBanditEpsilonGreedy(Agent):
  """Simple agent made for finite armed bandit problems."""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    self.n_arm = n_arm
    self.epsilon = epsilon
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success)
    self.prior_failure = np.array(prior_failure)

  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure)

  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure)

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    if np.isclose(reward, 1):
      self.prior_success[action] += 1
    elif np.isclose(reward, 0):
      self.prior_failure[action] += 1
    else:
      raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

  def pick_action(self, observation):
    """Take random action prob epsilon, else be greedy."""
    if np.random.rand() < self.epsilon:
      action = np.random.randint(self.n_arm)
    else:
      posterior_means = self.get_posterior_mean()
      action = random_argmax(posterior_means)

    return action


##############################################################################


class FiniteBernoulliBanditTS(FiniteBernoulliBanditEpsilonGreedy):
  """Thompson sampling on finite armed bandit."""

  def pick_action(self, observation):
    """Thompson sampling with Beta posterior for action selection."""
    sampled_means = self.get_posterior_sample()
    action = random_argmax(sampled_means)
    return action




"""
Experiment
"""
##############################################################################

class BaseExperiment(object):
  """Simple experiment that logs regret and action taken.

  If you want to do something more fancy then you should extend this class.
  """

  def __init__(self, agent, environment, n_steps,
               seed=0, rec_freq=1, unique_id='NULL'):
    """Setting up the experiment.

    Note that unique_id should be used to identify the job later for analysis.
    """
    self.agent = agent
    self.environment = environment
    self.n_steps = n_steps
    self.seed = seed
    self.unique_id = unique_id

    self.results = []
    self.data_dict = {}
    self.rec_freq = rec_freq


  def run_step_maybe_log(self, t):
    # Evolve the bandit (potentially contextual) for one step and pick action
    observation = self.environment.get_observation()
    action = self.agent.pick_action(observation)

    # Compute useful stuff for regret calculations
    optimal_reward = self.environment.get_optimal_reward()
    expected_reward = self.environment.get_expected_reward(action)
    reward = self.environment.get_stochastic_reward(action)

    # Update the agent using realized rewards + bandit learing
    self.agent.update_observation(observation, action, reward)

    # Log whatever we need for the plots we will want to use.
    instant_regret = optimal_reward - expected_reward
    self.cum_regret += instant_regret

    # Advance the environment (used in nonstationary experiment)
    self.environment.advance(action, reward)

    if (t + 1) % self.rec_freq == 0:
      self.data_dict = {'t': (t + 1),
                        'instant_regret': instant_regret,
                        'cum_regret': self.cum_regret,
                        'action': action,
                        'unique_id': self.unique_id}
      self.results.append(self.data_dict)


  def run_experiment(self):
    """Run the experiment for n_steps and collect data."""
    np.random.seed(self.seed)
    self.cum_regret = 0
    self.cum_optimal = 0

    for t in range(self.n_steps):
      self.run_step_maybe_log(t)

    self.results = pd.DataFrame(self.results)

```



### Chapter 3: Thompson Sampling for the Bernoulli Bandit
1. Greedy action: always select the action with the highest expectation in terms of posterior distributions without exploration.
2. Dithering is a common approach to exploration. (e.g. $\epsilon$-greedy exploration). Weakness: wastes resources by failing to write off actions regardlenss of how unlikely they are to be optimal.
3. **Thompson Sampling**: similar in the proceeding but estimate $\hat{\theta_k}$ by randomly sampled from the posterior distribution which is a beta distribution with parameters $\alpha_k$ and $\beta_k$ rather than taken to be the expectation $\frac{\alpha_k}{\alpha_k+\beta_k}$

Beta-Bernoulli Bandit: Rewards of different bandits follow Bernoulli distributions while prior belief over each $\theta$ is Beta Distribution.

When the differentiation of the expectations of different bandits are small, the regret is generally smaller over early time periods and larger over later time periods.



```{python}
"""
Step 1: Get Config by Name: finite_simple
Step 2: Experiment: Run Batch runner 
Step 3: load data from each experiment, concat them and plot
"""

#Step 1
def get_config():
  """Generates the config for the experiment."""
  name = 'finite_simple'
  n_arm = 3
  agents = collections.OrderedDict(
      [('greedy',
        functools.partial(FiniteBernoulliBanditEpsilonGreedy, n_arm)),
       ('ts', functools.partial(FiniteBernoulliBanditTS, n_arm))]
  )
  probs = [0.7, 0.8, 0.9]
  environments = collections.OrderedDict(
      [('env', functools.partial(FiniteArmedBernoulliBandit, probs))]
  )
  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 1000
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config


#Step 3.1
def load_data(experiment_name, data_path='./tmp/'):
  """Function to load in the data relevant to a specific experiment.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    df: dataframe of experiment data (uses cache for faster reloading).
  """
  if experiment_name in _DATA_CACHE:
    return _DATA_CACHE[experiment_name]
  else:
    all_files = os.listdir(data_path)
    good_files = []
    for file_name in all_files:
      if '.csv' not in file_name:
        continue
      else:
        file_experiment = file_name.split('exp=')[1].split('&')[0]
        if file_experiment == experiment_name:
          good_files.append(file_name)

    data = []
    for file_name in good_files:
      file_path = os.path.join(data_path, file_name)
      if 'id=' in file_name:
        if os.path.getsize(file_path) < 1024:
          continue
        else:
          data.append(pd.read_csv(file_path))
      elif 'params' in file_name:
        params_df = pd.read_csv(file_path)
        params_df['agent'] = params_df['agent'].apply(_name_cleaner)
      else:
        raise ValueError('Something is wrong with file names.')

    df = pd.concat(data)
    df = pd.merge(df, params_df, on='unique_id')
    _DATA_CACHE[experiment_name] = df
    return _DATA_CACHE[experiment_name]

def _name_cleaner(agent_name):
  """Renames agent_name to prettier string for plots."""
  rename_dict = {'correct_ts': 'Correct TS',
                 'kl_ucb': 'KL UCB',
                 'misspecified_ts': 'Misspecified TS',
                 'ucb1': 'UCB1',
                 'nonstationary_ts': 'Nonstationary TS',
                 'stationary_ts': 'Stationary TS',
                 'greedy': 'Greedy',
                 'ts': 'TS',
                 'action_0': 'Action 0',
                 'action_1': 'Action 1',
                 'action_2': 'Action 2',
                 'bootstrap': 'bootstrap TS',
                 'laplace': 'Laplace TS',
                 'thoughtful': 'Thoughtful TS',
                 'gibbs': 'Gibbs TS'}
  if agent_name in rename_dict:
    return rename_dict[agent_name]
  else:
    return agent_name

#Step 3.2
def compare_action_selection_plot(experiment_name='finite_simple',
                                  data_path=_DEFAULT_DATA_PATH):
  """Specialized plotting script for TS tutorial paper action proportion."""
  df = load_data(experiment_name, data_path)
  plot_dict = {}
  for agent, df_agent in df.groupby(['agent']):
    key_name = experiment_name + '_' + agent + '_action'
    plot_dict[key_name] = plot_action_proportion(df_agent)
  return plot_dict


def plot_action_proportion(df_agent):
  """Plot the action proportion for the sub-dataframe for a single agent."""
  n_action = np.max(df_agent.action) + 1
  plt_data = []
  for i in range(n_action):
    probs = (df_agent.groupby('t')
             .agg({'action': lambda x: np.mean(x == i)})
             .rename(columns={'action': 'action_' + str(i)}))
    plt_data.append(probs)
  plt_df = pd.concat(plt_data, axis=1).reset_index()
  p = (gg.ggplot(pd.melt(plt_df, id_vars='t'))
       + gg.aes('t', 'value', colour='variable', group='variable')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('Action probability')
       + gg.ylim(0, 1)
       + gg.scale_colour_brewer(name='Variable', type='qual', palette='Set1'))
  return p
```

![finite_simple_greedy_action](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/finite_simple_greedy_action.png)

![finite_simple_ts_action](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/finite_simple_ts_action.png)


Figure 3.2a (In python program is 4a)
```{python}
def simple_algorithm_plot(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Simple plot of average instantaneous regret by agent, per timestep.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
  """
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'instant_regret': np.mean})
            .reset_index())
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'instant_regret', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('per-period regret')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
  
  plot_dict = {experiment_name + '_simple': p}
  return plot_dict
```

Figure 3.2b (In python program is 4b)
```{python}
def get_config():
  """Generates the config for the experiment."""
  name = 'finite_simple_rand'
  n_arm = 3
  agents = collections.OrderedDict(
      [('greedy',
        functools.partial(FiniteBernoulliBanditEpsilonGreedy, n_arm)),
       ('ts', functools.partial(FiniteBernoulliBanditTS, n_arm))]
  )

#Below is the main difference between rand with normal version
  environments = collections.OrderedDict()
  n_env = 100
  for env in range(n_env):
    probs = np.random.rand(n_arm)
    environments[env] = functools.partial(FiniteArmedBernoulliBandit, probs)
#


  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 1000
  n_seeds = 100
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
```

![finite_simple_simple](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/finite_simple_simple.png)

![finite_simple_rand_simple](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/finite_simple_rand_simple.png)

### Chapter 4: General Thompson Sampling

##### Independent Travel Times
$\theta_e$ is independent and log-Gaussian-distributed with parameters $\mu_e$ and $\sigma^2$.

$y_{t,e}|\theta$ log-Gaussian with parameters $ln(\theta_e)-\sigma^2/2$ and $\sigma^2$.

##### Correlated Travel Times
$$
y_{t,e}=\zeta_{t,e}\eta_t v_{t,l(e)} \theta_e
$$

with $\zeta_{t,e}$, $\eta_t$, $v_{t,l(e)}$ all follow log-Gaussian-distributed with parameters $-\sigma^2/6$ and $\sigma^2/3$ and known.

Define $\phi_e=ln(\theta_e)$ and the posterior distribution of $\phi_e$ is Gaussian with mean vector $\mu$ and covariance matrix $\Sigma$. Updated rule according to 

$$
(\mu,\Sigma)=((\Sigma^{-1}+C)^{-1}(\Sigma^{-1}\mu+Cz_t),(\Sigma^{-1}+C)^{-1})
$$

$$
\Sigma_{e,e'}=\left\{
\begin{aligned}
\sigma^2, &\ for\ e=e' \\
2\sigma^2/3 , &\ for \ e\neq e', l(e)=l(e') \\
\sigma^2/3 , & \ otherwise
\end{aligned}
\right.
$$
and 
$$
C_{e,e'}=\left\{
\begin{aligned}
\Sigma_{e,e'}^{-1},& & if\  e,e'\in x_t\\
0&,& otherwise
\end{aligned}
\right.
$$

### Chapter 5: Approximations
##### Binary Feedback
$\theta_e$ is independent and gamma-distributed and observations be generated according to 
$$
y_t|\theta_e~\left\{
\begin{aligned}
1,& & with probability \ \frac{1}{1+exp(\sum_e\in x_t \theta_e - M)} \\
0,& & otherwise
\end{aligned}
\right.
$$

to model in Internet route recommendation service. Drivers will express whether the route is desirable (whether travel time is less than prior expectation M)

However this model does not enjoy conjugacy properties leveraged in Chapter 4. Therefore, approximation method is needed.

For Bernoulli bandit problem, the posterior density has the function:
$$
f(\theta)\propto \theta^s(1-\theta)^{n-s}
$$
where s is the success count.

Taking logs gives:
$$
\begin{aligned}
L(\theta)&=const+sln(\theta)+(n-s)ln(1-\theta)\\
\frac{dL(\theta)}{d\theta}&=\frac{s}{\theta}-\frac{n-s}{1-\theta}=0\rightarrow \theta_0=\frac{s}{n}\\
\frac{d^2L(\theta)}{d\theta^2}&=-\frac{s}{\theta^2}-\frac{n-s}{(1-\theta)^2}=-\frac{n}{\theta(1-\theta)}\\
\end{aligned}
$$


1. Gibbs Sampling (still computationally demanding)

2. Laplace Approximation (approximate probability density near the global maximum by second-order Taylor approximation and Gaussian distribution)-- backtracking line search, drawback: not appropriate for posterior distribution is not sufficiently close to Gaussian

```{python}
class FiniteBernoulliBanditLaplace(FiniteBernoulliBanditTS):
  """Laplace Thompson sampling on finite armed bandit."""

  def get_posterior_sample(self):
    """Gaussian approximation to posterior density (match moments)."""
    (a, b) = (self.prior_success + 1e-6 - 1, self.prior_failure + 1e-6 - 1)
    # The modes are not well defined unless alpha, beta > 1
    assert np.all(a > 0)
    assert np.all(b > 0)
    mode = a / (a + b)
    hessian = a / mode + b / (1 - mode)
    laplace_sample = mode + np.sqrt(1 / hessian) * np.random.randn(self.n_arm)
    return laplace_sample
```
(Is there any wrong with the posterior laplace approximation?)


3. Langevin Monte Carlo: 
   $$
   d\phi_t=\nabla ln(g(\phi_t))dt+\sqrt{2}dB_t
   $$
   This process has g as its stationary distribution and under reasonable technical conditions, the distribution of $\phi_t$ converges rapidly to this stationary distribution. Typically, one instead implements a Euler discretization of this stochastic differential equation
   $$
   \phi_{n+1}=\phi_n+\epsilon\nabla ln(g(\phi_n))+\sqrt{2\epsilon}W_n
   $$
   where $W_n, n=1,2,\dots,$ are standard Gaussian random variables.

   Modification: Precondition
   $$
   \phi_{n+1}=\phi_n+\epsilon A \nabla ln(g(\phi_n))+\sqrt{2\epsilon}A^{1/2}W_n
   $$
   where $A=-(\nabla^2 ln(g(\phi))|_{\phi=\phi_0})^{-1}$ and $\phi_0=argmax_{\phi} ln(g(\phi))$

```{python}
class FiniteBernoulliBanditLangevin(FiniteBernoulliBanditTS):
  '''Langevin method for approximate posterior sampling.'''
  
  def __init__(self,n_arm, step_count=100,step_size=0.01,a0=1, b0=1, epsilon=0.0):
    FiniteBernoulliBanditTS.__init__(self,n_arm, a0, b0, epsilon)
    self.step_count = step_count
    self.step_size = step_size
  
  def project(self,x):
    '''projects the vector x onto [_SMALL_NUMBER,1-_SMALL_NUMBER] to prevent
    numerical overflow.'''
    return np.minimum(1-_SMALL_NUMBER,np.maximum(x,_SMALL_NUMBER))
  
  def compute_gradient(self,x):
    grad = (self.prior_success-1)/x - (self.prior_failure-1)/(1-x)
    return grad
    
  def compute_preconditioners(self,x):
    second_derivatives = (self.prior_success-1)/(x**2) + (self.prior_failure-1)/((1-x)**2)#self.prior_success-1=s,self.prior_failure-1=n-s
    second_derivatives = np.maximum(second_derivatives,_SMALL_NUMBER)
    preconditioner = np.diag(1/second_derivatives)
    preconditioner_sqrt = np.diag(1/np.sqrt(second_derivatives))
    return preconditioner,preconditioner_sqrt
    
  def get_posterior_sample(self):
        
    (a, b) = (self.prior_success + 1e-6 - 1, self.prior_failure + 1e-6 - 1)
    # The modes are not well defined unless alpha, beta > 1
    assert np.all(a > 0)
    assert np.all(b > 0)
    x_map = a / (a + b)
    x_map = self.project(x_map)
    
    preconditioner, preconditioner_sqrt=self.compute_preconditioners(x_map)
   
    x = x_map
    for i in range(self.step_count):
      g = self.compute_gradient(x)
      scaled_grad = preconditioner.dot(g)
      scaled_noise= preconditioner_sqrt.dot(np.random.randn(self.n_arm)) 
      x = x + self.step_size*scaled_grad + np.sqrt(2*self.step_size)*scaled_noise
      x = self.project(x)
      
    return x
```



   Two modifications are implemented. 1. Stochastic gradient Langevin Monte Carlo, which uses sampled minibatches of data to compute approximate rather than exact gradients.  2. the use of a preconditioning matrix:
$$
\phi_{n+1}=\phi_n+\epsilon A\nabla ln(g(\phi_n))+\sqrt{2\epsilon}A^{1/2}W_n
$$

4. Bootstrapping

   In order  to incorporate prior into traditional bootstrap method for evaluating the sampling distribution of the maximum likelihood estimate of $\theta$, the method proceeds as follows:

   1. Draw a hypothetical history $\mathbb{\hat{H}}_{t-1}=((\hat{x}_1,\hat{y}_1),\dots,(\hat{x}_1,\hat{y}_1))$, which is made up of t-1 action-observation pairs, each sampled uniformly with replacement from $\mathbb{H}_{t-1}$  (Traditional Bootstrap for likelihood)

   2. Draw a sample $\theta^0$ from the prior distribution $f_0$, Let  $\Sigma$ denote the covariance matrix of the prior $f_0$

   3. Finally we solve the maximization problem and treat $\hat{\theta}$ as an approximate posterior sample.
      $$
      \hat{\theta}=argmax_{\theta\in \mathbb{R}^k} e^{-(\theta-\theta^0)^T\Sigma(\theta-\theta^0)\hat{L}_{t-1}(\theta)}
      $$
      One advantage of the bootstrap is that it is nonparametric, and may work reasonably regardless of the functional form of the posterior distribution, whereas the Laplace approximation relies on a Gaussian approximation and Langevin Monte Carlo relies on log-concavity and other regularity assumptions.

```{python}
class FiniteBernoulliBanditBootstrap(FiniteBernoulliBanditTS):
  """Bootstrapped Thompson sampling on finite armed bandit."""

  def get_posterior_sample(self):
    """Use bootstrap resampling instead of posterior sample."""
    total_tries = self.prior_success + self.prior_failure
    prob_success = self.prior_success / total_tries
    boot_sample = np.random.binomial(total_tries, prob_success) / total_tries
    return boot_sample
```


![finite_simple_sanity_simple](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/finite_simple_sanity_simple.png)

**Incremental Implementation**

We refer to an algorithm as *incremental* if it operates with fixed rather than growing per-period compute time.

Incremental version of Laplace approximation:

$$
H_t=H_{t-1}+\nabla^2g_t(\overline{\theta}_{t-1})\\
\overline{\theta}_t=\overline{\theta}_{t-1}-H_t^{-1}\nabla g_t(\overline{\theta}_{t-1})
$$
Since $\nabla^2g_t(\overline{\theta}_{t-1})$ has rank one, therefore $H_t^{-1}=(H_{t-1}+\nabla^2 g_t(\overline{\theta}_{t-1}))^{-1}$ can be updated incrementally using the Sherman-Woodbury-Morrison formula.



**Ensemble sampling**

Totally cannot understand 555



### Chapter 6: Practical Modeling Considerations

######Misspecified

The choice of prior can be important.


```{python}
class DriftingFiniteBernoulliBanditTS(FiniteBernoulliBanditTS):
  """Thompson sampling on finite armed bandit."""

  def __init__(self, n_arm, a0=1, b0=1, gamma=0.01):
    self.n_arm = n_arm
    self.a0 = a0
    self.b0 = b0
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])
    self.gamma = gamma

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    # All values decay slightly, observation updated
    self.prior_success = self.prior_success * (
        1 - self.gamma) + self.a0 * self.gamma
    self.prior_failure = self.prior_failure * (
        1 - self.gamma) + self.b0 * self.gamma
    self.prior_success[action] += reward
    self.prior_failure[action] += 1 - reward
    
    
"""
plot
"""
#############################################################################
# Misspecified prior plots

def misspecified_plot(experiment_name='finite_misspecified',
                      data_path=_DEFAULT_DATA_PATH):
  """Specialized plotting script for TS tutorial paper misspecified TS."""
  df = load_data(experiment_name, data_path)

  def _parse_np_array(np_string):
    return np.array(np_string.replace('[', '')
                    .replace(']', '')
                    .strip()
                    .split())
  df['posterior_mean'] = df.posterior_mean.apply(_parse_np_array)

  # Action means
  new_col_list = ['mean_0', 'mean_1', 'mean_2']
  for n, col in enumerate(new_col_list):
    df[col] = df['posterior_mean'].apply(lambda x: float(x[n]))

  plt_df = (df.groupby(['agent', 't'])
            .agg({'instant_regret': np.mean,
                  'mean_0': np.mean,
                  'mean_1': np.mean,
                  'mean_2': np.mean})
            .reset_index())

  regret_plot = (gg.ggplot(plt_df)
                 + gg.aes('t', 'instant_regret', colour='agent')
                 + gg.geom_line(size=1.25, alpha=0.75)
                 + gg.xlab('time period (t)')
                 + gg.ylab('per-period regret')
                 + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1')
                 + gg.coord_cartesian(ylim=(0, 0.02)))

  melt_df = pd.melt(plt_df, id_vars=['agent', 't'], value_vars=new_col_list)
  melt_df['group_id'] = melt_df.agent + melt_df.variable
  action_plot = (gg.ggplot(melt_df)
                 + gg.aes('t', 'value', colour='agent', group='group_id')
                 + gg.geom_line(size=1.25, alpha=0.75)
                 + gg.coord_cartesian(ylim=(0, 0.05))
                 + gg.xlab('time period (t)')
                 + gg.ylab('Expected mean reward')
                 + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))

  plot_dict = {}
  plot_dict['misspecified_regret'] = regret_plot
  plot_dict['misspecified_action'] = action_plot
  return plot_dict
```

![misspecified_action](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/misspecified_action.png)



![misspecified_regret](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/misspecified_regret.png)

#####Extensions:

1. Time-varying constraints

2. Contextual online decision problems: in such problems, the response $y_t$ to action $x_t$ also depends on an independent random variable $z_t$ that the agent observes prior to making her decision. The conditional distribution of $y_t$ takes the form $p_\theta(\cdot|x_t,z_t)$ instead of $p_\theta(\cdot|x_t)$. For the shortest path problem, this can be interpreted as allowing the agent to dictate both the weather report and the path to traverse but constraining the agent to provide a weather report identical to the one observed through the news channel.

3. Baseline

#####Simple approach to addressing nonstationary systems problems (with time-varying parameters $\theta_1,\theta_2,\dots$):

1. Ignoring historical observations made beyond some number $\tau$ of time periods in the past. The agent never ceases to explore, since the degree to which the posterior distribution can concentrate is limited by the number of observations taken into account.

2. Discounts the relevance of past observations and tracks a time-varying parameters $\theta+t$:
   $$
   (\alpha_k,\beta_k)=\left\{
   \begin{aligned}
   ((1-\gamma)\alpha_k+\gamma \overline{\alpha},(1-\gamma)\beta_k+\gamma \overline{\beta})& & x_t\neq k\\
   ((1-\gamma)\alpha_k+\gamma \overline{\alpha}+r_t,(1-\gamma)\beta_k+\gamma \overline{\beta}+1-r_t)& & x_t=k\\
   \end{aligned}
   \right.
   $$
   where $\overline{\alpha}$ and $\overline{\beta}$ is r.v.. Intuitively, the process can be thought of as randomly perturbing model parameters in each time period, injecting uncertainty.

   Another modification can be transforming the distribution updating process from:
   $$
   p(u)\leftarrow \frac{p(u)q_u(y_t|x_t)}{\sum_vp(v)q_v(y_t|x_t)}
   $$
   into:


$$
p(u)\leftarrow \frac{\overline{p}^\gamma(u)p^{1-\gamma}(u)q_u(y_t|x_t)}{\sum_v\overline{p}^\gamma(v)p^{1-\gamma(v)}q_v(y_t|x_t)}
$$

![finite_drift_simple](F:/ts_tutorial-master/ts_tutorial-master/src/tmp/finite_drift_simple.png)

#####Concurrence

Involving synchronous action selection and posterior updating



### Chapter 7: Further Examples

##### News Article Recommendation

Feature vector $z_t\in \mathbb{R}^d$ associated with the t th user, chooses a news article $x_t$ and a positive review occurs with probability $g(z_t^T\theta_{x_t})$ where g is the logistic function, given by $g(a)=\frac{1}{1+e^{-a}}$。

##### Product Assortment

##### Cascading Recommendations

A cascading bandit model is identified by a triple (K,J,$\theta$), where K is the number of items, $J\leq K$ is the number of items recommended in each period and $\theta\in [0,1]^K$ is a vector of attraction probabilities. Recommendation platform recommend a desirable list of items to a user. 

The results demonstrate that TS far outperforms UCB1. In particular, h(x, $U_t$) represents the probability of a click if every item in x simultaneously takes on the largest attraction probability that is statistically plausible. However, due to the statistical independence of item attractions, the agent is unlikely to have substantially over-estimated the attraction probability of every item in x. As such, $h(x.U_t)$ tends to be far too optimistic. CascadeTS, on the other hand, samples components $\hat{\theta}_k$ independently across item. Another import likely source of loss stems from the shape of confidence sets used by CascadeUCB. Note that the algorithm uses hyper-rectangular confidence sets, since the set of statistically plausible attraction probability vectors is characterizzed by a Cartesian product item-level confidence intervals. However, the Bayesian central limit theorem suggests that ellipsoidal confidence sets offer a more suitable choice. Therefore, it may be possible that CascadeUCB can outperform CascadeTS when the dimension is low.



##### Active Learning with Neural Networks

先放一下，查一下ensemble model



##### Reinforcement Learning in Markov Decision Processes

RL extends upon contextual online decision problems to allow for **delayed feedback** and **long term consequences**.

Similar with contextual multi-arm bandit problem, the response $y_t$ to the action $x_t$ depends on a context $z_t$ but we also allow the evolution of the context $z_{t+1}$ is independent of $y_t$. As such, the action $x_t$ may affect not only the reward $r(y_t)$ but also, through the effect upon the context $z_{t+1}$.

**Deep Exploration**: sampling only once prior to each episode and holding the policy fixed for the duration of the episode rather than resampling after each step.

Posterior sampling for reinforcement learning (PSRL) fits in the broader family of Bayesian approaches to efficient reinforcement learning.



### Chapter 8: Theoretical Analysis

**Why Thompson Sampling Works**

###### Regret Analysis fr Classical Bandit Problems

Regret analysis for classical bandit problems reveals that: 
$$
lim_{T\rightarrow\infty} \frac{\mathbb{E}[Regret(T)|\theta]}{log(T)}=\sum_{k\neq k*}\frac{\theta_{k*}-\theta}{d_{KL}(\theta_{k*}||\theta_k)}
$$
Assuming that there is a unique optimal action k*. This is the lower bound of all possible algorithm. The regret of TS exhibits this scaling and a series of papers provided proofs that formalize this finding. The bound essentially focuses on a regime in which the agent is highly confident of which action is best but continues to occasionally explore in order to become even more confident. If we specialize to the case in which rewards, conditioned on $\theta$, are Gaussian with unit variance, for which $d_{KL}(\theta||\theta')=(\theta-\theta')^2/2$, then:
$$
\mathbb{E}[Regret(T)|\theta]=\sum_{k\neq k*}\frac{2}{\theta_{k*}-\theta_k}log(T)
$$
The final expression is dominated by **near-optimal** actions reflects that in the relevant asymptotic regime other actions can be essentially ruled out using far fewer samples.For the Bernoulli bandit problem, Agrawal and Goyal established that when TS is initialized with a uniform prior: (8.3)
$$
max_{\theta'}\mathbb{E}[Regret(T)|\theta=\theta']=O(\sqrt{KTlog(T)})              
$$
This regret bounds hold uniformly over all problem instances, ensuring that there are no instances of bandit problems with binary rewards that will cause the regret of TS to explode. Also this bound is nearly order-optimal. 

###### Regret Analysis for Complex Online Decision Problems

Let
$$
\mu(x,\theta)=\mathbb{E}[r(g(x,\theta,w_t))|\theta]
$$
The algorithm Bayesian regret:
$$
\mathbb{E}[Regret(T)]=\mathbb{E}_{\theta,w_t}[\sum_{t=1}^T(\mu(x*,\theta)-\mu(x,\theta))]
$$
No single algorithm can minimize conditional expected regret $\mathbb{E}[Regret(T)|\theta=\theta']$ for every problem instance $\theta'$. However, minimizing integrated regret $\mathbb{E}[Regret(T)]=\mathbb{E}[\mathbb{E}[Regret(T)|\theta]]$ can direct the algorithm priortize strong performance in more likely scenarios. With any choice of $U_t$, regret over the period decomposes according to 
$$
\begin{aligned}
\mu(x^*,\theta)-\mu(\overline{x}_t,\theta)&=\mu(x^*,\theta)-U_t(\overline{x}_t)+-U_t(\overline{x}_t)-\mu(\overline{x}_t,\theta)\\
&\leq \underbrace{\mu(x^*,\theta)-U_t(x^*)}+\underbrace{U_t(\overline{x}_t)-\mu(\overline{x}_t,\theta)}\\
& \quad \quad \quad pessimism \quad \quad \quad \quad \quad width
\end{aligned}
$$
Regret bounds for UCB algorithms are obtained by characterizing the rate at which this slack diminishes as actions are applied.

For TS, similar results can be derived
$$
\begin{aligned}\mu(x^*,\theta)-\mu(\overline{x}_t,\theta)&=\mu(x^*,\theta)-U_t(\overline{x}_t)+-U_t(\overline{x}_t)-\mu(\overline{x}_t,\theta)\\
&=\underbrace{\mu(x^*,\theta)-U_t(x^*)}+\underbrace{U_t(\overline{x}_t)-\mu(\overline{x}_t,\theta)}\\& \quad \quad \quad pessimism \quad \quad \quad \quad \quad width\end{aligned}
$$
The second equation puzzles me a lot and remains to be solve... However, this expression reveal an important difference between UCB algorithm and TS: UCB regret bounds depend on the specific choice of $U_t$ by the algorithm in question while with TS $U_t$ plays no role in the algorithm and appears only as a figment of regret analysis. This suggests that, while the regret of a UCB algorithm depends critically on the specific choice of upper-confidence bound, TS depends only on the best possible choice. This is the crucial advantage when there are complicated dependencies among actions, as designing and computing with appropriate upper-confidence bounds present significant challenges.

Under TS:(8.5)
$$
\mathbb{E}[Regret(T)]=O(d\sqrt{T}log(T))
$$
An important feature of this bound is that it depends on the complexity of the parameterized model through the dimension d and not on the number of actions. Indeed, when there are a very large, or even infinite number of actions, bounds like (8.3) are vacuous whereas (8.5) may still provide a meaning full guarantee.

###### Why Randomize Actions

TS is a stationary randomized strategy: stationary means action distribution is determined by the posterior distribution of $\theta$ and otherwise independent of the time period. randomized means each action is randomly sampled from a distribution. It follows that, for any deterministic stationary strategy, there exists a prior probability $p_0$ such that expected cumulative regret grows linearly with time. As such, for expected cumulative regret to exhibit a sublinear horizon dependence, as is the case with the bounds we have discussed, a stationary strategy must randomize actions.



##### Limitations of Thompson Sampling

1. Problems that do not require exploration: TS is usually outperformed by greedier algorithms that do not invest in costly exploration in which problem learning does not require active exploration. In contextual bandit problems, even when actions influence observations, randomness of context can give rise to sufficient exploration so that additional active exploration incurs unnecessary cost.(cannot totally understand the Ex. 8.2)

2. Problems that do not require exploitation: e.g. pure-exploration problem

3. Time Sensitivity: In time-sensitive learning problems where it si better to exploit a high performing suboptimal action than to invest resources exploring actions that might offer slightly improved performance. 

   Example 8.3 (Many-Armed Deterministic Bandit) no algorithm can expect to select $x^*$ within time t<<K. On the other hand, by simply selecting actions in order, with x1=1, x2=2,..., the agent can expect to identify an $\epsilon$-optimal action within t=1/$\epsilon$ time periods, independent of K: settle for the first action x for which $\theta_x\geq 1-\epsilon$.

   Therefore, satisfying TS, a variant of TS that is designed to minimize exploration costs required to identify an action that is sufficiently close to optimal.

4. Problems requiring careful assessment of information gain

   Example 8.4 A Revealing Action

   Example 8.5 Sparse Linear Model





### Some advance methods in python

defaultdict

functools.partial

collections.OrderedDict