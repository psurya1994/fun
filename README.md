## Surya's personal project

This repository contains implementations and illustrative code to personal projects that I took up in my free time. Feel free to reuse any of my code here.

Raise an issue if something doesn't work.

### Summary of projects


RL related projects:
* [Owl environment](/baselines_owl): I created an environment where the agent (which you can think of as an owl) needs to turn it's head to track prey. This was an idea I got while reading the book rethinking consciousness. The goal is to give the agent both eyes and ears, with the help of which it can track prey. I think interesting related research questions that we can ask are: "will training an agent to do task optimally this give raise to attention that switches between ears and eyes?".
* [Tabular agents](/agents_tabular): This folder consists of various RL agents that I built. All the agents here are compatible with other environments in the repository.
* [RL environments](/envs): This folder consists of various RL environments that I built. All the environments here are [OpenAI gym](https://gym.openai.com/) compatible and can be used with other [agents](../agents_tabular) in the repository. Includes environments such as gridworld, pixelgridworld, lineworld and owl.
* [Mixed Policy Successor Features](/mpsf): A way to learn state representations with unsupervised exploration. My MSc thesis.
* [Markov Chain Analysis](/markov_chain_analysis)


ML learning related:
* [Inferring cost function from vector field](/infer_cost_from_vector_field): Can we use the weight change information during training to infer the nature of the loss function?
* [Memory for RL baselines](/memory_rl_baselines): I implement several baselines that use memory in RL (LSTM, RNN and LSTM+DNC) on a simple remember lights environment where the agent has to go left, remember the lights and take appropriate action in the right most cell.
* [Visualizing steepness of minima](/nn_minima_steepness): Estimating the steepness of minima. Research has shown that small batch size gives flatter minima which generalize better whereas large batch size goes steeper minima that don't generalize as well. I test that hypothesis here.
* [Computing Hessian for minima in NNs](/computing_hessian_nn): Following up on the above project, I try to estimate the hessian of the minimas. This can be used to test similar hypothesis as in the above project.
* [Bias and variance in NN](/nn_bias_variance): Analysis bias and variance in neural networks.
* [RNN training](/rnn_training): Training RNNs on toy datasets.
* [Scale invariant CNN](/scale_invariant_cnn): Incorporating a new inductive bias into CNN so that it is more robust to changes in scale.
* [SimCLR and BYOL analysis](/simclr_byol_analysis): Understanding how representations change over time as these networks learn.

Other repository based:
* [DeepRL old](/deep_rl_old): This library is based on Zhang's [DeepRL repository](https://github.com/ShangtongZhang/DeepRL). This is an older version that I forked (from probably mid 2019). There have been several changes made to the repo since then for simplicity, efficiency and robustness. (for example, the DQN for images using this repo sometimes runs out of memory, so use this with caution)
* [DeepRL](/deep_rl): Latest version of the above repo.

