This library is based on Zhang's [DeepRL repository](https://github.com/ShangtongZhang/DeepRL). This is an older version that I forked (from probably mid 2019). There have been several changes made to the repo since then for simplicity, efficiency and robustness. (for example, the DQN for images using this repo sometimes runs out of memory, so use this with caution)

Things I added to the library:
- DQNAgent_v2: feature to finetune on the final layers of the DQN while freezing the rest.
- avDSRAgent: To learn state representations using mixed policy successor features. 

A more [recent version](/deep_rl) is also in this repo, with the same features, so use that to be safe.

Note: `requirements.txt` does not include the installation of [openai baselines package](https://github.com/openai/baselines). You might have to install it seperately. 