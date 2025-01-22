# TCNGAN: Generative Adversarial Network for Financial Time-Series

**Goal:**
TCNGAN are developed by adding TCN block to vanilla GAN for finanical time-series data generation to support strategy backtesting and modification.

**Discussion:**
First, the model can be improved to incorporate prior information to better estimate the tail.
Second, the baseline models should include traditional financial statistical model for thorough evalualtion.

**Result:**

TCNGAN Generated Cumulative Log Return
![TCNGAN Generated Cumulative Log Return](https://github.com/TracyWu7724/TCNGAN/blob/main/checkpoints/ret.png)

TCNGAN Generated Cumulative Log Return V.S. Historical Cumulative Log Return
![TCNGAN Generated Cumulative Log Return V.S. Historical Cumulative Log Return](https://github.com/TracyWu7724/TCNGAN/blob/main/checkpoints/window.png)

