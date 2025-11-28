# Purpose and Data
This repository includes the Python code for the task: Quantile regression prediction of residential electricity (power) demand in two locations Houston and Austin, Texas using two datasets:
- Historical weather data between (2015-2025),
- Electricty consumption (2015-2025) in residential sector in Texas.
The targeted consumers are socially vulnerable communities, whose power consumption is approximately 10% lower than the average.
# Method
The qunatile regression is implemented through a Sequence-to-Sequence RNN network equipped with Attention mechanism (Bahdanau Method). In Bahdanau method, encoder hidden state is updated using attention score:

$h_{(t)} = \hat{y}_{(t-1)}, c_{(t)}, h_{(t-1)}$,

$c_{(t)} = \sum_{j=1}^{T} \alpha_{t,j} h_{(j)}$,

$\alpha_{t,j} = \frac{e_{t,j}}{\sum_{k=1}^{T}e_{t,k}}$
# Repo Guide
A description of the files in the repository is provided below.
