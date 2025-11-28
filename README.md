# Purpose and Data

This repository includes the Python code for the task: Quantile regression prediction of residential electricity (power) demand in two locations Houston and Austin, Texas using two datasets:
- Historical weather data between (2015-2025),
- Electricty consumption (2015-2025) in residential sector in Texas.
The targeted consumers are socially vulnerable communities, whose power consumption is approximately 10% lower than the average.
# Method
The qunatile regression is implemented through a Sequence-to-Sequence RNN network equipped with Attention mechanism (Bahdanau Method). In Bahdanau method, encoder hidden state is updated using attention score:

$$
h_{(t)} = f([\hat{y}_{(t-1)}; c_{(t)}], h_{(t-1)})
$$

$$
\hat{y}_{(t)} = g(\hat{y}_{(t-1)}, c_{(t)}, h^{enc}_{(t)})
$$
Where
$$
c_{(t)} = \sum_{j=1}^{T} \alpha_{t,j} h^{enc}_{(j)}
$$

$$
\alpha_{t,j} = \frac{exp(e_{t,j})}{\sum_{k=1}^{T}exp(e_{t,k})}
$$

$$
e_{t,j} = V^\top tanh(W(h^{enc}_{(t-1)}, h^{dec}_{(j)}))
$$

# Repo Guide
A description of the files in the repository is provided below.
*  **Attention.py**: The attention mechanism class, with learnable parameter matrix W and vector V.
*  **Decoder.py**:  Decoder class with GRU cells, attention, and final quantile prediction linear layer.
*  **Encoder.py**:  Encoder class with GRU cells and linear+sigmoid layers.
*  **Inference.py**:  Inference function.
*  **Loss.py**: Quantile loss class with forward function calculating loss for each quantile $q$:

$$L_q = q (y-\hat{y})_+ + (1-q)(\hat{y} - y)_+$$

*  **Seq2Seq_RNN_Attention.py**:  Seq2Seq class containing the Seq2Seq model itself, loss criterion, Adam optimizer, and early stopping attributes. The functions are training the model, evaluating it, and prediction. The file also includes related helpful functions such as MAE loss, MAPE loss, certainty calculator, plotting functions, etc.
*  **Seq2Seq.py**:   Seq2Seq model class with encoder, decoder attributes, and forward function. It is then embeded into Seq2Seq_RNN_Attention.py as a class.
*  **main.ipynb**:  The main file collecting data, importing data, doing training, and getting inference.

Other additional files are:
*  **DataPreprocessro.py**:  Functions to prepare data including getting train and validation dataset and getting batches of data.
*  **Load.py**:  Functions to get the demand from datasets, fill any missing values with the mean for numerical data, scale the demand for vulnerable population, and getting time features.
*  **LocationPicker.py**:  Get the information of each location (for convenience and clear code in main.)
*  **TimeFeatures.py**:   Extracting time data features.
*  **Weather.py**:  Collects weather data from the given url using webscrapping, refines the collected data to remove unknown symbols, categorizes the weather conditions, etc.




