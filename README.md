# SIMBa
## System Identification Methods with Backpropagation

**SIMBa** (**S**ystem **I**dentification **M**ethods leveraging **B**ackpropagation) is an open-source toolbox leveraging Pytorch's Automatic Differentiation framework for stable state-space linear SysID. It allows the user to incorporate prior knowledge (like sparsity patterns of the state-space matrices) during the identification procedure.
SIMBa is available on pypi, it can be installed with `pip install simbapy`.  
Note that this will NOT install SIPPY or matlabengine to avoid compatibility issues (typically dependent on your MATLAB version). You can install them separately to allow Simba's initialization properties. Alternatively, you can clone [this github repository](https://github.com/Cemempamoi/simba) to use simba locally.

## Compatibility with matlab
If matlab is installed on your machine, you can install `matlabengine`. If you are on the latest version of MATLAB, `pip install matlabengine` works, otherwise you might need to install an older version of matlabengine. See [here](https://pypi.org/project/matlabengine) the supported version of MATLAB.  
SIMBa needs access the `System Identification Toolbox` and `Symbolic Math Toolbox` in MATLAB.  
You can disable the use of matlab by overwriting `IS_MATLAB` in `simba.parameters`

## Project status
SIMBa was first presented in [Stable Linear Subspace Identification: A Machine Learning Approach](https://arxiv.org/pdf/2311.03197.pdf) and subsequently extended in [SIMBa: System Identification Methods leveraging Backpropagation](https://arxiv.org/pdf/2311.13889.pdf).

## SIMBa vs Linear RNN

| Model     | Metric  |
| --------- | -------- |
| n         | 2        |
| LR        | 1.00E-02 |
| Max e     | 5000     |
| Grad clip | 100      |

| Model | Epoch | Train loss | Val loss | Test loss | Time |
| ----- | ----- | ---------- | -------- | --------- | ---- |
| SIMBa | 1500  | 1.76E-02   | 4.09E-02 | 2.30E-01  | 52"  |
| RNN   | 463   | 2.34E-02   | 7.27E-02 | 7.68E-01  | 11"  |

| Model | max e | # params | Total time | avg time/100e |
| ----- | ----- | -------- | ---------- | ------------- |
| SIMBa | 5000  | 53       | 02'47"     | 03"           |
| RNN   | 5000  | 35       | 02'03"     | 02"           |

========================================================
| Model     | Metric  |
| --------- | -------- |
| n         | 2        |
| LR        | 1.00E-03 |
| Max e     | 25000    |
| Grad clip | 100      |

| Model | Epoch | Train loss | Val loss | Test loss | Time   |
| ----- | ----- | ---------- | -------- | --------- | ------ |
| SIMBa | 1555  | 1.79E-02   | 3.93E-02 | 2.50E-01  | 54"    |
| RNN   | 4123  | 2.25E-02   | 8.20E-02 | 8.57E-01  | 01'38" |

| Model | max e | # params | Total time | avg time/100e |
| ----- | ----- | -------- | ---------- | ------------- |
| SIMBa | 25000 | 53       | 14'03"     | 03"           |
| RNN   | 25000 | 35       | 09'37"     | 02"           |

========================================================
| Model     | Metric  |
| --------- | -------- |
| n         | 6        |
| LR        | 1.00E-03 |
| Max e     | 10000    |
| Grad clip | 100      |

| Model | Epoch | Train loss | Val loss | Test loss | Time   |
| ----- | ----- | ---------- | -------- | --------- | ------ |
| SIMBa | 2438  | 4.38E-03   | 3.57E-02 | 5.29E-01  | 01'26" |
| RNN   | 1618  | 1.28E-02   | 7.57E-02 | 8.81E-01  | 34"    |

| Model | max e | # params | Total time | avg time/100e |
| ----- | ----- | -------- | ---------- | ------------- |
| SIMBa | 10000 | 249      | 05'20"     | 03"           |
| RNN   | 10000 | 99       | 03'35"     | 02"           |
