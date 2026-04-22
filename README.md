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

## Extensions
I added:
- example.py which runs the main run of original SIMBa
- simba/linear_rnn.py which contains the linear recurrent neural network module
- edited simba/model.py to add times to the saved model
- rnn_run.py which contains functions for running and loading rnn model
- simba_run.py which contains functions for running and loading simba model
- main.py to do a simba and rnn run based on the given parameters (n, grad_clip, learning rate, max epoch, print_each, and whether to initialize simba or not)

## Linear RNN vs SIMBa (not initialized)
| n           | LR          | Max e       | Grad clip   |
| :---------: | :---------: | :---------: | :---------: |
| 2           | 1.00E-03    | 20000       | 100.0       |

**Best model performance:**
| Model       | Epoch       | Train loss  | Val loss    | Test loss   | Time        |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| SIMBa       | 8666        | 1.77E-02    | 3.88E-02    | 2.83E-01    | 04'29"      |
| RNN         | 1743        | 3.06E-02    | 6.60E-02    | 6.19E-01    | 36"         |

**Convergence speed performance:**
| Model       | Epsilon     | Epoch       | Train loss  | Val loss    | Test loss   | Time        |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| SIMBa       | 7.50E-01    | 850         | 6.54E-01    | 8.25E-01    | 3.21E+00    | 26"         |
| RNN         | 7.50E-01    | 55          | 7.20E-01    | 8.23E-01    | 1.27E+00    | 01"         |
| SIMBa       | 5.00E-01    | 1265        | 3.25E-01    | 5.50E-01    | 3.80E+00    | 38"         |
| RNN         | 5.00E-01    | 156         | 3.98E-01    | 5.48E-01    | 1.31E+00    | 03"         |
| SIMBa       | 2.50E-01    | 4708        | 8.86E-02    | 2.75E-01    | 3.03E+00    | 02'25"      |
| RNN         | 2.50E-01    | 460         | 1.48E-01    | 2.75E-01    | 1.44E+00    | 09"         |
| SIMBa       | 1.00E-01    | 5573        | 3.78E-02    | 1.10E-01    | 1.56E+00    | 02'51"      |
| RNN         | 1.00E-01    | 1246        | 5.86E-02    | 1.10E-01    | 5.87E-01    | 26"         |
| SIMBa       | 7.50E-02    | 5802        | 3.01E-02    | 8.25E-02    | 1.20E+00    | 02'58"      |
| RNN         | 7.50E-02    | 1405        | 4.29E-02    | 8.25E-02    | 4.69E-01    | 29"         |
| SIMBa       | 5.00E-02    | 6483        | 2.15E-02    | 5.50E-02    | 7.94E-01    | 03'19"      |
| RNN         | 5.00E-02    | --------    | --------    | --------    | --------    | --------    |

**Time and space performance for training:**
| Model       | max e       | # params    | Total time  | avg t/100e  |
| :---------: | :---------: | :---------: | :---------: | :---------: |
| SIMBa       | 20000       | 53          | 10'12"      | 03"         |
| RNN         | 20000       | 35          | 07'02"      | 02"         |
______________________________________________________________________

