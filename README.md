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

## Linear RNN vs SIMBa
| n           | LR          | Max e       | Grad clip   |
| :---------: | :---------: | :---------: | :---------: |
| 2           | 1.00E-03    | 20000       | 100         |

**Best model performance:**
| Model       | Epoch       | Train loss  | Val loss    | Test loss   | Time        |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| SIMBa       | 1563        | 1.79E-02    | 3.89E-02    | 2.50E-01    | 50"         |
| RNN         | 3473        | 2.21E-02    | 7.90E-02    | 6.95E-01    | 01'20"      |

**Convergence speed performance:**
| Model       | Epsilon     | Epoch       | Train loss  | Val loss    | Test loss   | Time        |
| :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| SIMBa       | 7.50E-01    | 1           | 5.66E-02    | 1.08E-01    | 1.12E+00    | 00"         |
| RNN         | 7.50E-01    | 28          | 6.36E-01    | 8.23E-01    | 1.21E+00    | 00"         |
| SIMBa       | 5.00E-01    | 1           | 5.66E-02    | 1.08E-01    | 1.12E+00    | 00"         |
| RNN         | 5.00E-01    | 167         | 3.53E-01    | 5.50E-01    | 1.38E+00    | 03"         |
| SIMBa       | 2.50E-01    | 1           | 5.66E-02    | 1.08E-01    | 1.12E+00    | 00"         |
| RNN         | 2.50E-01    | 372         | 1.47E-01    | 2.75E-01    | 1.54E+00    | 08"         |
| SIMBa       | 1.00E-01    | 1           | 5.66E-02    | 1.08E-01    | 1.12E+00    | 00"         |
| RNN         | 1.00E-01    | 1906        | 2.54E-02    | 1.10E-01    | 1.20E+00    | 44"         |
| SIMBa       | 7.50E-02    | 194         | 2.91E-02    | 8.25E-02    | 1.10E+00    | 06"         |
| RNN         | 7.50E-02    | 2888        | 2.28E-02    | 8.25E-02    | 7.93E-01    | 01'07"      |
| SIMBa       | 5.00E-02    | 517         | 2.00E-02    | 5.50E-02    | 6.47E-01    | 16"         |
| RNN         | 5.00E-02    | --------    | --------    | --------    | --------    | --------    |

**Time and space performance for training:**
| Model       | max e       | # params    | Total time  | avg time/100e|
| :---------: | :---------: | :---------: | :---------: | :---------: |
| SIMBa       | 20000       | 53          | 11'07"      | 03"
| RNN         | 20000       | 35          | 07'44"      | 02"         |
______________________________________________________________________

