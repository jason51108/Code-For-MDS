# Code for MDS
This code repository is an open-source library designed for statisticians, especially those whose research focuses on MDS.

It provides a clean codebase to validate simulations in the cases of Binomial, Poisson, and Normal distributions, covering mainstream tasks in network data: **parameter estimation, classification, and matrix completion.**

## Usage

1. Install Python 3.8. For convenience, execute the following command.

```python
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well pre-processed datasets from Google, Then place the downloaded data in the folder`./dataset`.


3. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples

   ```bash
   # parameter_estimation
   bash ./scripts/parameter_estimation/Binomial.sh
   # classification
   bash ./scripts/classification/Poisson.sh
   # matrix_completion
   bash ./scripts/matrix_completion/Binomial.sh
   ```

4. Develop your own model.

   + Add the model file to the folder `./models`. You can follow the `./models/Binomial.py`.

   + Include the newly added model in the `Exp_Basic.model_dict` of `./exp/exp_basic.py`.

   + Create the corresponding scripts under the folder `./scripts`.

## Contact

If you have any questions or suggestions, feel free to contact:

- Yinghang Chen ([brainiaccc@foxmail.com]())


