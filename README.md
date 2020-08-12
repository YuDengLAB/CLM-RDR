# CLM-RDR
A Classification Model between cRBSs and the average Dynamic Range(CLM-RDR) is useful for predicting the dynamic range of biosensor designed by various cross-RBS(cRBS). The cRBS sequence dataset used to train CLM-RDR is rationally designed by analysis of variance (ANOVA) and WebLogo. CLM-RDR is useful for achieving the fine-tuning of biosensor.

## Update the overall code to support python3 and Update the model to enable processing of numerical features

# How to train?

1.put dataset in data floder

2.in cmd

```python
python3 training.py
```

and  results will be saved to runs/

# How to test and predict?

If you want to test with your own data, modify the test path in the conf/config.py file.

```python
python3 test.py
```

# How to plot?

You need to save the data to the .npy file and put it in the folder root directory.

```python
python3 data_result.py
```

# Prerequist

1. Tensorflow-gpu==1.9.0
2. sklearn, scipy, numpy, matplotlib
3. Python == 3.5
