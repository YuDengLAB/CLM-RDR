# DeepEDR
Deep-learning for enhancing the dynamic range of glucarate biosensor DeepEDR is useful for predicting the dynamic range of biosensor designed by various cross-RBS(cRBS). The cRBS sequence dataset used to train DeepEDR is rationally designed by analysis of variance (ANOVA) and WebLogo. DeepEDR is useful for achieving the fine-tuning of biosensor.

# How to train?

1.put dataset in data floder

2.in cmd

```python
python2 training.py
```

and  results will be saved to runs/

# How to test?

If you want to test with your own data, modify the test path in the conf/config.py file.

```python
python2 test.py
```

# How to plot?

You need to save the data to the .npy file and put it in the folder root directory.

```python
python3 data_result.py
```

# Prerequist

1. Tensorflow-gpu==1.4.0
2. sklearn, scipy, numpy, matplotlib
3. Python == 2.7