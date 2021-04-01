# BCI Sample Size Determination
A common problem for BCI studies is the determination of an appropriate sample size: how many subjects and trials are needed to achieve an acceptable performance given the resources available. While it may be difficult to  This code is a simulation-based sample size determination (SSD) for BCI studies
# BCI Sample Size Determination (SSD)

A common problem for BCI studies is the determination of an appropriate sample size: how many subjects and trials are needed to achieve an acceptable performance given the resources available? This project provides a simulation-based approach for SSD as described in [this paper ](https://doi.org/10.1109/EMBC.2016.7591000). Unlike the null hypothesis statistical testing framework which provides a p-value of the accuracy of the BCI, this approach uses a Bayesian new statistics framework which computes a 95% confidence interval of the accuracy of the BCI.

To use this tool, you would first want to determine an acceptable 95% confidence level (CI) for your BCI accuracy, in terms of the average length criterion (ALC) of the group level accuracy on the logit scale. For example, consider an ALC of 0.6. To interpret this number, we can look at the range between -0.3 to 0.3 in the logit function. This corresponds to roughly 30% on the probability scale. Since the logit scale has a non-linear mapping to the probabiltiy scale with the portion centered around 0 being the most linear, an ALC of 0.6 means that the CI will be at most 30% when the accuracy is roughly 50%, and narrower as the accuracy tends to 100%.

With the ALC CI defined, you can then specify different number of subjects (Ns) and trials (T) and run the code. From the results, you can determine which configuration meets the ALC criterion while minimizing Ns and T.

An example of the ALC plot is shown below:
![alt text](https://github.com/jason-leung/bci-ssd/blob/master/results/202103312159_alc_plot.png "ALC Plot")
