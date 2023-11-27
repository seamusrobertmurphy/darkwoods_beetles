Remote sensing of Mountain pine beetle outbreaks; Darkwoods Conservation
Area
================
SMurphy
2020-11-02

## Import & tidy

Set seeds to “123”. Import excel data representing plot-sampled pixel
data from candidate spectral indices.

``` r
set.seed(123) 
darkwoods_beetle_plots_data <- read_excel("2.1.darkwoods_beetle_ground_plots.xlsx")
print(darkwoods_beetle_plots_data)
```

    ## # A tibble: 28 × 7
    ##     plot pi_mpb_killed `pi_mpb_killed%`    ndmi taswet tasgre tasbri
    ##    <dbl>         <dbl>            <dbl>   <dbl>  <dbl>  <dbl>  <dbl>
    ##  1    28          1.18             2.35 -0.0530 -3943.  8455. 16365 
    ##  2    27         44.8             89.7  -0.163  -3627.  8214. 15823.
    ##  3    26         47.4             94.7  -0.166  -3653.  8290. 15922.
    ##  4    25         10.5             20.9  -0.117  -3779.  8399. 16154.
    ##  5    24         12.4             24.9  -0.121  -4430.  8708. 16996.
    ##  6    23         15.7             31.4  -0.128  -3957.  8499. 16350.
    ##  7    22         29.6             59.2  -0.138  -4460.  8701. 16967.
    ##  8    21         27.1             54.1  -0.133  -4374.  8755. 17018.
    ##  9    20         32.1             64.3  -0.139  -4821.  9452. 18099.
    ## 10    19         42.3             84.6  -0.156  -4057.  8474. 16402.
    ## # ℹ 18 more rows

## Derive NDMI predictors

``` r
beetle_ndmi <- lm(pi_mpb_killed ~ ndmi, data = darkwoods_beetle_plots_data)
beetle_ndmi_residuals <- resid(beetle_ndmi)
summary(beetle_ndmi)
```

    ## 
    ## Call:
    ## lm(formula = pi_mpb_killed ~ ndmi, data = darkwoods_beetle_plots_data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -10.241  -8.130  -5.118   7.465  26.500 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  -16.066      5.997  -2.679   0.0126 *  
    ## ndmi        -294.942     49.005  -6.019 2.35e-06 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 9.991 on 26 degrees of freedom
    ## Multiple R-squared:  0.5822, Adjusted R-squared:  0.5661 
    ## F-statistic: 36.22 on 1 and 26 DF,  p-value: 2.346e-06

``` r
plot(pi_mpb_killed ~ ndmi, data = darkwoods_beetle_plots_data, 
     main=NULL, 
     ylab = "Basal area of pine trees killed by MPB", xlab = "NDMI", col="blue") +
  abline(lm(pi_mpb_killed ~ ndmi, data = darkwoods_beetle_plots_data), col = "red")
```

![](darkwoods_beetles_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

    ## integer(0)

## Train models

Splitting data 70:30 for training samples based on outcome variable.

``` r
beetle_training.samples <- createDataPartition(darkwoods_beetle_plots_data$pi_mpb_killed, p=0.70, list = FALSE)
beetle_train.data <- darkwoods_beetle_plots_data[beetle_training.samples, ]
beetle_test.data <- darkwoods_beetle_plots_data[-beetle_training.samples, ]
```

#### Model training method: time-series

``` r
model_training_time_series <- trainControl(method = "timeslice",
                                           initialWindow = 36,
                                           horizon = 12,
                                           fixedWindow = TRUE)
```

#### Model training method: 10K-Fold x10repeat

``` r
model_training_10kfold <- trainControl(method = "repeatedcv", 
                                       number = 10, repeats = 10)
```
