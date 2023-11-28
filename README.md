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
    ##     plot pi_mpb_killed pi_mpb_killed_pc   ndmi taswet tasgre tasbri
    ##    <dbl>         <dbl>            <dbl>  <dbl>  <dbl>  <dbl>  <dbl>
    ##  1    26          47.4             75.4 -0.166 -3653.  8290. 15922.
    ##  2    27          44.8             77.9 -0.163 -3627.  8214. 15823.
    ##  3    19          42.3             80.3 -0.156 -4057.  8474. 16402.
    ##  4     2          39.8             80.7 -0.153 -3478.  8063. 15531.
    ##  5     3          37.2             80.7 -0.152 -3576.  8188. 15796.
    ##  6    17          34.7             81.3 -0.140 -4135.  8744. 16855.
    ##  7    20          32.1             81.5 -0.139 -4821.  9452. 18099.
    ##  8    22          29.6             81.7 -0.138 -4460.  8701. 16967.
    ##  9    21          27.1             82.4 -0.133 -4374.  8755. 17018.
    ## 10    18          26.8             82.5 -0.131 -5081.  9102. 17824.
    ## # ℹ 18 more rows

## Derive NDMI predictors

``` r
beetle_ndmi <- lm(pi_mpb_killed_pc ~ ndmi, data = darkwoods_beetle_plots_data)
beetle_ndmi_residuals <- resid(beetle_ndmi)
summary(beetle_ndmi)
```

    ## 
    ## Call:
    ## lm(formula = pi_mpb_killed_pc ~ ndmi, data = darkwoods_beetle_plots_data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -6.7582 -1.5253  0.5376  2.0439  3.0846 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)  100.470      1.431   70.20  < 2e-16 ***
    ## ndmi         124.421     11.695   10.64 5.75e-11 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 2.384 on 26 degrees of freedom
    ## Multiple R-squared:  0.8132, Adjusted R-squared:  0.806 
    ## F-statistic: 113.2 on 1 and 26 DF,  p-value: 5.75e-11

``` r
plot(pi_mpb_killed_pc ~ ndmi, data = darkwoods_beetle_plots_data, 
     main=NULL, 
     ylab = "Percent of basal area of pine trees killed by MPB", xlab = "NDMI", col="blue") +
  abline(lm(pi_mpb_killed_pc ~ ndmi, data = darkwoods_beetle_plots_data), col = "red")
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
