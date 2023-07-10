Lifestyle Project 2
================
Kristina Golden and Demetrios Samaras
2023-07-02

# Lifestyle

## Required Packages

``` r
library(tidyverse)
library(knitr)
library(GGally)
library(corrplot)
library(qwraps2)
library(vtable)
library(psych)
library(ggplot2)
library(cowplot)
library(caret)
library(gbm)
library(randomForest)
library(tree)
library(class)
library(bst)
library(reshape)
library(reshape2)
library(corrr)
library(ggcorrplot)
library(FactoMineR)
library(factoextra)
library(data.table)
```

## Introduction

In this report we will be looking at the Lifestyle data channel of the
online news popularity data set. This data set looks at a wide range of
variables from 39644 different news articles. The response variable that
we will be focusing on is **shares**. The purpose of this analysis is to
try to predict how many shares a Lifestyle article will get based on the
values of those other variables. We will be modeling shares using two
different linear regression models and two ensemble tree based models.

## Read in the Data

``` r
setwd("C:/Documents/Github/ST_558_Project_2")
#setwd("C:/Users/Demetri/Documents/NCSU_masters/ST558/Repos/GitHub/ST_558_Project_2")
 

online <- read.csv('OnlineNewsPopularity.csv')
colnames(online) <- c('url', 'days', 'n.Title', 'n.Content', 'Rate.Unique', 
                      'Rate.Nonstop', 'Rate.Unique.Nonstop', 'n.Links', 
                      'n.Other', 'n.Images', 'n.Videos',
                      'Avg.Words', 'n.Key', 'Lifestyle', 'Entertainment',
                      'Business', 'Social.Media', 'Tech', 'World', 'Min.Worst.Key',
                      'Max.Worst.Key', 'Avg.Worst.Key', 'Min.Best.Key', 
                      'Max.Best.Key', 'Avg.Best.Key', 'Avg.Min.Key', 'Avg.Max.Key',
                      'Avg.Avg.Key', 'Min.Ref', 'Max.Ref', 'Avg.Ref', 'Mon', 
                      'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun', 'Weekend',
                      'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03', 'LDA_04', 
                      'Global.Subj', 'Global.Pol', 'Global.Pos.Rate',
                      'Global.Neg.Rate', 'Rate.Pos', 'Rate.Neg', 'Avg.Pos.Pol',
                      'Min.Pos.Pol', 'Max.Pos.Pol', 'Avg.Neg.Pol', 'Min.Neg.Pol',
                      'Max.Neg.Pol', 'Title.Subj', 'Title.Pol', 'Abs.Subj',
                      'Abs.Pol', 'shares')
#Dropped url and timedelta because they are non-predictive. 
online <- online[ , c(3:61)]
```

## Write Functions

``` r
summary_table <- function(data_input) {
    min <- min(data_input$shares)
    q1 <- quantile(data_input$shares, 0.25)
    med <- median(data_input$shares)
    q3 <- quantile(data_input$shares, 0.75)
    max <- max(data_input$shares)
    mean1 <- mean(data_input$shares)
    sd1 <- sd(data_input$shares)
    data <- matrix(c(min, q1, med, q3, max, mean1, sd1), 
                   ncol=1)
    rownames(data) <- c("Minimum", "Q1", "Median", "Q3",
                           "Maximum", "Mean", "SD")
    colnames(data) <- c('Shares')
    data <- as.table(data)
    data
}
```

``` r
#Create correlation table and graph for a training dataset
correlation_table <- function(data_input) {
  #drop binary variables
  correlations <- cor(subset(data_input, select = c(2:4, 6:24,
                                                    33:50)))
  kable(correlations, caption = 'Correlations Lifestyle')
}
```

``` r
# Create correlation graph
correlation_graph <- function(data_input,sig=0.5){
  corr <- cor(subset(data_input, select = c(2:4, 6:24, 33:50)))
  corr[lower.tri(corr, diag = TRUE)] <- NA
  corr <- melt(corr, na.rm = TRUE)
  corr <- subset(corr, abs(value) > 0.5)
  corr[order(-abs(corr$value)),]
  print(corr)
  mtx_corr <- reshape2::acast(corr, Var1~Var2, value.var="value")
  corrplot(mtx_corr, is.corr=FALSE, tl.col="black", na.label=" ")
}
```

## Lifestyle EDA

### Lifestyle

``` r
## filters rows based on when parameter is 1 
data_channel <-  online %>% filter( !!rlang::sym(params$DataChannel) == 1)

## Drop the data_channel_is columns 
data_channel <- data_channel[ , -c(12:17)]

## reorder to put shares first 
data_channel <- data_channel[ , c(53, 1:52)]
```

``` r
set.seed(5432)

# Split the data into a training and test set (70/30 split)
# indices

train <- sample(1:nrow(data_channel), size = nrow(data_channel)*.70)
test <- setdiff(1:nrow(data_channel), train)

# training and testing subsets
data_channel_train <- data_channel[train, ]
data_channel_test <- data_channel[test, ]
```

## Lifestyle Summarizations

``` r
#Shares table for data_channel_train
summary_table(data_channel_train)
```

    ##            Shares
    ## Minimum     28.00
    ## Q1        1100.00
    ## Median    1700.00
    ## Q3        3200.00
    ## Maximum 208300.00
    ## Mean      3770.66
    ## SD        9802.02

The above table displays the Lifestyle 5-number summary for the shares.
It also includes the mean and standard deviation. Because the mean is
greater than the median, we suspect that the Lifestyle shares
distribution is right skewed.

``` r
#Correlation table for lifestyle_train
correlation_table(data_channel_train)
```

<table>
<caption>
Correlations Lifestyle
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
n.Title
</th>
<th style="text-align:right;">
n.Content
</th>
<th style="text-align:right;">
Rate.Unique
</th>
<th style="text-align:right;">
Rate.Unique.Nonstop
</th>
<th style="text-align:right;">
n.Links
</th>
<th style="text-align:right;">
n.Other
</th>
<th style="text-align:right;">
n.Images
</th>
<th style="text-align:right;">
n.Videos
</th>
<th style="text-align:right;">
Avg.Words
</th>
<th style="text-align:right;">
n.Key
</th>
<th style="text-align:right;">
Min.Worst.Key
</th>
<th style="text-align:right;">
Max.Worst.Key
</th>
<th style="text-align:right;">
Avg.Worst.Key
</th>
<th style="text-align:right;">
Min.Best.Key
</th>
<th style="text-align:right;">
Max.Best.Key
</th>
<th style="text-align:right;">
Avg.Best.Key
</th>
<th style="text-align:right;">
Avg.Min.Key
</th>
<th style="text-align:right;">
Avg.Max.Key
</th>
<th style="text-align:right;">
Avg.Avg.Key
</th>
<th style="text-align:right;">
Min.Ref
</th>
<th style="text-align:right;">
Max.Ref
</th>
<th style="text-align:right;">
Avg.Ref
</th>
<th style="text-align:right;">
LDA_00
</th>
<th style="text-align:right;">
LDA_01
</th>
<th style="text-align:right;">
LDA_02
</th>
<th style="text-align:right;">
LDA_03
</th>
<th style="text-align:right;">
LDA_04
</th>
<th style="text-align:right;">
Global.Subj
</th>
<th style="text-align:right;">
Global.Pol
</th>
<th style="text-align:right;">
Global.Pos.Rate
</th>
<th style="text-align:right;">
Global.Neg.Rate
</th>
<th style="text-align:right;">
Rate.Pos
</th>
<th style="text-align:right;">
Rate.Neg
</th>
<th style="text-align:right;">
Avg.Pos.Pol
</th>
<th style="text-align:right;">
Min.Pos.Pol
</th>
<th style="text-align:right;">
Max.Pos.Pol
</th>
<th style="text-align:right;">
Avg.Neg.Pol
</th>
<th style="text-align:right;">
Min.Neg.Pol
</th>
<th style="text-align:right;">
Max.Neg.Pol
</th>
<th style="text-align:right;">
Title.Subj
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
n.Title
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.011079
</td>
<td style="text-align:right;">
-0.029940
</td>
<td style="text-align:right;">
-0.029210
</td>
<td style="text-align:right;">
-0.110268
</td>
<td style="text-align:right;">
0.018499
</td>
<td style="text-align:right;">
-0.015207
</td>
<td style="text-align:right;">
-0.002968
</td>
<td style="text-align:right;">
-0.077994
</td>
<td style="text-align:right;">
-0.079886
</td>
<td style="text-align:right;">
-0.053087
</td>
<td style="text-align:right;">
0.022875
</td>
<td style="text-align:right;">
-0.006395
</td>
<td style="text-align:right;">
0.005347
</td>
<td style="text-align:right;">
0.074913
</td>
<td style="text-align:right;">
0.140916
</td>
<td style="text-align:right;">
-0.003285
</td>
<td style="text-align:right;">
-0.002632
</td>
<td style="text-align:right;">
0.016281
</td>
<td style="text-align:right;">
0.051233
</td>
<td style="text-align:right;">
0.031382
</td>
<td style="text-align:right;">
0.051214
</td>
<td style="text-align:right;">
-0.036639
</td>
<td style="text-align:right;">
0.082119
</td>
<td style="text-align:right;">
0.070854
</td>
<td style="text-align:right;">
-0.007787
</td>
<td style="text-align:right;">
-0.017102
</td>
<td style="text-align:right;">
-0.071070
</td>
<td style="text-align:right;">
-0.083425
</td>
<td style="text-align:right;">
-0.052356
</td>
<td style="text-align:right;">
-0.005521
</td>
<td style="text-align:right;">
-0.043062
</td>
<td style="text-align:right;">
0.002932
</td>
<td style="text-align:right;">
-0.119614
</td>
<td style="text-align:right;">
-0.049172
</td>
<td style="text-align:right;">
-0.051152
</td>
<td style="text-align:right;">
0.005163
</td>
<td style="text-align:right;">
0.005099
</td>
<td style="text-align:right;">
-0.014219
</td>
<td style="text-align:right;">
0.052428
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Content
</td>
<td style="text-align:right;">
0.011079
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.517923
</td>
<td style="text-align:right;">
-0.365188
</td>
<td style="text-align:right;">
0.301552
</td>
<td style="text-align:right;">
0.190009
</td>
<td style="text-align:right;">
0.487941
</td>
<td style="text-align:right;">
0.044104
</td>
<td style="text-align:right;">
0.029242
</td>
<td style="text-align:right;">
0.080111
</td>
<td style="text-align:right;">
-0.116425
</td>
<td style="text-align:right;">
0.016178
</td>
<td style="text-align:right;">
-0.007309
</td>
<td style="text-align:right;">
0.069030
</td>
<td style="text-align:right;">
0.113787
</td>
<td style="text-align:right;">
0.068772
</td>
<td style="text-align:right;">
0.088171
</td>
<td style="text-align:right;">
0.069445
</td>
<td style="text-align:right;">
0.112245
</td>
<td style="text-align:right;">
-0.000130
</td>
<td style="text-align:right;">
0.041956
</td>
<td style="text-align:right;">
0.025468
</td>
<td style="text-align:right;">
0.067821
</td>
<td style="text-align:right;">
-0.026198
</td>
<td style="text-align:right;">
-0.020044
</td>
<td style="text-align:right;">
0.076676
</td>
<td style="text-align:right;">
-0.093965
</td>
<td style="text-align:right;">
0.107761
</td>
<td style="text-align:right;">
0.083529
</td>
<td style="text-align:right;">
0.135087
</td>
<td style="text-align:right;">
0.042620
</td>
<td style="text-align:right;">
0.091849
</td>
<td style="text-align:right;">
-0.001836
</td>
<td style="text-align:right;">
0.094991
</td>
<td style="text-align:right;">
-0.260219
</td>
<td style="text-align:right;">
0.346296
</td>
<td style="text-align:right;">
-0.095444
</td>
<td style="text-align:right;">
-0.370576
</td>
<td style="text-align:right;">
0.232453
</td>
<td style="text-align:right;">
0.001182
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Unique
</td>
<td style="text-align:right;">
-0.029940
</td>
<td style="text-align:right;">
-0.517923
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.914854
</td>
<td style="text-align:right;">
-0.225292
</td>
<td style="text-align:right;">
-0.112118
</td>
<td style="text-align:right;">
-0.290798
</td>
<td style="text-align:right;">
0.009884
</td>
<td style="text-align:right;">
0.580602
</td>
<td style="text-align:right;">
-0.136942
</td>
<td style="text-align:right;">
0.098461
</td>
<td style="text-align:right;">
-0.045251
</td>
<td style="text-align:right;">
-0.013340
</td>
<td style="text-align:right;">
-0.057948
</td>
<td style="text-align:right;">
-0.089298
</td>
<td style="text-align:right;">
-0.093131
</td>
<td style="text-align:right;">
-0.030749
</td>
<td style="text-align:right;">
-0.061359
</td>
<td style="text-align:right;">
-0.108167
</td>
<td style="text-align:right;">
0.040123
</td>
<td style="text-align:right;">
0.002013
</td>
<td style="text-align:right;">
0.027682
</td>
<td style="text-align:right;">
-0.047395
</td>
<td style="text-align:right;">
0.043395
</td>
<td style="text-align:right;">
0.000154
</td>
<td style="text-align:right;">
-0.114924
</td>
<td style="text-align:right;">
0.105141
</td>
<td style="text-align:right;">
0.252409
</td>
<td style="text-align:right;">
0.013360
</td>
<td style="text-align:right;">
0.056693
</td>
<td style="text-align:right;">
0.115817
</td>
<td style="text-align:right;">
0.249518
</td>
<td style="text-align:right;">
0.184077
</td>
<td style="text-align:right;">
0.238482
</td>
<td style="text-align:right;">
0.396612
</td>
<td style="text-align:right;">
-0.082420
</td>
<td style="text-align:right;">
-0.114758
</td>
<td style="text-align:right;">
0.185637
</td>
<td style="text-align:right;">
-0.325942
</td>
<td style="text-align:right;">
-0.056535
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Unique.Nonstop
</td>
<td style="text-align:right;">
-0.029210
</td>
<td style="text-align:right;">
-0.365188
</td>
<td style="text-align:right;">
0.914854
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.270214
</td>
<td style="text-align:right;">
-0.088724
</td>
<td style="text-align:right;">
-0.386222
</td>
<td style="text-align:right;">
0.009262
</td>
<td style="text-align:right;">
0.601072
</td>
<td style="text-align:right;">
-0.167695
</td>
<td style="text-align:right;">
0.093919
</td>
<td style="text-align:right;">
-0.038626
</td>
<td style="text-align:right;">
-0.007540
</td>
<td style="text-align:right;">
-0.064248
</td>
<td style="text-align:right;">
-0.083177
</td>
<td style="text-align:right;">
-0.128024
</td>
<td style="text-align:right;">
-0.074120
</td>
<td style="text-align:right;">
-0.076958
</td>
<td style="text-align:right;">
-0.151022
</td>
<td style="text-align:right;">
0.021848
</td>
<td style="text-align:right;">
0.004134
</td>
<td style="text-align:right;">
0.018952
</td>
<td style="text-align:right;">
0.003465
</td>
<td style="text-align:right;">
0.037785
</td>
<td style="text-align:right;">
0.014646
</td>
<td style="text-align:right;">
-0.234805
</td>
<td style="text-align:right;">
0.141808
</td>
<td style="text-align:right;">
0.282397
</td>
<td style="text-align:right;">
0.004785
</td>
<td style="text-align:right;">
0.114151
</td>
<td style="text-align:right;">
0.174075
</td>
<td style="text-align:right;">
0.294599
</td>
<td style="text-align:right;">
0.221276
</td>
<td style="text-align:right;">
0.267111
</td>
<td style="text-align:right;">
0.292551
</td>
<td style="text-align:right;">
0.028125
</td>
<td style="text-align:right;">
-0.166273
</td>
<td style="text-align:right;">
0.037706
</td>
<td style="text-align:right;">
-0.246117
</td>
<td style="text-align:right;">
-0.091412
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Links
</td>
<td style="text-align:right;">
-0.110268
</td>
<td style="text-align:right;">
0.301552
</td>
<td style="text-align:right;">
-0.225292
</td>
<td style="text-align:right;">
-0.270214
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.273171
</td>
<td style="text-align:right;">
0.434763
</td>
<td style="text-align:right;">
0.052543
</td>
<td style="text-align:right;">
0.168497
</td>
<td style="text-align:right;">
0.212160
</td>
<td style="text-align:right;">
-0.155671
</td>
<td style="text-align:right;">
0.016270
</td>
<td style="text-align:right;">
-0.015846
</td>
<td style="text-align:right;">
0.145140
</td>
<td style="text-align:right;">
0.157879
</td>
<td style="text-align:right;">
0.142402
</td>
<td style="text-align:right;">
0.184287
</td>
<td style="text-align:right;">
0.136877
</td>
<td style="text-align:right;">
0.248898
</td>
<td style="text-align:right;">
0.020976
</td>
<td style="text-align:right;">
0.056935
</td>
<td style="text-align:right;">
0.046843
</td>
<td style="text-align:right;">
0.043129
</td>
<td style="text-align:right;">
-0.071811
</td>
<td style="text-align:right;">
-0.152193
</td>
<td style="text-align:right;">
0.292203
</td>
<td style="text-align:right;">
-0.158069
</td>
<td style="text-align:right;">
0.263668
</td>
<td style="text-align:right;">
0.213333
</td>
<td style="text-align:right;">
0.141634
</td>
<td style="text-align:right;">
-0.048647
</td>
<td style="text-align:right;">
0.153941
</td>
<td style="text-align:right;">
-0.067105
</td>
<td style="text-align:right;">
0.219458
</td>
<td style="text-align:right;">
-0.077682
</td>
<td style="text-align:right;">
0.323588
</td>
<td style="text-align:right;">
-0.106109
</td>
<td style="text-align:right;">
-0.187425
</td>
<td style="text-align:right;">
0.049850
</td>
<td style="text-align:right;">
0.080444
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Other
</td>
<td style="text-align:right;">
0.018499
</td>
<td style="text-align:right;">
0.190009
</td>
<td style="text-align:right;">
-0.112118
</td>
<td style="text-align:right;">
-0.088724
</td>
<td style="text-align:right;">
0.273171
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.192776
</td>
<td style="text-align:right;">
0.027553
</td>
<td style="text-align:right;">
0.057571
</td>
<td style="text-align:right;">
0.170006
</td>
<td style="text-align:right;">
0.091984
</td>
<td style="text-align:right;">
0.053587
</td>
<td style="text-align:right;">
0.063547
</td>
<td style="text-align:right;">
-0.019359
</td>
<td style="text-align:right;">
-0.117532
</td>
<td style="text-align:right;">
-0.142765
</td>
<td style="text-align:right;">
-0.028710
</td>
<td style="text-align:right;">
0.012927
</td>
<td style="text-align:right;">
-0.048721
</td>
<td style="text-align:right;">
-0.019669
</td>
<td style="text-align:right;">
0.144986
</td>
<td style="text-align:right;">
0.072596
</td>
<td style="text-align:right;">
-0.019453
</td>
<td style="text-align:right;">
0.070084
</td>
<td style="text-align:right;">
-0.121461
</td>
<td style="text-align:right;">
-0.017899
</td>
<td style="text-align:right;">
0.051394
</td>
<td style="text-align:right;">
0.060158
</td>
<td style="text-align:right;">
0.109793
</td>
<td style="text-align:right;">
0.092329
</td>
<td style="text-align:right;">
-0.089398
</td>
<td style="text-align:right;">
0.148856
</td>
<td style="text-align:right;">
-0.085263
</td>
<td style="text-align:right;">
0.034521
</td>
<td style="text-align:right;">
-0.089759
</td>
<td style="text-align:right;">
0.113006
</td>
<td style="text-align:right;">
-0.026830
</td>
<td style="text-align:right;">
-0.033832
</td>
<td style="text-align:right;">
0.023907
</td>
<td style="text-align:right;">
-0.006787
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Images
</td>
<td style="text-align:right;">
-0.015207
</td>
<td style="text-align:right;">
0.487941
</td>
<td style="text-align:right;">
-0.290798
</td>
<td style="text-align:right;">
-0.386222
</td>
<td style="text-align:right;">
0.434763
</td>
<td style="text-align:right;">
0.192776
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.056453
</td>
<td style="text-align:right;">
-0.052654
</td>
<td style="text-align:right;">
0.168653
</td>
<td style="text-align:right;">
-0.085235
</td>
<td style="text-align:right;">
-0.026204
</td>
<td style="text-align:right;">
-0.053350
</td>
<td style="text-align:right;">
0.113469
</td>
<td style="text-align:right;">
0.089591
</td>
<td style="text-align:right;">
0.148379
</td>
<td style="text-align:right;">
0.200345
</td>
<td style="text-align:right;">
0.105788
</td>
<td style="text-align:right;">
0.249893
</td>
<td style="text-align:right;">
0.048703
</td>
<td style="text-align:right;">
0.035712
</td>
<td style="text-align:right;">
0.048894
</td>
<td style="text-align:right;">
-0.068073
</td>
<td style="text-align:right;">
-0.045309
</td>
<td style="text-align:right;">
-0.125918
</td>
<td style="text-align:right;">
0.446937
</td>
<td style="text-align:right;">
-0.190265
</td>
<td style="text-align:right;">
0.190594
</td>
<td style="text-align:right;">
0.166575
</td>
<td style="text-align:right;">
0.071797
</td>
<td style="text-align:right;">
-0.037717
</td>
<td style="text-align:right;">
0.015938
</td>
<td style="text-align:right;">
-0.060429
</td>
<td style="text-align:right;">
0.168745
</td>
<td style="text-align:right;">
-0.024529
</td>
<td style="text-align:right;">
0.180650
</td>
<td style="text-align:right;">
-0.082057
</td>
<td style="text-align:right;">
-0.115728
</td>
<td style="text-align:right;">
0.044325
</td>
<td style="text-align:right;">
0.109149
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Videos
</td>
<td style="text-align:right;">
-0.002968
</td>
<td style="text-align:right;">
0.044104
</td>
<td style="text-align:right;">
0.009884
</td>
<td style="text-align:right;">
0.009262
</td>
<td style="text-align:right;">
0.052543
</td>
<td style="text-align:right;">
0.027553
</td>
<td style="text-align:right;">
-0.056453
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.006501
</td>
<td style="text-align:right;">
0.028262
</td>
<td style="text-align:right;">
-0.065031
</td>
<td style="text-align:right;">
0.063870
</td>
<td style="text-align:right;">
0.044159
</td>
<td style="text-align:right;">
0.039881
</td>
<td style="text-align:right;">
0.061721
</td>
<td style="text-align:right;">
0.133973
</td>
<td style="text-align:right;">
0.082204
</td>
<td style="text-align:right;">
0.040461
</td>
<td style="text-align:right;">
0.086569
</td>
<td style="text-align:right;">
0.044464
</td>
<td style="text-align:right;">
0.077394
</td>
<td style="text-align:right;">
0.090119
</td>
<td style="text-align:right;">
-0.018832
</td>
<td style="text-align:right;">
0.017744
</td>
<td style="text-align:right;">
0.015972
</td>
<td style="text-align:right;">
0.130918
</td>
<td style="text-align:right;">
-0.086545
</td>
<td style="text-align:right;">
0.029414
</td>
<td style="text-align:right;">
-0.002859
</td>
<td style="text-align:right;">
-0.008714
</td>
<td style="text-align:right;">
0.028579
</td>
<td style="text-align:right;">
-0.010584
</td>
<td style="text-align:right;">
0.013747
</td>
<td style="text-align:right;">
0.030988
</td>
<td style="text-align:right;">
0.039629
</td>
<td style="text-align:right;">
0.052229
</td>
<td style="text-align:right;">
-0.018207
</td>
<td style="text-align:right;">
-0.060470
</td>
<td style="text-align:right;">
0.025502
</td>
<td style="text-align:right;">
0.009568
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Words
</td>
<td style="text-align:right;">
-0.077994
</td>
<td style="text-align:right;">
0.029242
</td>
<td style="text-align:right;">
0.580602
</td>
<td style="text-align:right;">
0.601072
</td>
<td style="text-align:right;">
0.168497
</td>
<td style="text-align:right;">
0.057571
</td>
<td style="text-align:right;">
-0.052654
</td>
<td style="text-align:right;">
-0.006501
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.068216
</td>
<td style="text-align:right;">
-0.013787
</td>
<td style="text-align:right;">
-0.028914
</td>
<td style="text-align:right;">
-0.012399
</td>
<td style="text-align:right;">
-0.020097
</td>
<td style="text-align:right;">
0.017169
</td>
<td style="text-align:right;">
-0.035611
</td>
<td style="text-align:right;">
-0.009212
</td>
<td style="text-align:right;">
-0.013974
</td>
<td style="text-align:right;">
-0.046613
</td>
<td style="text-align:right;">
0.051334
</td>
<td style="text-align:right;">
0.049818
</td>
<td style="text-align:right;">
0.060141
</td>
<td style="text-align:right;">
0.001353
</td>
<td style="text-align:right;">
0.056982
</td>
<td style="text-align:right;">
0.032676
</td>
<td style="text-align:right;">
-0.117987
</td>
<td style="text-align:right;">
0.049741
</td>
<td style="text-align:right;">
0.464044
</td>
<td style="text-align:right;">
0.127310
</td>
<td style="text-align:right;">
0.247915
</td>
<td style="text-align:right;">
0.152107
</td>
<td style="text-align:right;">
0.491135
</td>
<td style="text-align:right;">
0.204513
</td>
<td style="text-align:right;">
0.399788
</td>
<td style="text-align:right;">
0.165102
</td>
<td style="text-align:right;">
0.318878
</td>
<td style="text-align:right;">
-0.221902
</td>
<td style="text-align:right;">
-0.138014
</td>
<td style="text-align:right;">
-0.165329
</td>
<td style="text-align:right;">
-0.075975
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Key
</td>
<td style="text-align:right;">
-0.079886
</td>
<td style="text-align:right;">
0.080111
</td>
<td style="text-align:right;">
-0.136942
</td>
<td style="text-align:right;">
-0.167695
</td>
<td style="text-align:right;">
0.212160
</td>
<td style="text-align:right;">
0.170006
</td>
<td style="text-align:right;">
0.168653
</td>
<td style="text-align:right;">
0.028262
</td>
<td style="text-align:right;">
-0.068216
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.010437
</td>
<td style="text-align:right;">
0.077386
</td>
<td style="text-align:right;">
0.063473
</td>
<td style="text-align:right;">
-0.212738
</td>
<td style="text-align:right;">
0.017499
</td>
<td style="text-align:right;">
-0.212235
</td>
<td style="text-align:right;">
-0.089258
</td>
<td style="text-align:right;">
0.112154
</td>
<td style="text-align:right;">
0.061740
</td>
<td style="text-align:right;">
-0.001279
</td>
<td style="text-align:right;">
0.049277
</td>
<td style="text-align:right;">
0.031007
</td>
<td style="text-align:right;">
-0.034641
</td>
<td style="text-align:right;">
-0.088296
</td>
<td style="text-align:right;">
-0.085985
</td>
<td style="text-align:right;">
0.154331
</td>
<td style="text-align:right;">
-0.016801
</td>
<td style="text-align:right;">
0.063724
</td>
<td style="text-align:right;">
0.104751
</td>
<td style="text-align:right;">
0.061560
</td>
<td style="text-align:right;">
-0.018958
</td>
<td style="text-align:right;">
0.007983
</td>
<td style="text-align:right;">
-0.057047
</td>
<td style="text-align:right;">
0.059867
</td>
<td style="text-align:right;">
-0.069078
</td>
<td style="text-align:right;">
0.103138
</td>
<td style="text-align:right;">
0.014380
</td>
<td style="text-align:right;">
-0.005001
</td>
<td style="text-align:right;">
0.038841
</td>
<td style="text-align:right;">
0.020747
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Worst.Key
</td>
<td style="text-align:right;">
-0.053087
</td>
<td style="text-align:right;">
-0.116425
</td>
<td style="text-align:right;">
0.098461
</td>
<td style="text-align:right;">
0.093919
</td>
<td style="text-align:right;">
-0.155671
</td>
<td style="text-align:right;">
0.091984
</td>
<td style="text-align:right;">
-0.085235
</td>
<td style="text-align:right;">
-0.065031
</td>
<td style="text-align:right;">
-0.013787
</td>
<td style="text-align:right;">
-0.010437
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.012374
</td>
<td style="text-align:right;">
0.086296
</td>
<td style="text-align:right;">
-0.148973
</td>
<td style="text-align:right;">
-0.850083
</td>
<td style="text-align:right;">
-0.662551
</td>
<td style="text-align:right;">
-0.215803
</td>
<td style="text-align:right;">
-0.155649
</td>
<td style="text-align:right;">
-0.356963
</td>
<td style="text-align:right;">
-0.101106
</td>
<td style="text-align:right;">
-0.071002
</td>
<td style="text-align:right;">
-0.106049
</td>
<td style="text-align:right;">
-0.125308
</td>
<td style="text-align:right;">
0.020869
</td>
<td style="text-align:right;">
-0.098723
</td>
<td style="text-align:right;">
-0.157904
</td>
<td style="text-align:right;">
0.244640
</td>
<td style="text-align:right;">
-0.050124
</td>
<td style="text-align:right;">
0.032080
</td>
<td style="text-align:right;">
0.021200
</td>
<td style="text-align:right;">
-0.069852
</td>
<td style="text-align:right;">
0.057372
</td>
<td style="text-align:right;">
-0.071500
</td>
<td style="text-align:right;">
-0.045392
</td>
<td style="text-align:right;">
0.033136
</td>
<td style="text-align:right;">
-0.125519
</td>
<td style="text-align:right;">
0.048509
</td>
<td style="text-align:right;">
0.118302
</td>
<td style="text-align:right;">
-0.025506
</td>
<td style="text-align:right;">
-0.033401
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Worst.Key
</td>
<td style="text-align:right;">
0.022875
</td>
<td style="text-align:right;">
0.016178
</td>
<td style="text-align:right;">
-0.045251
</td>
<td style="text-align:right;">
-0.038626
</td>
<td style="text-align:right;">
0.016270
</td>
<td style="text-align:right;">
0.053587
</td>
<td style="text-align:right;">
-0.026204
</td>
<td style="text-align:right;">
0.063870
</td>
<td style="text-align:right;">
-0.028914
</td>
<td style="text-align:right;">
0.077386
</td>
<td style="text-align:right;">
-0.012374
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.965551
</td>
<td style="text-align:right;">
-0.022810
</td>
<td style="text-align:right;">
0.013233
</td>
<td style="text-align:right;">
-0.017461
</td>
<td style="text-align:right;">
0.013764
</td>
<td style="text-align:right;">
0.639108
</td>
<td style="text-align:right;">
0.445614
</td>
<td style="text-align:right;">
0.150412
</td>
<td style="text-align:right;">
0.067218
</td>
<td style="text-align:right;">
0.124298
</td>
<td style="text-align:right;">
0.052834
</td>
<td style="text-align:right;">
-0.007945
</td>
<td style="text-align:right;">
-0.016931
</td>
<td style="text-align:right;">
0.044850
</td>
<td style="text-align:right;">
-0.066464
</td>
<td style="text-align:right;">
-0.045917
</td>
<td style="text-align:right;">
-0.022336
</td>
<td style="text-align:right;">
-0.029395
</td>
<td style="text-align:right;">
-0.025469
</td>
<td style="text-align:right;">
-0.014048
</td>
<td style="text-align:right;">
-0.015855
</td>
<td style="text-align:right;">
-0.029002
</td>
<td style="text-align:right;">
-0.027448
</td>
<td style="text-align:right;">
-0.004531
</td>
<td style="text-align:right;">
0.017267
</td>
<td style="text-align:right;">
0.005621
</td>
<td style="text-align:right;">
0.008545
</td>
<td style="text-align:right;">
0.005476
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Worst.Key
</td>
<td style="text-align:right;">
-0.006395
</td>
<td style="text-align:right;">
-0.007309
</td>
<td style="text-align:right;">
-0.013340
</td>
<td style="text-align:right;">
-0.007540
</td>
<td style="text-align:right;">
-0.015846
</td>
<td style="text-align:right;">
0.063547
</td>
<td style="text-align:right;">
-0.053350
</td>
<td style="text-align:right;">
0.044159
</td>
<td style="text-align:right;">
-0.012399
</td>
<td style="text-align:right;">
0.063473
</td>
<td style="text-align:right;">
0.086296
</td>
<td style="text-align:right;">
0.965551
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.068415
</td>
<td style="text-align:right;">
-0.088154
</td>
<td style="text-align:right;">
-0.140251
</td>
<td style="text-align:right;">
-0.036737
</td>
<td style="text-align:right;">
0.594928
</td>
<td style="text-align:right;">
0.371556
</td>
<td style="text-align:right;">
0.125573
</td>
<td style="text-align:right;">
0.051745
</td>
<td style="text-align:right;">
0.100539
</td>
<td style="text-align:right;">
0.046034
</td>
<td style="text-align:right;">
0.010544
</td>
<td style="text-align:right;">
-0.020724
</td>
<td style="text-align:right;">
0.017494
</td>
<td style="text-align:right;">
-0.046420
</td>
<td style="text-align:right;">
-0.054947
</td>
<td style="text-align:right;">
-0.025413
</td>
<td style="text-align:right;">
-0.032738
</td>
<td style="text-align:right;">
-0.029244
</td>
<td style="text-align:right;">
-0.003554
</td>
<td style="text-align:right;">
-0.014912
</td>
<td style="text-align:right;">
-0.031127
</td>
<td style="text-align:right;">
-0.020465
</td>
<td style="text-align:right;">
-0.024113
</td>
<td style="text-align:right;">
0.019431
</td>
<td style="text-align:right;">
0.021717
</td>
<td style="text-align:right;">
-0.001691
</td>
<td style="text-align:right;">
0.000130
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Best.Key
</td>
<td style="text-align:right;">
0.005347
</td>
<td style="text-align:right;">
0.069030
</td>
<td style="text-align:right;">
-0.057948
</td>
<td style="text-align:right;">
-0.064248
</td>
<td style="text-align:right;">
0.145140
</td>
<td style="text-align:right;">
-0.019359
</td>
<td style="text-align:right;">
0.113469
</td>
<td style="text-align:right;">
0.039881
</td>
<td style="text-align:right;">
-0.020097
</td>
<td style="text-align:right;">
-0.212738
</td>
<td style="text-align:right;">
-0.148973
</td>
<td style="text-align:right;">
-0.022810
</td>
<td style="text-align:right;">
-0.068415
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.167530
</td>
<td style="text-align:right;">
0.383823
</td>
<td style="text-align:right;">
0.612251
</td>
<td style="text-align:right;">
0.097407
</td>
<td style="text-align:right;">
0.337719
</td>
<td style="text-align:right;">
0.034276
</td>
<td style="text-align:right;">
0.021677
</td>
<td style="text-align:right;">
0.035863
</td>
<td style="text-align:right;">
-0.011945
</td>
<td style="text-align:right;">
-0.088291
</td>
<td style="text-align:right;">
-0.087854
</td>
<td style="text-align:right;">
0.226861
</td>
<td style="text-align:right;">
-0.085348
</td>
<td style="text-align:right;">
0.074054
</td>
<td style="text-align:right;">
0.012291
</td>
<td style="text-align:right;">
0.024746
</td>
<td style="text-align:right;">
0.032437
</td>
<td style="text-align:right;">
-0.015081
</td>
<td style="text-align:right;">
0.015350
</td>
<td style="text-align:right;">
0.037208
</td>
<td style="text-align:right;">
0.009350
</td>
<td style="text-align:right;">
0.076820
</td>
<td style="text-align:right;">
-0.018236
</td>
<td style="text-align:right;">
-0.060631
</td>
<td style="text-align:right;">
0.018276
</td>
<td style="text-align:right;">
0.015501
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Best.Key
</td>
<td style="text-align:right;">
0.074913
</td>
<td style="text-align:right;">
0.113787
</td>
<td style="text-align:right;">
-0.089298
</td>
<td style="text-align:right;">
-0.083177
</td>
<td style="text-align:right;">
0.157879
</td>
<td style="text-align:right;">
-0.117532
</td>
<td style="text-align:right;">
0.089591
</td>
<td style="text-align:right;">
0.061721
</td>
<td style="text-align:right;">
0.017169
</td>
<td style="text-align:right;">
0.017499
</td>
<td style="text-align:right;">
-0.850083
</td>
<td style="text-align:right;">
0.013233
</td>
<td style="text-align:right;">
-0.088154
</td>
<td style="text-align:right;">
0.167530
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.749163
</td>
<td style="text-align:right;">
0.247928
</td>
<td style="text-align:right;">
0.170934
</td>
<td style="text-align:right;">
0.416403
</td>
<td style="text-align:right;">
0.108904
</td>
<td style="text-align:right;">
0.067482
</td>
<td style="text-align:right;">
0.108826
</td>
<td style="text-align:right;">
0.123843
</td>
<td style="text-align:right;">
-0.028399
</td>
<td style="text-align:right;">
0.076690
</td>
<td style="text-align:right;">
0.156109
</td>
<td style="text-align:right;">
-0.231468
</td>
<td style="text-align:right;">
0.045996
</td>
<td style="text-align:right;">
-0.024840
</td>
<td style="text-align:right;">
-0.024481
</td>
<td style="text-align:right;">
0.061214
</td>
<td style="text-align:right;">
-0.048524
</td>
<td style="text-align:right;">
0.064830
</td>
<td style="text-align:right;">
0.041989
</td>
<td style="text-align:right;">
-0.025164
</td>
<td style="text-align:right;">
0.116384
</td>
<td style="text-align:right;">
-0.036005
</td>
<td style="text-align:right;">
-0.105902
</td>
<td style="text-align:right;">
0.019388
</td>
<td style="text-align:right;">
0.018912
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Best.Key
</td>
<td style="text-align:right;">
0.140916
</td>
<td style="text-align:right;">
0.068772
</td>
<td style="text-align:right;">
-0.093131
</td>
<td style="text-align:right;">
-0.128024
</td>
<td style="text-align:right;">
0.142402
</td>
<td style="text-align:right;">
-0.142765
</td>
<td style="text-align:right;">
0.148379
</td>
<td style="text-align:right;">
0.133973
</td>
<td style="text-align:right;">
-0.035611
</td>
<td style="text-align:right;">
-0.212235
</td>
<td style="text-align:right;">
-0.662551
</td>
<td style="text-align:right;">
-0.017461
</td>
<td style="text-align:right;">
-0.140251
</td>
<td style="text-align:right;">
0.383823
</td>
<td style="text-align:right;">
0.749163
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.416668
</td>
<td style="text-align:right;">
0.189141
</td>
<td style="text-align:right;">
0.514218
</td>
<td style="text-align:right;">
0.119556
</td>
<td style="text-align:right;">
0.070541
</td>
<td style="text-align:right;">
0.119509
</td>
<td style="text-align:right;">
0.068934
</td>
<td style="text-align:right;">
-0.083048
</td>
<td style="text-align:right;">
0.100136
</td>
<td style="text-align:right;">
0.326593
</td>
<td style="text-align:right;">
-0.294355
</td>
<td style="text-align:right;">
0.034629
</td>
<td style="text-align:right;">
-0.065584
</td>
<td style="text-align:right;">
-0.034977
</td>
<td style="text-align:right;">
0.103796
</td>
<td style="text-align:right;">
-0.122138
</td>
<td style="text-align:right;">
0.089942
</td>
<td style="text-align:right;">
0.016454
</td>
<td style="text-align:right;">
0.004064
</td>
<td style="text-align:right;">
0.070471
</td>
<td style="text-align:right;">
-0.049242
</td>
<td style="text-align:right;">
-0.115998
</td>
<td style="text-align:right;">
0.002891
</td>
<td style="text-align:right;">
0.064584
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Min.Key
</td>
<td style="text-align:right;">
-0.003285
</td>
<td style="text-align:right;">
0.088171
</td>
<td style="text-align:right;">
-0.030749
</td>
<td style="text-align:right;">
-0.074120
</td>
<td style="text-align:right;">
0.184287
</td>
<td style="text-align:right;">
-0.028710
</td>
<td style="text-align:right;">
0.200345
</td>
<td style="text-align:right;">
0.082204
</td>
<td style="text-align:right;">
-0.009212
</td>
<td style="text-align:right;">
-0.089258
</td>
<td style="text-align:right;">
-0.215803
</td>
<td style="text-align:right;">
0.013764
</td>
<td style="text-align:right;">
-0.036737
</td>
<td style="text-align:right;">
0.612251
</td>
<td style="text-align:right;">
0.247928
</td>
<td style="text-align:right;">
0.416668
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.122267
</td>
<td style="text-align:right;">
0.498282
</td>
<td style="text-align:right;">
0.093047
</td>
<td style="text-align:right;">
0.057706
</td>
<td style="text-align:right;">
0.092309
</td>
<td style="text-align:right;">
-0.010522
</td>
<td style="text-align:right;">
-0.110551
</td>
<td style="text-align:right;">
-0.078748
</td>
<td style="text-align:right;">
0.307811
</td>
<td style="text-align:right;">
-0.138731
</td>
<td style="text-align:right;">
0.131112
</td>
<td style="text-align:right;">
0.039754
</td>
<td style="text-align:right;">
0.025805
</td>
<td style="text-align:right;">
0.070792
</td>
<td style="text-align:right;">
-0.038080
</td>
<td style="text-align:right;">
0.045621
</td>
<td style="text-align:right;">
0.110124
</td>
<td style="text-align:right;">
0.072118
</td>
<td style="text-align:right;">
0.107947
</td>
<td style="text-align:right;">
-0.040603
</td>
<td style="text-align:right;">
-0.061484
</td>
<td style="text-align:right;">
-0.013208
</td>
<td style="text-align:right;">
0.030613
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Max.Key
</td>
<td style="text-align:right;">
-0.002632
</td>
<td style="text-align:right;">
0.069445
</td>
<td style="text-align:right;">
-0.061359
</td>
<td style="text-align:right;">
-0.076958
</td>
<td style="text-align:right;">
0.136877
</td>
<td style="text-align:right;">
0.012927
</td>
<td style="text-align:right;">
0.105788
</td>
<td style="text-align:right;">
0.040461
</td>
<td style="text-align:right;">
-0.013974
</td>
<td style="text-align:right;">
0.112154
</td>
<td style="text-align:right;">
-0.155649
</td>
<td style="text-align:right;">
0.639108
</td>
<td style="text-align:right;">
0.594928
</td>
<td style="text-align:right;">
0.097407
</td>
<td style="text-align:right;">
0.170934
</td>
<td style="text-align:right;">
0.189141
</td>
<td style="text-align:right;">
0.122267
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.817641
</td>
<td style="text-align:right;">
0.179458
</td>
<td style="text-align:right;">
0.075802
</td>
<td style="text-align:right;">
0.144548
</td>
<td style="text-align:right;">
0.044021
</td>
<td style="text-align:right;">
-0.033703
</td>
<td style="text-align:right;">
-0.057930
</td>
<td style="text-align:right;">
0.203884
</td>
<td style="text-align:right;">
-0.145392
</td>
<td style="text-align:right;">
0.029466
</td>
<td style="text-align:right;">
0.011261
</td>
<td style="text-align:right;">
-0.003502
</td>
<td style="text-align:right;">
-0.005212
</td>
<td style="text-align:right;">
-0.015637
</td>
<td style="text-align:right;">
-0.003008
</td>
<td style="text-align:right;">
0.029457
</td>
<td style="text-align:right;">
-0.020298
</td>
<td style="text-align:right;">
0.083018
</td>
<td style="text-align:right;">
-0.031817
</td>
<td style="text-align:right;">
-0.041631
</td>
<td style="text-align:right;">
-0.015512
</td>
<td style="text-align:right;">
0.046499
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Avg.Key
</td>
<td style="text-align:right;">
0.016281
</td>
<td style="text-align:right;">
0.112245
</td>
<td style="text-align:right;">
-0.108167
</td>
<td style="text-align:right;">
-0.151022
</td>
<td style="text-align:right;">
0.248898
</td>
<td style="text-align:right;">
-0.048721
</td>
<td style="text-align:right;">
0.249893
</td>
<td style="text-align:right;">
0.086569
</td>
<td style="text-align:right;">
-0.046613
</td>
<td style="text-align:right;">
0.061740
</td>
<td style="text-align:right;">
-0.356963
</td>
<td style="text-align:right;">
0.445614
</td>
<td style="text-align:right;">
0.371556
</td>
<td style="text-align:right;">
0.337719
</td>
<td style="text-align:right;">
0.416403
</td>
<td style="text-align:right;">
0.514218
</td>
<td style="text-align:right;">
0.498282
</td>
<td style="text-align:right;">
0.817641
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.224209
</td>
<td style="text-align:right;">
0.106617
</td>
<td style="text-align:right;">
0.193836
</td>
<td style="text-align:right;">
0.053653
</td>
<td style="text-align:right;">
-0.086750
</td>
<td style="text-align:right;">
-0.096096
</td>
<td style="text-align:right;">
0.437213
</td>
<td style="text-align:right;">
-0.283401
</td>
<td style="text-align:right;">
0.097491
</td>
<td style="text-align:right;">
0.024141
</td>
<td style="text-align:right;">
0.006295
</td>
<td style="text-align:right;">
0.058387
</td>
<td style="text-align:right;">
-0.072412
</td>
<td style="text-align:right;">
0.037013
</td>
<td style="text-align:right;">
0.084143
</td>
<td style="text-align:right;">
0.007961
</td>
<td style="text-align:right;">
0.135991
</td>
<td style="text-align:right;">
-0.063439
</td>
<td style="text-align:right;">
-0.098898
</td>
<td style="text-align:right;">
-0.013534
</td>
<td style="text-align:right;">
0.085847
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Ref
</td>
<td style="text-align:right;">
0.051233
</td>
<td style="text-align:right;">
-0.000130
</td>
<td style="text-align:right;">
0.040123
</td>
<td style="text-align:right;">
0.021848
</td>
<td style="text-align:right;">
0.020976
</td>
<td style="text-align:right;">
-0.019669
</td>
<td style="text-align:right;">
0.048703
</td>
<td style="text-align:right;">
0.044464
</td>
<td style="text-align:right;">
0.051334
</td>
<td style="text-align:right;">
-0.001279
</td>
<td style="text-align:right;">
-0.101106
</td>
<td style="text-align:right;">
0.150412
</td>
<td style="text-align:right;">
0.125573
</td>
<td style="text-align:right;">
0.034276
</td>
<td style="text-align:right;">
0.108904
</td>
<td style="text-align:right;">
0.119556
</td>
<td style="text-align:right;">
0.093047
</td>
<td style="text-align:right;">
0.179458
</td>
<td style="text-align:right;">
0.224209
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.415052
</td>
<td style="text-align:right;">
0.793982
</td>
<td style="text-align:right;">
-0.038419
</td>
<td style="text-align:right;">
-0.000891
</td>
<td style="text-align:right;">
0.002617
</td>
<td style="text-align:right;">
0.083729
</td>
<td style="text-align:right;">
-0.026384
</td>
<td style="text-align:right;">
0.056686
</td>
<td style="text-align:right;">
-0.022845
</td>
<td style="text-align:right;">
-0.038095
</td>
<td style="text-align:right;">
0.036779
</td>
<td style="text-align:right;">
-0.012112
</td>
<td style="text-align:right;">
0.058550
</td>
<td style="text-align:right;">
0.041341
</td>
<td style="text-align:right;">
0.023251
</td>
<td style="text-align:right;">
0.041362
</td>
<td style="text-align:right;">
-0.056921
</td>
<td style="text-align:right;">
-0.061575
</td>
<td style="text-align:right;">
-0.037369
</td>
<td style="text-align:right;">
0.025522
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Ref
</td>
<td style="text-align:right;">
0.031382
</td>
<td style="text-align:right;">
0.041956
</td>
<td style="text-align:right;">
0.002013
</td>
<td style="text-align:right;">
0.004134
</td>
<td style="text-align:right;">
0.056935
</td>
<td style="text-align:right;">
0.144986
</td>
<td style="text-align:right;">
0.035712
</td>
<td style="text-align:right;">
0.077394
</td>
<td style="text-align:right;">
0.049818
</td>
<td style="text-align:right;">
0.049277
</td>
<td style="text-align:right;">
-0.071002
</td>
<td style="text-align:right;">
0.067218
</td>
<td style="text-align:right;">
0.051745
</td>
<td style="text-align:right;">
0.021677
</td>
<td style="text-align:right;">
0.067482
</td>
<td style="text-align:right;">
0.070541
</td>
<td style="text-align:right;">
0.057706
</td>
<td style="text-align:right;">
0.075802
</td>
<td style="text-align:right;">
0.106617
</td>
<td style="text-align:right;">
0.415052
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.874362
</td>
<td style="text-align:right;">
0.017111
</td>
<td style="text-align:right;">
-0.004237
</td>
<td style="text-align:right;">
-0.021514
</td>
<td style="text-align:right;">
0.053999
</td>
<td style="text-align:right;">
-0.042299
</td>
<td style="text-align:right;">
0.050734
</td>
<td style="text-align:right;">
-0.016549
</td>
<td style="text-align:right;">
-0.007786
</td>
<td style="text-align:right;">
0.014211
</td>
<td style="text-align:right;">
0.007528
</td>
<td style="text-align:right;">
0.024261
</td>
<td style="text-align:right;">
0.019440
</td>
<td style="text-align:right;">
-0.029423
</td>
<td style="text-align:right;">
0.060195
</td>
<td style="text-align:right;">
-0.060411
</td>
<td style="text-align:right;">
-0.064054
</td>
<td style="text-align:right;">
-0.007519
</td>
<td style="text-align:right;">
0.026994
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Ref
</td>
<td style="text-align:right;">
0.051214
</td>
<td style="text-align:right;">
0.025468
</td>
<td style="text-align:right;">
0.027682
</td>
<td style="text-align:right;">
0.018952
</td>
<td style="text-align:right;">
0.046843
</td>
<td style="text-align:right;">
0.072596
</td>
<td style="text-align:right;">
0.048894
</td>
<td style="text-align:right;">
0.090119
</td>
<td style="text-align:right;">
0.060141
</td>
<td style="text-align:right;">
0.031007
</td>
<td style="text-align:right;">
-0.106049
</td>
<td style="text-align:right;">
0.124298
</td>
<td style="text-align:right;">
0.100539
</td>
<td style="text-align:right;">
0.035863
</td>
<td style="text-align:right;">
0.108826
</td>
<td style="text-align:right;">
0.119509
</td>
<td style="text-align:right;">
0.092309
</td>
<td style="text-align:right;">
0.144548
</td>
<td style="text-align:right;">
0.193836
</td>
<td style="text-align:right;">
0.793982
</td>
<td style="text-align:right;">
0.874362
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.008698
</td>
<td style="text-align:right;">
-0.002634
</td>
<td style="text-align:right;">
-0.012472
</td>
<td style="text-align:right;">
0.087184
</td>
<td style="text-align:right;">
-0.047506
</td>
<td style="text-align:right;">
0.066272
</td>
<td style="text-align:right;">
-0.027842
</td>
<td style="text-align:right;">
-0.031494
</td>
<td style="text-align:right;">
0.028565
</td>
<td style="text-align:right;">
-0.003838
</td>
<td style="text-align:right;">
0.050424
</td>
<td style="text-align:right;">
0.030493
</td>
<td style="text-align:right;">
-0.007290
</td>
<td style="text-align:right;">
0.057030
</td>
<td style="text-align:right;">
-0.072051
</td>
<td style="text-align:right;">
-0.074201
</td>
<td style="text-align:right;">
-0.029810
</td>
<td style="text-align:right;">
0.031699
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_00
</td>
<td style="text-align:right;">
-0.036639
</td>
<td style="text-align:right;">
0.067821
</td>
<td style="text-align:right;">
-0.047395
</td>
<td style="text-align:right;">
0.003465
</td>
<td style="text-align:right;">
0.043129
</td>
<td style="text-align:right;">
-0.019453
</td>
<td style="text-align:right;">
-0.068073
</td>
<td style="text-align:right;">
-0.018832
</td>
<td style="text-align:right;">
0.001353
</td>
<td style="text-align:right;">
-0.034641
</td>
<td style="text-align:right;">
-0.125308
</td>
<td style="text-align:right;">
0.052834
</td>
<td style="text-align:right;">
0.046034
</td>
<td style="text-align:right;">
-0.011945
</td>
<td style="text-align:right;">
0.123843
</td>
<td style="text-align:right;">
0.068934
</td>
<td style="text-align:right;">
-0.010522
</td>
<td style="text-align:right;">
0.044021
</td>
<td style="text-align:right;">
0.053653
</td>
<td style="text-align:right;">
-0.038419
</td>
<td style="text-align:right;">
0.017111
</td>
<td style="text-align:right;">
-0.008698
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.092470
</td>
<td style="text-align:right;">
-0.094577
</td>
<td style="text-align:right;">
-0.142541
</td>
<td style="text-align:right;">
-0.675299
</td>
<td style="text-align:right;">
0.015889
</td>
<td style="text-align:right;">
0.048371
</td>
<td style="text-align:right;">
0.055841
</td>
<td style="text-align:right;">
-0.006792
</td>
<td style="text-align:right;">
0.040688
</td>
<td style="text-align:right;">
-0.036118
</td>
<td style="text-align:right;">
0.046873
</td>
<td style="text-align:right;">
-0.050300
</td>
<td style="text-align:right;">
0.095242
</td>
<td style="text-align:right;">
-0.005972
</td>
<td style="text-align:right;">
-0.038599
</td>
<td style="text-align:right;">
0.018905
</td>
<td style="text-align:right;">
-0.023495
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_01
</td>
<td style="text-align:right;">
0.082119
</td>
<td style="text-align:right;">
-0.026198
</td>
<td style="text-align:right;">
0.043395
</td>
<td style="text-align:right;">
0.037785
</td>
<td style="text-align:right;">
-0.071811
</td>
<td style="text-align:right;">
0.070084
</td>
<td style="text-align:right;">
-0.045309
</td>
<td style="text-align:right;">
0.017744
</td>
<td style="text-align:right;">
0.056982
</td>
<td style="text-align:right;">
-0.088296
</td>
<td style="text-align:right;">
0.020869
</td>
<td style="text-align:right;">
-0.007945
</td>
<td style="text-align:right;">
0.010544
</td>
<td style="text-align:right;">
-0.088291
</td>
<td style="text-align:right;">
-0.028399
</td>
<td style="text-align:right;">
-0.083048
</td>
<td style="text-align:right;">
-0.110551
</td>
<td style="text-align:right;">
-0.033703
</td>
<td style="text-align:right;">
-0.086750
</td>
<td style="text-align:right;">
-0.000891
</td>
<td style="text-align:right;">
-0.004237
</td>
<td style="text-align:right;">
-0.002634
</td>
<td style="text-align:right;">
-0.092470
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.056873
</td>
<td style="text-align:right;">
-0.118521
</td>
<td style="text-align:right;">
-0.144415
</td>
<td style="text-align:right;">
-0.027224
</td>
<td style="text-align:right;">
-0.084199
</td>
<td style="text-align:right;">
-0.064257
</td>
<td style="text-align:right;">
0.035522
</td>
<td style="text-align:right;">
-0.031195
</td>
<td style="text-align:right;">
0.066319
</td>
<td style="text-align:right;">
-0.047845
</td>
<td style="text-align:right;">
0.026157
</td>
<td style="text-align:right;">
-0.034030
</td>
<td style="text-align:right;">
0.004204
</td>
<td style="text-align:right;">
0.031245
</td>
<td style="text-align:right;">
0.003944
</td>
<td style="text-align:right;">
0.009718
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_02
</td>
<td style="text-align:right;">
0.070854
</td>
<td style="text-align:right;">
-0.020044
</td>
<td style="text-align:right;">
0.000154
</td>
<td style="text-align:right;">
0.014646
</td>
<td style="text-align:right;">
-0.152193
</td>
<td style="text-align:right;">
-0.121461
</td>
<td style="text-align:right;">
-0.125918
</td>
<td style="text-align:right;">
0.015972
</td>
<td style="text-align:right;">
0.032676
</td>
<td style="text-align:right;">
-0.085985
</td>
<td style="text-align:right;">
-0.098723
</td>
<td style="text-align:right;">
-0.016931
</td>
<td style="text-align:right;">
-0.020724
</td>
<td style="text-align:right;">
-0.087854
</td>
<td style="text-align:right;">
0.076690
</td>
<td style="text-align:right;">
0.100136
</td>
<td style="text-align:right;">
-0.078748
</td>
<td style="text-align:right;">
-0.057930
</td>
<td style="text-align:right;">
-0.096096
</td>
<td style="text-align:right;">
0.002617
</td>
<td style="text-align:right;">
-0.021514
</td>
<td style="text-align:right;">
-0.012472
</td>
<td style="text-align:right;">
-0.094577
</td>
<td style="text-align:right;">
-0.056873
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.167550
</td>
<td style="text-align:right;">
-0.160588
</td>
<td style="text-align:right;">
-0.151909
</td>
<td style="text-align:right;">
-0.144264
</td>
<td style="text-align:right;">
-0.088670
</td>
<td style="text-align:right;">
0.064849
</td>
<td style="text-align:right;">
-0.084198
</td>
<td style="text-align:right;">
0.071397
</td>
<td style="text-align:right;">
-0.101531
</td>
<td style="text-align:right;">
-0.022562
</td>
<td style="text-align:right;">
-0.066246
</td>
<td style="text-align:right;">
-0.003322
</td>
<td style="text-align:right;">
-0.026164
</td>
<td style="text-align:right;">
0.034042
</td>
<td style="text-align:right;">
-0.059191
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_03
</td>
<td style="text-align:right;">
-0.007787
</td>
<td style="text-align:right;">
0.076676
</td>
<td style="text-align:right;">
-0.114924
</td>
<td style="text-align:right;">
-0.234805
</td>
<td style="text-align:right;">
0.292203
</td>
<td style="text-align:right;">
-0.017899
</td>
<td style="text-align:right;">
0.446937
</td>
<td style="text-align:right;">
0.130918
</td>
<td style="text-align:right;">
-0.117987
</td>
<td style="text-align:right;">
0.154331
</td>
<td style="text-align:right;">
-0.157904
</td>
<td style="text-align:right;">
0.044850
</td>
<td style="text-align:right;">
0.017494
</td>
<td style="text-align:right;">
0.226861
</td>
<td style="text-align:right;">
0.156109
</td>
<td style="text-align:right;">
0.326593
</td>
<td style="text-align:right;">
0.307811
</td>
<td style="text-align:right;">
0.203884
</td>
<td style="text-align:right;">
0.437213
</td>
<td style="text-align:right;">
0.083729
</td>
<td style="text-align:right;">
0.053999
</td>
<td style="text-align:right;">
0.087184
</td>
<td style="text-align:right;">
-0.142541
</td>
<td style="text-align:right;">
-0.118521
</td>
<td style="text-align:right;">
-0.167550
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.471086
</td>
<td style="text-align:right;">
0.146245
</td>
<td style="text-align:right;">
0.089903
</td>
<td style="text-align:right;">
0.019898
</td>
<td style="text-align:right;">
0.067604
</td>
<td style="text-align:right;">
-0.114315
</td>
<td style="text-align:right;">
0.020535
</td>
<td style="text-align:right;">
0.136240
</td>
<td style="text-align:right;">
0.063521
</td>
<td style="text-align:right;">
0.100775
</td>
<td style="text-align:right;">
-0.055925
</td>
<td style="text-align:right;">
-0.042612
</td>
<td style="text-align:right;">
-0.044215
</td>
<td style="text-align:right;">
0.131413
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_04
</td>
<td style="text-align:right;">
-0.017102
</td>
<td style="text-align:right;">
-0.093965
</td>
<td style="text-align:right;">
0.105141
</td>
<td style="text-align:right;">
0.141808
</td>
<td style="text-align:right;">
-0.158069
</td>
<td style="text-align:right;">
0.051394
</td>
<td style="text-align:right;">
-0.190265
</td>
<td style="text-align:right;">
-0.086545
</td>
<td style="text-align:right;">
0.049741
</td>
<td style="text-align:right;">
-0.016801
</td>
<td style="text-align:right;">
0.244640
</td>
<td style="text-align:right;">
-0.066464
</td>
<td style="text-align:right;">
-0.046420
</td>
<td style="text-align:right;">
-0.085348
</td>
<td style="text-align:right;">
-0.231468
</td>
<td style="text-align:right;">
-0.294355
</td>
<td style="text-align:right;">
-0.138731
</td>
<td style="text-align:right;">
-0.145392
</td>
<td style="text-align:right;">
-0.283401
</td>
<td style="text-align:right;">
-0.026384
</td>
<td style="text-align:right;">
-0.042299
</td>
<td style="text-align:right;">
-0.047506
</td>
<td style="text-align:right;">
-0.675299
</td>
<td style="text-align:right;">
-0.144415
</td>
<td style="text-align:right;">
-0.160588
</td>
<td style="text-align:right;">
-0.471086
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.048807
</td>
<td style="text-align:right;">
-0.021409
</td>
<td style="text-align:right;">
-0.006538
</td>
<td style="text-align:right;">
-0.076922
</td>
<td style="text-align:right;">
0.086631
</td>
<td style="text-align:right;">
-0.032202
</td>
<td style="text-align:right;">
-0.080047
</td>
<td style="text-align:right;">
-0.001798
</td>
<td style="text-align:right;">
-0.113814
</td>
<td style="text-align:right;">
0.043591
</td>
<td style="text-align:right;">
0.061522
</td>
<td style="text-align:right;">
0.000699
</td>
<td style="text-align:right;">
-0.052219
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Subj
</td>
<td style="text-align:right;">
-0.071070
</td>
<td style="text-align:right;">
0.107761
</td>
<td style="text-align:right;">
0.252409
</td>
<td style="text-align:right;">
0.282397
</td>
<td style="text-align:right;">
0.263668
</td>
<td style="text-align:right;">
0.060158
</td>
<td style="text-align:right;">
0.190594
</td>
<td style="text-align:right;">
0.029414
</td>
<td style="text-align:right;">
0.464044
</td>
<td style="text-align:right;">
0.063724
</td>
<td style="text-align:right;">
-0.050124
</td>
<td style="text-align:right;">
-0.045917
</td>
<td style="text-align:right;">
-0.054947
</td>
<td style="text-align:right;">
0.074054
</td>
<td style="text-align:right;">
0.045996
</td>
<td style="text-align:right;">
0.034629
</td>
<td style="text-align:right;">
0.131112
</td>
<td style="text-align:right;">
0.029466
</td>
<td style="text-align:right;">
0.097491
</td>
<td style="text-align:right;">
0.056686
</td>
<td style="text-align:right;">
0.050734
</td>
<td style="text-align:right;">
0.066272
</td>
<td style="text-align:right;">
0.015889
</td>
<td style="text-align:right;">
-0.027224
</td>
<td style="text-align:right;">
-0.151909
</td>
<td style="text-align:right;">
0.146245
</td>
<td style="text-align:right;">
-0.048807
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.378718
</td>
<td style="text-align:right;">
0.392313
</td>
<td style="text-align:right;">
0.160576
</td>
<td style="text-align:right;">
0.358805
</td>
<td style="text-align:right;">
0.071730
</td>
<td style="text-align:right;">
0.601449
</td>
<td style="text-align:right;">
0.186679
</td>
<td style="text-align:right;">
0.444479
</td>
<td style="text-align:right;">
-0.350954
</td>
<td style="text-align:right;">
-0.290406
</td>
<td style="text-align:right;">
-0.107613
</td>
<td style="text-align:right;">
0.126064
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Pol
</td>
<td style="text-align:right;">
-0.083425
</td>
<td style="text-align:right;">
0.083529
</td>
<td style="text-align:right;">
0.013360
</td>
<td style="text-align:right;">
0.004785
</td>
<td style="text-align:right;">
0.213333
</td>
<td style="text-align:right;">
0.109793
</td>
<td style="text-align:right;">
0.166575
</td>
<td style="text-align:right;">
-0.002859
</td>
<td style="text-align:right;">
0.127310
</td>
<td style="text-align:right;">
0.104751
</td>
<td style="text-align:right;">
0.032080
</td>
<td style="text-align:right;">
-0.022336
</td>
<td style="text-align:right;">
-0.025413
</td>
<td style="text-align:right;">
0.012291
</td>
<td style="text-align:right;">
-0.024840
</td>
<td style="text-align:right;">
-0.065584
</td>
<td style="text-align:right;">
0.039754
</td>
<td style="text-align:right;">
0.011261
</td>
<td style="text-align:right;">
0.024141
</td>
<td style="text-align:right;">
-0.022845
</td>
<td style="text-align:right;">
-0.016549
</td>
<td style="text-align:right;">
-0.027842
</td>
<td style="text-align:right;">
0.048371
</td>
<td style="text-align:right;">
-0.084199
</td>
<td style="text-align:right;">
-0.144264
</td>
<td style="text-align:right;">
0.089903
</td>
<td style="text-align:right;">
-0.021409
</td>
<td style="text-align:right;">
0.378718
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.589506
</td>
<td style="text-align:right;">
-0.443352
</td>
<td style="text-align:right;">
0.711757
</td>
<td style="text-align:right;">
-0.658267
</td>
<td style="text-align:right;">
0.555767
</td>
<td style="text-align:right;">
0.038496
</td>
<td style="text-align:right;">
0.468377
</td>
<td style="text-align:right;">
0.220524
</td>
<td style="text-align:right;">
0.243814
</td>
<td style="text-align:right;">
-0.019148
</td>
<td style="text-align:right;">
0.081061
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Pos.Rate
</td>
<td style="text-align:right;">
-0.052356
</td>
<td style="text-align:right;">
0.135087
</td>
<td style="text-align:right;">
0.056693
</td>
<td style="text-align:right;">
0.114151
</td>
<td style="text-align:right;">
0.141634
</td>
<td style="text-align:right;">
0.092329
</td>
<td style="text-align:right;">
0.071797
</td>
<td style="text-align:right;">
-0.008714
</td>
<td style="text-align:right;">
0.247915
</td>
<td style="text-align:right;">
0.061560
</td>
<td style="text-align:right;">
0.021200
</td>
<td style="text-align:right;">
-0.029395
</td>
<td style="text-align:right;">
-0.032738
</td>
<td style="text-align:right;">
0.024746
</td>
<td style="text-align:right;">
-0.024481
</td>
<td style="text-align:right;">
-0.034977
</td>
<td style="text-align:right;">
0.025805
</td>
<td style="text-align:right;">
-0.003502
</td>
<td style="text-align:right;">
0.006295
</td>
<td style="text-align:right;">
-0.038095
</td>
<td style="text-align:right;">
-0.007786
</td>
<td style="text-align:right;">
-0.031494
</td>
<td style="text-align:right;">
0.055841
</td>
<td style="text-align:right;">
-0.064257
</td>
<td style="text-align:right;">
-0.088670
</td>
<td style="text-align:right;">
0.019898
</td>
<td style="text-align:right;">
-0.006538
</td>
<td style="text-align:right;">
0.392313
</td>
<td style="text-align:right;">
0.589506
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.047619
</td>
<td style="text-align:right;">
0.572519
</td>
<td style="text-align:right;">
-0.379184
</td>
<td style="text-align:right;">
0.264602
</td>
<td style="text-align:right;">
-0.168665
</td>
<td style="text-align:right;">
0.429378
</td>
<td style="text-align:right;">
-0.049286
</td>
<td style="text-align:right;">
-0.107028
</td>
<td style="text-align:right;">
0.003597
</td>
<td style="text-align:right;">
0.123285
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Neg.Rate
</td>
<td style="text-align:right;">
-0.005521
</td>
<td style="text-align:right;">
0.042620
</td>
<td style="text-align:right;">
0.115817
</td>
<td style="text-align:right;">
0.174075
</td>
<td style="text-align:right;">
-0.048647
</td>
<td style="text-align:right;">
-0.089398
</td>
<td style="text-align:right;">
-0.037717
</td>
<td style="text-align:right;">
0.028579
</td>
<td style="text-align:right;">
0.152107
</td>
<td style="text-align:right;">
-0.018958
</td>
<td style="text-align:right;">
-0.069852
</td>
<td style="text-align:right;">
-0.025469
</td>
<td style="text-align:right;">
-0.029244
</td>
<td style="text-align:right;">
0.032437
</td>
<td style="text-align:right;">
0.061214
</td>
<td style="text-align:right;">
0.103796
</td>
<td style="text-align:right;">
0.070792
</td>
<td style="text-align:right;">
-0.005212
</td>
<td style="text-align:right;">
0.058387
</td>
<td style="text-align:right;">
0.036779
</td>
<td style="text-align:right;">
0.014211
</td>
<td style="text-align:right;">
0.028565
</td>
<td style="text-align:right;">
-0.006792
</td>
<td style="text-align:right;">
0.035522
</td>
<td style="text-align:right;">
0.064849
</td>
<td style="text-align:right;">
0.067604
</td>
<td style="text-align:right;">
-0.076922
</td>
<td style="text-align:right;">
0.160576
</td>
<td style="text-align:right;">
-0.443352
</td>
<td style="text-align:right;">
0.047619
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.551535
</td>
<td style="text-align:right;">
0.818606
</td>
<td style="text-align:right;">
0.132955
</td>
<td style="text-align:right;">
0.058151
</td>
<td style="text-align:right;">
0.122822
</td>
<td style="text-align:right;">
-0.199870
</td>
<td style="text-align:right;">
-0.412537
</td>
<td style="text-align:right;">
0.190981
</td>
<td style="text-align:right;">
0.032759
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Pos
</td>
<td style="text-align:right;">
-0.043062
</td>
<td style="text-align:right;">
0.091849
</td>
<td style="text-align:right;">
0.249518
</td>
<td style="text-align:right;">
0.294599
</td>
<td style="text-align:right;">
0.153941
</td>
<td style="text-align:right;">
0.148856
</td>
<td style="text-align:right;">
0.015938
</td>
<td style="text-align:right;">
-0.010584
</td>
<td style="text-align:right;">
0.491135
</td>
<td style="text-align:right;">
0.007983
</td>
<td style="text-align:right;">
0.057372
</td>
<td style="text-align:right;">
-0.014048
</td>
<td style="text-align:right;">
-0.003554
</td>
<td style="text-align:right;">
-0.015081
</td>
<td style="text-align:right;">
-0.048524
</td>
<td style="text-align:right;">
-0.122138
</td>
<td style="text-align:right;">
-0.038080
</td>
<td style="text-align:right;">
-0.015637
</td>
<td style="text-align:right;">
-0.072412
</td>
<td style="text-align:right;">
-0.012112
</td>
<td style="text-align:right;">
0.007528
</td>
<td style="text-align:right;">
-0.003838
</td>
<td style="text-align:right;">
0.040688
</td>
<td style="text-align:right;">
-0.031195
</td>
<td style="text-align:right;">
-0.084198
</td>
<td style="text-align:right;">
-0.114315
</td>
<td style="text-align:right;">
0.086631
</td>
<td style="text-align:right;">
0.358805
</td>
<td style="text-align:right;">
0.711757
</td>
<td style="text-align:right;">
0.572519
</td>
<td style="text-align:right;">
-0.551535
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.687246
</td>
<td style="text-align:right;">
0.323857
</td>
<td style="text-align:right;">
-0.027513
</td>
<td style="text-align:right;">
0.374079
</td>
<td style="text-align:right;">
0.004948
</td>
<td style="text-align:right;">
0.136271
</td>
<td style="text-align:right;">
-0.184907
</td>
<td style="text-align:right;">
-0.012643
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Neg
</td>
<td style="text-align:right;">
0.002932
</td>
<td style="text-align:right;">
-0.001836
</td>
<td style="text-align:right;">
0.184077
</td>
<td style="text-align:right;">
0.221276
</td>
<td style="text-align:right;">
-0.067105
</td>
<td style="text-align:right;">
-0.085263
</td>
<td style="text-align:right;">
-0.060429
</td>
<td style="text-align:right;">
0.013747
</td>
<td style="text-align:right;">
0.204513
</td>
<td style="text-align:right;">
-0.057047
</td>
<td style="text-align:right;">
-0.071500
</td>
<td style="text-align:right;">
-0.015855
</td>
<td style="text-align:right;">
-0.014912
</td>
<td style="text-align:right;">
0.015350
</td>
<td style="text-align:right;">
0.064830
</td>
<td style="text-align:right;">
0.089942
</td>
<td style="text-align:right;">
0.045621
</td>
<td style="text-align:right;">
-0.003008
</td>
<td style="text-align:right;">
0.037013
</td>
<td style="text-align:right;">
0.058550
</td>
<td style="text-align:right;">
0.024261
</td>
<td style="text-align:right;">
0.050424
</td>
<td style="text-align:right;">
-0.036118
</td>
<td style="text-align:right;">
0.066319
</td>
<td style="text-align:right;">
0.071397
</td>
<td style="text-align:right;">
0.020535
</td>
<td style="text-align:right;">
-0.032202
</td>
<td style="text-align:right;">
0.071730
</td>
<td style="text-align:right;">
-0.658267
</td>
<td style="text-align:right;">
-0.379184
</td>
<td style="text-align:right;">
0.818606
</td>
<td style="text-align:right;">
-0.687246
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.061661
</td>
<td style="text-align:right;">
0.174419
</td>
<td style="text-align:right;">
-0.049452
</td>
<td style="text-align:right;">
-0.243744
</td>
<td style="text-align:right;">
-0.358346
</td>
<td style="text-align:right;">
0.096580
</td>
<td style="text-align:right;">
-0.048362
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Pos.Pol
</td>
<td style="text-align:right;">
-0.119614
</td>
<td style="text-align:right;">
0.094991
</td>
<td style="text-align:right;">
0.238482
</td>
<td style="text-align:right;">
0.267111
</td>
<td style="text-align:right;">
0.219458
</td>
<td style="text-align:right;">
0.034521
</td>
<td style="text-align:right;">
0.168745
</td>
<td style="text-align:right;">
0.030988
</td>
<td style="text-align:right;">
0.399788
</td>
<td style="text-align:right;">
0.059867
</td>
<td style="text-align:right;">
-0.045392
</td>
<td style="text-align:right;">
-0.029002
</td>
<td style="text-align:right;">
-0.031127
</td>
<td style="text-align:right;">
0.037208
</td>
<td style="text-align:right;">
0.041989
</td>
<td style="text-align:right;">
0.016454
</td>
<td style="text-align:right;">
0.110124
</td>
<td style="text-align:right;">
0.029457
</td>
<td style="text-align:right;">
0.084143
</td>
<td style="text-align:right;">
0.041341
</td>
<td style="text-align:right;">
0.019440
</td>
<td style="text-align:right;">
0.030493
</td>
<td style="text-align:right;">
0.046873
</td>
<td style="text-align:right;">
-0.047845
</td>
<td style="text-align:right;">
-0.101531
</td>
<td style="text-align:right;">
0.136240
</td>
<td style="text-align:right;">
-0.080047
</td>
<td style="text-align:right;">
0.601449
</td>
<td style="text-align:right;">
0.555767
</td>
<td style="text-align:right;">
0.264602
</td>
<td style="text-align:right;">
0.132955
</td>
<td style="text-align:right;">
0.323857
</td>
<td style="text-align:right;">
0.061661
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.389497
</td>
<td style="text-align:right;">
0.635302
</td>
<td style="text-align:right;">
-0.125538
</td>
<td style="text-align:right;">
-0.123271
</td>
<td style="text-align:right;">
-0.039344
</td>
<td style="text-align:right;">
0.043027
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Pos.Pol
</td>
<td style="text-align:right;">
-0.049172
</td>
<td style="text-align:right;">
-0.260219
</td>
<td style="text-align:right;">
0.396612
</td>
<td style="text-align:right;">
0.292551
</td>
<td style="text-align:right;">
-0.077682
</td>
<td style="text-align:right;">
-0.089759
</td>
<td style="text-align:right;">
-0.024529
</td>
<td style="text-align:right;">
0.039629
</td>
<td style="text-align:right;">
0.165102
</td>
<td style="text-align:right;">
-0.069078
</td>
<td style="text-align:right;">
0.033136
</td>
<td style="text-align:right;">
-0.027448
</td>
<td style="text-align:right;">
-0.020465
</td>
<td style="text-align:right;">
0.009350
</td>
<td style="text-align:right;">
-0.025164
</td>
<td style="text-align:right;">
0.004064
</td>
<td style="text-align:right;">
0.072118
</td>
<td style="text-align:right;">
-0.020298
</td>
<td style="text-align:right;">
0.007961
</td>
<td style="text-align:right;">
0.023251
</td>
<td style="text-align:right;">
-0.029423
</td>
<td style="text-align:right;">
-0.007290
</td>
<td style="text-align:right;">
-0.050300
</td>
<td style="text-align:right;">
0.026157
</td>
<td style="text-align:right;">
-0.022562
</td>
<td style="text-align:right;">
0.063521
</td>
<td style="text-align:right;">
-0.001798
</td>
<td style="text-align:right;">
0.186679
</td>
<td style="text-align:right;">
0.038496
</td>
<td style="text-align:right;">
-0.168665
</td>
<td style="text-align:right;">
0.058151
</td>
<td style="text-align:right;">
-0.027513
</td>
<td style="text-align:right;">
0.174419
</td>
<td style="text-align:right;">
0.389497
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.088157
</td>
<td style="text-align:right;">
0.034949
</td>
<td style="text-align:right;">
0.180394
</td>
<td style="text-align:right;">
-0.129087
</td>
<td style="text-align:right;">
-0.018127
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Pos.Pol
</td>
<td style="text-align:right;">
-0.051152
</td>
<td style="text-align:right;">
0.346296
</td>
<td style="text-align:right;">
-0.082420
</td>
<td style="text-align:right;">
0.028125
</td>
<td style="text-align:right;">
0.323588
</td>
<td style="text-align:right;">
0.113006
</td>
<td style="text-align:right;">
0.180650
</td>
<td style="text-align:right;">
0.052229
</td>
<td style="text-align:right;">
0.318878
</td>
<td style="text-align:right;">
0.103138
</td>
<td style="text-align:right;">
-0.125519
</td>
<td style="text-align:right;">
-0.004531
</td>
<td style="text-align:right;">
-0.024113
</td>
<td style="text-align:right;">
0.076820
</td>
<td style="text-align:right;">
0.116384
</td>
<td style="text-align:right;">
0.070471
</td>
<td style="text-align:right;">
0.107947
</td>
<td style="text-align:right;">
0.083018
</td>
<td style="text-align:right;">
0.135991
</td>
<td style="text-align:right;">
0.041362
</td>
<td style="text-align:right;">
0.060195
</td>
<td style="text-align:right;">
0.057030
</td>
<td style="text-align:right;">
0.095242
</td>
<td style="text-align:right;">
-0.034030
</td>
<td style="text-align:right;">
-0.066246
</td>
<td style="text-align:right;">
0.100775
</td>
<td style="text-align:right;">
-0.113814
</td>
<td style="text-align:right;">
0.444479
</td>
<td style="text-align:right;">
0.468377
</td>
<td style="text-align:right;">
0.429378
</td>
<td style="text-align:right;">
0.122822
</td>
<td style="text-align:right;">
0.374079
</td>
<td style="text-align:right;">
-0.049452
</td>
<td style="text-align:right;">
0.635302
</td>
<td style="text-align:right;">
-0.088157
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.147988
</td>
<td style="text-align:right;">
-0.313449
</td>
<td style="text-align:right;">
0.104479
</td>
<td style="text-align:right;">
0.044484
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Neg.Pol
</td>
<td style="text-align:right;">
0.005163
</td>
<td style="text-align:right;">
-0.095444
</td>
<td style="text-align:right;">
-0.114758
</td>
<td style="text-align:right;">
-0.166273
</td>
<td style="text-align:right;">
-0.106109
</td>
<td style="text-align:right;">
-0.026830
</td>
<td style="text-align:right;">
-0.082057
</td>
<td style="text-align:right;">
-0.018207
</td>
<td style="text-align:right;">
-0.221902
</td>
<td style="text-align:right;">
0.014380
</td>
<td style="text-align:right;">
0.048509
</td>
<td style="text-align:right;">
0.017267
</td>
<td style="text-align:right;">
0.019431
</td>
<td style="text-align:right;">
-0.018236
</td>
<td style="text-align:right;">
-0.036005
</td>
<td style="text-align:right;">
-0.049242
</td>
<td style="text-align:right;">
-0.040603
</td>
<td style="text-align:right;">
-0.031817
</td>
<td style="text-align:right;">
-0.063439
</td>
<td style="text-align:right;">
-0.056921
</td>
<td style="text-align:right;">
-0.060411
</td>
<td style="text-align:right;">
-0.072051
</td>
<td style="text-align:right;">
-0.005972
</td>
<td style="text-align:right;">
0.004204
</td>
<td style="text-align:right;">
-0.003322
</td>
<td style="text-align:right;">
-0.055925
</td>
<td style="text-align:right;">
0.043591
</td>
<td style="text-align:right;">
-0.350954
</td>
<td style="text-align:right;">
0.220524
</td>
<td style="text-align:right;">
-0.049286
</td>
<td style="text-align:right;">
-0.199870
</td>
<td style="text-align:right;">
0.004948
</td>
<td style="text-align:right;">
-0.243744
</td>
<td style="text-align:right;">
-0.125538
</td>
<td style="text-align:right;">
0.034949
</td>
<td style="text-align:right;">
-0.147988
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.696680
</td>
<td style="text-align:right;">
0.550389
</td>
<td style="text-align:right;">
-0.001598
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Neg.Pol
</td>
<td style="text-align:right;">
0.005099
</td>
<td style="text-align:right;">
-0.370576
</td>
<td style="text-align:right;">
0.185637
</td>
<td style="text-align:right;">
0.037706
</td>
<td style="text-align:right;">
-0.187425
</td>
<td style="text-align:right;">
-0.033832
</td>
<td style="text-align:right;">
-0.115728
</td>
<td style="text-align:right;">
-0.060470
</td>
<td style="text-align:right;">
-0.138014
</td>
<td style="text-align:right;">
-0.005001
</td>
<td style="text-align:right;">
0.118302
</td>
<td style="text-align:right;">
0.005621
</td>
<td style="text-align:right;">
0.021717
</td>
<td style="text-align:right;">
-0.060631
</td>
<td style="text-align:right;">
-0.105902
</td>
<td style="text-align:right;">
-0.115998
</td>
<td style="text-align:right;">
-0.061484
</td>
<td style="text-align:right;">
-0.041631
</td>
<td style="text-align:right;">
-0.098898
</td>
<td style="text-align:right;">
-0.061575
</td>
<td style="text-align:right;">
-0.064054
</td>
<td style="text-align:right;">
-0.074201
</td>
<td style="text-align:right;">
-0.038599
</td>
<td style="text-align:right;">
0.031245
</td>
<td style="text-align:right;">
-0.026164
</td>
<td style="text-align:right;">
-0.042612
</td>
<td style="text-align:right;">
0.061522
</td>
<td style="text-align:right;">
-0.290406
</td>
<td style="text-align:right;">
0.243814
</td>
<td style="text-align:right;">
-0.107028
</td>
<td style="text-align:right;">
-0.412537
</td>
<td style="text-align:right;">
0.136271
</td>
<td style="text-align:right;">
-0.358346
</td>
<td style="text-align:right;">
-0.123271
</td>
<td style="text-align:right;">
0.180394
</td>
<td style="text-align:right;">
-0.313449
</td>
<td style="text-align:right;">
0.696680
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.000816
</td>
<td style="text-align:right;">
-0.003612
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Neg.Pol
</td>
<td style="text-align:right;">
-0.014219
</td>
<td style="text-align:right;">
0.232453
</td>
<td style="text-align:right;">
-0.325942
</td>
<td style="text-align:right;">
-0.246117
</td>
<td style="text-align:right;">
0.049850
</td>
<td style="text-align:right;">
0.023907
</td>
<td style="text-align:right;">
0.044325
</td>
<td style="text-align:right;">
0.025502
</td>
<td style="text-align:right;">
-0.165329
</td>
<td style="text-align:right;">
0.038841
</td>
<td style="text-align:right;">
-0.025506
</td>
<td style="text-align:right;">
0.008545
</td>
<td style="text-align:right;">
-0.001691
</td>
<td style="text-align:right;">
0.018276
</td>
<td style="text-align:right;">
0.019388
</td>
<td style="text-align:right;">
0.002891
</td>
<td style="text-align:right;">
-0.013208
</td>
<td style="text-align:right;">
-0.015512
</td>
<td style="text-align:right;">
-0.013534
</td>
<td style="text-align:right;">
-0.037369
</td>
<td style="text-align:right;">
-0.007519
</td>
<td style="text-align:right;">
-0.029810
</td>
<td style="text-align:right;">
0.018905
</td>
<td style="text-align:right;">
0.003944
</td>
<td style="text-align:right;">
0.034042
</td>
<td style="text-align:right;">
-0.044215
</td>
<td style="text-align:right;">
0.000699
</td>
<td style="text-align:right;">
-0.107613
</td>
<td style="text-align:right;">
-0.019148
</td>
<td style="text-align:right;">
0.003597
</td>
<td style="text-align:right;">
0.190981
</td>
<td style="text-align:right;">
-0.184907
</td>
<td style="text-align:right;">
0.096580
</td>
<td style="text-align:right;">
-0.039344
</td>
<td style="text-align:right;">
-0.129087
</td>
<td style="text-align:right;">
0.104479
</td>
<td style="text-align:right;">
0.550389
</td>
<td style="text-align:right;">
-0.000816
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.013166
</td>
</tr>
<tr>
<td style="text-align:left;">
Title.Subj
</td>
<td style="text-align:right;">
0.052428
</td>
<td style="text-align:right;">
0.001182
</td>
<td style="text-align:right;">
-0.056535
</td>
<td style="text-align:right;">
-0.091412
</td>
<td style="text-align:right;">
0.080444
</td>
<td style="text-align:right;">
-0.006787
</td>
<td style="text-align:right;">
0.109149
</td>
<td style="text-align:right;">
0.009568
</td>
<td style="text-align:right;">
-0.075975
</td>
<td style="text-align:right;">
0.020747
</td>
<td style="text-align:right;">
-0.033401
</td>
<td style="text-align:right;">
0.005476
</td>
<td style="text-align:right;">
0.000130
</td>
<td style="text-align:right;">
0.015501
</td>
<td style="text-align:right;">
0.018912
</td>
<td style="text-align:right;">
0.064584
</td>
<td style="text-align:right;">
0.030613
</td>
<td style="text-align:right;">
0.046499
</td>
<td style="text-align:right;">
0.085847
</td>
<td style="text-align:right;">
0.025522
</td>
<td style="text-align:right;">
0.026994
</td>
<td style="text-align:right;">
0.031699
</td>
<td style="text-align:right;">
-0.023495
</td>
<td style="text-align:right;">
0.009718
</td>
<td style="text-align:right;">
-0.059191
</td>
<td style="text-align:right;">
0.131413
</td>
<td style="text-align:right;">
-0.052219
</td>
<td style="text-align:right;">
0.126064
</td>
<td style="text-align:right;">
0.081061
</td>
<td style="text-align:right;">
0.123285
</td>
<td style="text-align:right;">
0.032759
</td>
<td style="text-align:right;">
-0.012643
</td>
<td style="text-align:right;">
-0.048362
</td>
<td style="text-align:right;">
0.043027
</td>
<td style="text-align:right;">
-0.018127
</td>
<td style="text-align:right;">
0.044484
</td>
<td style="text-align:right;">
-0.001598
</td>
<td style="text-align:right;">
-0.003612
</td>
<td style="text-align:right;">
-0.013166
</td>
<td style="text-align:right;">
1.000000
</td>
</tr>
</tbody>
</table>

The above table gives the correlations between all variables in the
Lifestyle data set. This allows us to see which two variables have
strong correlation. If we have two variables with a high correlation, we
might want to remove one of them to avoid too much multicollinearity.

``` r
#Correlation graph for lifestyle_train
correlation_graph(data_channel_train)
```
![r params$DataChannel corr_graph-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/dbaa4f7c-0fcc-40bc-85ff-169b6488f79f)

Because the correlation table above is large, it can be difficult to
read. The correlation graph above gives a visual summary of the table.
Using the legend, we are able to see the correlations between variables,
how strong the correlation is, and in what direction.

``` r
## creates a new column that is if shares is higher than average or not 
shareshigh <- data_channel_train %>% select(shares) %>% mutate (shareshigh = (shares> mean(shares)))

ggplot(shareshigh, aes(x=Rate.Pos, y=Rate.Neg,
                       color=Days_of_Week)) +
    geom_point(size=2)
```

![scatterplot-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/29b4aea9-f3ab-4193-adba-86be4c3abe15)

Once seeing the correlation table and graph, it is possible to graph two
variables on a scatterplot. This provides a visual of the linear
relationship. A scatterplot of two variables in the Lifestyle dataset
has been created above.

``` r
## mean of shares 
mean(data_channel_train$shares)
```

    ## [1] 3770.66

``` r
## sd of shares 
sd(data_channel_train$shares)
```

    ## [1] 9802.02

``` r
## creates a new column that is if shares is higher than average or not 
shareshigh <- data_channel_train %>% select(shares) %>% mutate (shareshigh = (shares> mean(shares)))

## creates a contingency table of shareshigh and whether it is the weekend 
table(shareshigh$shareshigh, data_channel_train$Weekend)
```

    ##        
    ##           0   1
    ##   FALSE 941 211
    ##   TRUE  238  79

These above contingency tables will look at the shareshigh factor which
says whether the number of shares is higher than the mean number of
shares or not and compares it to the weekend. Using these we can see if
the number of shares tends to be higher or not on the weekend.

``` r
## creates a new column that is if shares is higher than average or not 
shareshigh <- data_channel_train %>% mutate (shareshigh = (shares> mean(shares)))

## create a new column that combines Mon-Fri into weekdays
shareshigh <- mutate(shareshigh, 
                  Weekday = ifelse(Mon == 1 |
                                     Tues ==1 |
                                     Wed == 1 |
                                     Thurs == 1 |
                                     Fri == 1, 
                                    'Weekday', 'Weekend'))
shareshigh <- mutate(shareshigh, 
                  Days_of_Week = ifelse(Mon == 1 & 
                                Weekday == 'Weekday', 'Mon',
                              ifelse(Tues == 1  &
                                Weekday == "Weekday", 'Tues',
                              ifelse(Wed == 1 &
                                Weekday == "Weekday", 'Wed',
                              ifelse(Thurs ==1 &
                                Weekday == 'Weekday', 'Thurs',
                              ifelse(Fri == 1 & 
                                       Weekday == 'Weekday',
                                     'Fri', 'Weekend'))))))

shareshigh$Days_of_Week <- ordered(shareshigh$Days_of_Week, 
                                levels=c("Mon", "Tues",
                                         "Wed", "Thurs", 
                                         "Fri", "Weekend"))

## creates a contingency table of shareshigh and whether it is a weekday 
print(prop.table(table(shareshigh$Weekday,
                       shareshigh$shareshigh)))
```

    ##          
    ##               FALSE      TRUE
    ##   Weekday 0.6405718 0.1620150
    ##   Weekend 0.1436351 0.0537781

The contingency table above looks at the before-mentioned shareshigh
factor and compares it to the whether the day was a weekend or a
weekday. This allows us to see if shares tend to be higher on weekends
or weekdays. The frequencies are displayed as relative frequencies.

``` r
## creates  a contingency table of shareshigh and the day of the week
a <- prop.table(table(shareshigh$Days_of_Week,
                 shareshigh$shareshigh))
b <- as.data.frame(a)
print(a)
```

    ##          
    ##               FALSE      TRUE
    ##   Mon     0.1198094 0.0360790
    ##   Tues    0.1327434 0.0326753
    ##   Wed     0.1463581 0.0333560
    ##   Thurs   0.1238938 0.0360790
    ##   Fri     0.1177672 0.0238257
    ##   Weekend 0.1436351 0.0537781

After comparing shareshigh with whether or not the day was a weekend or
weekday, the above contingency table compares shareshigh for each
specific day of the week. Again, the frequencies are displayed as
relative frequencies.

``` r
ggplot(shareshigh, aes(x = Weekday, fill = shareshigh)) +
  geom_bar(aes(y = (after_stat(count))/sum(after_stat(count)))) + xlab('Weekday or Weekend?') + 
  ylab('Relative Frequency')
```
![weekday bar graph-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/c37d006f-1bb2-4b41-b1e4-6cb106ca8d16)

``` r
ggplot(shareshigh, aes(x = Days_of_Week, fill = shareshigh)) +
  geom_bar(aes(y = (after_stat(count))/sum(after_stat(count)))) + xlab('Day of the Week') + 
  ylab('Relative Frequency')
```
![day of the week graph-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/c9fbf341-ca08-483d-a666-41b6d17e7eb4)

The above bar graphs are a visual representation of the contingency
tables between weekends/weekdays and shareshigh and the days of the week
and shareshigh.. Graphs can improve the stakeholders ability to
interpret the results quickly.

``` r
a <- table(shareshigh$Days_of_Week)
# a <- prop.table(table(shareshigh$Days_of_Week,
#                  shareshigh$shareshigh))
b <- as.data.frame(a)
colnames(b) <- c('Day of Week', 'Freq')
b <- filter(b, Freq == max(b$Freq))
d <- as.character(b[1,1])
g <- mutate(shareshigh, 
                  Most_Freq = ifelse(Days_of_Week == d,
                                    'Most Freq Day',
                                    'Not Most Freq Day'
                                    ))
paste0(" For ", 
        params$DataChannel, " ", 
       d, " is the most frequent day of the week")
```

    ## [1] " For Lifestyle Weekend is the most frequent day of the week"

``` r
table(shareshigh$shareshigh, g$Most_Freq)
```

    ##        
    ##         Most Freq Day Not Most Freq Day
    ##   FALSE           211               941
    ##   TRUE             79               238

The above contingency table compares shareshigh to the Lifestyle day
that occurs most frequently. This allows us to see if the most frequent
day tends to have more shareshigh.

``` r
## creates plotting object of shares
a <- ggplot(data_channel_train, aes(x=shares))

## histogram of shares 
a+geom_histogram(color= "red", fill="blue")+ ggtitle("Shares histogram")
```

    ## `stat_bin()` using `bins = 30`. Pick better value with
    ## `binwidth`.

![shares histogram-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/5d78f694-8d9f-467f-b349-66548eb24a2d)

Above we can see the frequency distribution of shares of the Lifestyle
data channel. We should always see a long tail to the right because a
small number of articles will get a very high number of shares. But
looking at by looking at the distribution we can say how many shares
most of these articles got.

``` r
## creates plotting object with number of words in title and shares
b<- ggplot(data_channel_train, aes(x=n.Title, y=shares))

## creates a bar chart with number of words in title and shares 
b+ geom_col(fill="blue")+ ggtitle("Number of words in title vs shares") + labs(x="Number of words in title")
```
![col graph-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/3ff8a8e8-8fb3-4281-ad1e-de5ba7292589)

In the above graph we are looking at the number of shares based on how
many words are in the title of the article. if we see a large peak on at
the higher number of words it means for this data channel there were
more shares on longer titles, and if we see a peak at smaller number of
words then there were more shares on smaller titles.

``` r
## makes correlation of every variable with shares 
shares_correlations <- cor(data_channel_train)[1,] %>% sort() 

shares_correlations
```

    ##              LDA_02         Rate.Unique       Min.Worst.Key 
    ##         -0.04348573         -0.03785618         -0.03489802 
    ##         Min.Neg.Pol              LDA_01            Rate.Pos 
    ##         -0.03407651         -0.03239435         -0.03232085 
    ##                 Fri              LDA_04           Avg.Words 
    ##         -0.03198059         -0.02974391         -0.02699409 
    ##        Rate.Nonstop Rate.Unique.Nonstop                 Wed 
    ##         -0.02407608         -0.01981291         -0.01969601 
    ##         Min.Pos.Pol             n.Title         Avg.Neg.Pol 
    ##         -0.01913550         -0.01827770         -0.01826575 
    ##          Title.Subj          Global.Pol     Global.Pos.Rate 
    ##         -0.01356287         -0.01261071         -0.01063640 
    ##           Title.Pol             n.Other             Abs.Pol 
    ##         -0.00947727         -0.00828155         -0.00621530 
    ##                 Sun               Thurs             Max.Ref 
    ##         -0.00411168         -0.00260831          0.00174931 
    ##        Min.Best.Key       Avg.Worst.Key         Max.Pos.Pol 
    ##          0.00230129          0.00401108          0.00428011 
    ##             Weekend       Max.Worst.Key                Tues 
    ##          0.00722207          0.00831918          0.00883056 
    ##             Avg.Ref     Global.Neg.Rate         Avg.Pos.Pol 
    ##          0.01092108          0.01225355          0.01431699 
    ##                 Sat         Global.Subj            Rate.Neg 
    ##          0.01438414          0.01508142          0.01662890 
    ##         Avg.Min.Key         Max.Neg.Pol             Min.Ref 
    ##          0.01756940          0.02053757          0.02171007 
    ##        Avg.Best.Key              LDA_00            Abs.Subj 
    ##          0.02373203          0.02838517          0.03724761 
    ##                 Mon               n.Key        Max.Best.Key 
    ##          0.03724912          0.03762309          0.03816487 
    ##             n.Links         Avg.Max.Key              LDA_03 
    ##          0.03981655          0.04729846          0.04730723 
    ##            n.Images           n.Content         Avg.Avg.Key 
    ##          0.04928267          0.05991674          0.08870629 
    ##            n.Videos              shares 
    ##          0.10640517          1.00000000

``` r
## take the name of the highest correlated variable
highest_cor <-shares_correlations[52]  %>% names()

highest_cor
```

    ## [1] "n.Videos"

``` r
## creats scatter plot looking at shares vs highest correlated variable
g <-ggplot(data_channel_train,  aes(y=shares, x= data_channel_train[[highest_cor]])) 


g+ geom_point(aes(color=as.factor(Weekend))) +geom_smooth(method = lm) + ggtitle(" Highest correlated variable with shares") + labs(x="Highest correlated variable vs shares", color="Weekend")
```

    ## `geom_smooth()` using formula = 'y ~ x'

![graph of shares with highest correlated var-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/0d070389-7dc1-40fb-8e49-f80761996871)


The above graph looks at the relationship between shares and the
variable with the highest correlation for the Lifestyle data channel,
and colored based on whether or not it is the weekend. because this is
the most positively correlated variable we should always see an upward
trend but the more correlated they are the more the dots will fall onto
the line of best fit.

## Modeling

## Linear Regression

Linear regression is a tool with many applications available to data
scientists. In linear regression, a linear relationship between one
dependent variable and one or more independent variables is assumed. In
computerized linear regression, many linear regressions between the
response variable and the explanatory variable(s) are calculated. The
regression that is considered the “best fit” is the least squares
regression line. To determine the LSRL, the sum of the squared residuals
is calculated for each regression. The best model is the regression that
minimizes the sum of the squared residuals. Linear regression is used to
predict responses for explanatory variable(s); it is also used to
examine trends in the data.

### Linear regression 1

``` r
## linear regression model using all predictors 
set.seed(13)

linear_model_1 <- train( shares ~ ., 
                         data = data_channel_train,
                         method = "lm",
                         preProcess = c("center", "scale"),
                         trControl = trainControl(method = "cv", 
                                                  number = 5))

## prediction of test with model 
linear_model_1_pred <- predict(linear_model_1, newdata = dplyr::select(data_channel_test, -shares))

## storing error of model on test set 
linear_1_RMSE<- postResample(linear_model_1_pred, obs = data_channel_test$shares)
```

### Linear regression 2

``` r
#Removed rate.Nonstop because it was only 1 and removed the days of the week.
linear_model_2 <- train( shares ~. - Rate.Nonstop - Mon
                         - Tues - Wed - Thurs - Fri - Sat
                         - Sun - Weekend, 
                        data = data_channel_train,
                         method = "lm",
                         preProcess = c("center", 
                                        "scale"),
                         trControl = trainControl(
                           method= "cv", 
                           number = 5))
## prediction of test with model 
linear_model_2_pred <- predict(linear_model_2, newdata = dplyr::select(data_channel_test, -shares))

## storing error of model on test set 
linear_2_RMSE<- postResample(linear_model_2_pred, obs = data_channel_test$shares)
```

## Ensemble Models

### Random forest model

A random forest model is used in machine learning to generate
predictions or classifications. This is done through generating many
decision trees on many different samples and taking the average
(regression) or the majority vote (classification). Some of the benefits
to using random forest models are that over-fitting is minimized and the
model works with the presence of categorical and continuous variables.
With increased computer power and the increased knowledge in machine
learning, random forest models will continue to grow in popularity.

``` r
set.seed(10210526)
rfFit <- train(shares ~ ., 
        data = data_channel_train,
        method = "rf",
        trControl = trainControl(method = "cv",
                                        number = 5),
        preProcess = c("center", "scale"),
        tuneGrid = 
          data.frame(mtry = 1:sqrt(ncol(data_channel_train))))
rfFit_pred <- predict(rfFit, newdata = data_channel_test)
rfRMSE<- postResample(rfFit_pred, obs =
                            data_channel_test$shares)
```

### Boosted tree model

A decision tree makes a binary decision based on the value input. A
boosted tree model generates a predictive model based on an ensemble of
decision trees where better trees are generated based on the performance
of previous trees. Our boosted tree model can be tuned using four
different parameters: interaction.depth which defines the complexity of
the trees being built, n.trees which defines the number of trees built
(number of iterations), shrinkage which dictates the rate at which the
algorithm learns, and n.minobsinnode which dictates the number of
samples left to allow for a node to split.

``` r
## creates grid of possible tuning parameters 
gbm_grid <-  expand.grid(interaction.depth = c(1,4,7), 
  n.trees = c(1:20) , 
  shrinkage = 0.1,
  n.minobsinnode = c(10,20, 40))

## sets trainControl method 
fit_control <- trainControl(method = "repeatedcv",
                            number = 5,
                            repeats= 1)

set.seed(13)

## trains to find optimal tuning parameters except it is giving weird parameters 
gbm_tree_cv <- train(shares ~ . , data = data_channel_train,
                     method = "gbm",
                     preProcess = c("center", "scale"),
                     trControl = fit_control,
                     tuneGrid= gbm_grid,
                     verbose=FALSE)
## plot to visualize parameters 
plot(gbm_tree_cv)
```
![boosted tree tuning-1](https://github.com/kgolden4514/kgolden4514.github.io/assets/134096245/6961d60e-9346-4ee5-b83f-0360c77eab83)

``` r
## test set prediction
boosted_tree_model_pred <- predict(gbm_tree_cv, newdata = dplyr::select(data_channel_test, -shares), n.trees = 7)

## stores results 
boosted_tree_RMSE <- postResample(boosted_tree_model_pred, obs = data_channel_test$shares)
```

## Comparison

``` r
## creates a data frame of the four models RMSE on the 
models_RMSE <- data.frame(linear_1_RMSE=linear_1_RMSE[1],
                         linear_2_RMSE=linear_2_RMSE[1], 
                         rfRMSE=rfRMSE[1],
                          boosted_tree_RMSE =
                           boosted_tree_RMSE[1] )

models_RMSE

## gets the name of the column with the smallest rmse 
smallest_RMSE<-colnames(models_RMSE)[apply(models_RMSE,1,which.min)]

## declares the model with smallest RSME the winner 
paste0(" For ", 
        params$DataChannel, " ", 
       smallest_RMSE, " is the winner")
```

    ## [1] " For Lifestyle rfRMSE is the winner"

## Automation

This is the code used to automate the rendering of each document based
on the parameter of data_channel_is designated in the YAML.

``` r
## creates a list of all 6 desired params from online
data_channel_is <- c("Lifestyle", "Entertainment", "Business", "Social.Media", "Tech", "World")

## creates the output file name 
output_file <- paste0(data_channel_is, ".md")

#create a list for each channel with just the channel name parameter
params = lapply(data_channel_is, FUN = function(x){list(DataChannel = x)})

#put into a data frame
reports <- tibble(output_file, params)

## renders with params to all based on rows in reports
apply(reports, MARGIN=1, FUN = function(x){
## change first path to wherever yours is and output_dir to whatever folder you want it to output to   
rmarkdown::render('C:/Documents/Github/ST_558_Project_2/_Rmd/ST_558_Project_2.Rmd', output_dir = "./automations_test2_md", output_file = x[[1]], params = x[[2]]
    )
  }
)
```
