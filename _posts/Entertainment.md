Project 2
================
Kristina Golden and Demetrios Samaras
2023-07-02

# Entertainment

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

In this report we will be looking at the Entertainment data channel of
the online news popularity data set. This data set looks at a wide range
of variables from 39644 different news articles. The response variable
that we will be focusing on is **shares**. The purpose of this analysis
is to try to predict how many shares a Entertainment article will get
based on the values of those other variables. We will be modeling shares
using two different linear regression models and two ensemble tree based
models.

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

## Entertainment EDA

### Entertainment

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

## Entertainment Summarizations

``` r
#Shares table for data_channel_train
summary_table(data_channel_train)
```

    ##            Shares
    ## Minimum     49.00
    ## Q1         828.00
    ## Median    1200.00
    ## Q3        2100.00
    ## Maximum 210300.00
    ## Mean      2919.18
    ## SD        7642.04

The above table displays the Entertainment 5-number summary for the
shares. It also includes the mean and standard deviation. Because the
mean is greater than the median, we suspect that the Entertainment
shares distribution is right skewed.

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
0.048651
</td>
<td style="text-align:right;">
-0.014819
</td>
<td style="text-align:right;">
-0.015002
</td>
<td style="text-align:right;">
0.023271
</td>
<td style="text-align:right;">
0.011947
</td>
<td style="text-align:right;">
0.036683
</td>
<td style="text-align:right;">
0.077542
</td>
<td style="text-align:right;">
-0.067106
</td>
<td style="text-align:right;">
-0.012268
</td>
<td style="text-align:right;">
-0.151072
</td>
<td style="text-align:right;">
0.025923
</td>
<td style="text-align:right;">
-0.000795
</td>
<td style="text-align:right;">
-0.004444
</td>
<td style="text-align:right;">
0.164371
</td>
<td style="text-align:right;">
0.129565
</td>
<td style="text-align:right;">
0.009346
</td>
<td style="text-align:right;">
0.017950
</td>
<td style="text-align:right;">
0.036603
</td>
<td style="text-align:right;">
-0.032059
</td>
<td style="text-align:right;">
0.034275
</td>
<td style="text-align:right;">
0.009128
</td>
<td style="text-align:right;">
-0.034806
</td>
<td style="text-align:right;">
-0.059736
</td>
<td style="text-align:right;">
0.020061
</td>
<td style="text-align:right;">
0.078133
</td>
<td style="text-align:right;">
-0.070240
</td>
<td style="text-align:right;">
-0.033672
</td>
<td style="text-align:right;">
-0.067251
</td>
<td style="text-align:right;">
-0.065763
</td>
<td style="text-align:right;">
0.003340
</td>
<td style="text-align:right;">
-0.063504
</td>
<td style="text-align:right;">
0.028120
</td>
<td style="text-align:right;">
-0.050488
</td>
<td style="text-align:right;">
-0.022095
</td>
<td style="text-align:right;">
0.014117
</td>
<td style="text-align:right;">
-0.018277
</td>
<td style="text-align:right;">
-0.048041
</td>
<td style="text-align:right;">
0.025921
</td>
<td style="text-align:right;">
0.110429
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Content
</td>
<td style="text-align:right;">
0.048651
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.019574
</td>
<td style="text-align:right;">
0.021038
</td>
<td style="text-align:right;">
0.377318
</td>
<td style="text-align:right;">
0.387726
</td>
<td style="text-align:right;">
0.476333
</td>
<td style="text-align:right;">
0.220512
</td>
<td style="text-align:right;">
0.163794
</td>
<td style="text-align:right;">
-0.081501
</td>
<td style="text-align:right;">
-0.084185
</td>
<td style="text-align:right;">
-0.029344
</td>
<td style="text-align:right;">
-0.037321
</td>
<td style="text-align:right;">
0.053902
</td>
<td style="text-align:right;">
0.109493
</td>
<td style="text-align:right;">
0.057758
</td>
<td style="text-align:right;">
0.085725
</td>
<td style="text-align:right;">
-0.055541
</td>
<td style="text-align:right;">
-0.042816
</td>
<td style="text-align:right;">
-0.046932
</td>
<td style="text-align:right;">
0.050144
</td>
<td style="text-align:right;">
0.001865
</td>
<td style="text-align:right;">
0.010498
</td>
<td style="text-align:right;">
-0.024317
</td>
<td style="text-align:right;">
0.136332
</td>
<td style="text-align:right;">
-0.038886
</td>
<td style="text-align:right;">
0.017767
</td>
<td style="text-align:right;">
0.123865
</td>
<td style="text-align:right;">
-0.033395
</td>
<td style="text-align:right;">
0.107035
</td>
<td style="text-align:right;">
0.182649
</td>
<td style="text-align:right;">
0.035816
</td>
<td style="text-align:right;">
0.171411
</td>
<td style="text-align:right;">
0.182101
</td>
<td style="text-align:right;">
-0.258499
</td>
<td style="text-align:right;">
0.433123
</td>
<td style="text-align:right;">
-0.155826
</td>
<td style="text-align:right;">
-0.494800
</td>
<td style="text-align:right;">
0.258640
</td>
<td style="text-align:right;">
-0.010097
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Unique
</td>
<td style="text-align:right;">
-0.014819
</td>
<td style="text-align:right;">
0.019574
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.999981
</td>
<td style="text-align:right;">
-0.001289
</td>
<td style="text-align:right;">
0.029544
</td>
<td style="text-align:right;">
0.050518
</td>
<td style="text-align:right;">
-0.006587
</td>
<td style="text-align:right;">
0.013048
</td>
<td style="text-align:right;">
-0.000159
</td>
<td style="text-align:right;">
-0.003670
</td>
<td style="text-align:right;">
-0.001078
</td>
<td style="text-align:right;">
-0.002512
</td>
<td style="text-align:right;">
0.003504
</td>
<td style="text-align:right;">
0.003923
</td>
<td style="text-align:right;">
0.009712
</td>
<td style="text-align:right;">
0.015545
</td>
<td style="text-align:right;">
-0.006140
</td>
<td style="text-align:right;">
-0.003785
</td>
<td style="text-align:right;">
-0.002379
</td>
<td style="text-align:right;">
-0.004688
</td>
<td style="text-align:right;">
0.003597
</td>
<td style="text-align:right;">
-0.010726
</td>
<td style="text-align:right;">
-0.017545
</td>
<td style="text-align:right;">
-0.010825
</td>
<td style="text-align:right;">
-0.015792
</td>
<td style="text-align:right;">
-0.009995
</td>
<td style="text-align:right;">
-0.050145
</td>
<td style="text-align:right;">
-0.013104
</td>
<td style="text-align:right;">
-0.029614
</td>
<td style="text-align:right;">
-0.020239
</td>
<td style="text-align:right;">
-0.044156
</td>
<td style="text-align:right;">
-0.026000
</td>
<td style="text-align:right;">
-0.044497
</td>
<td style="text-align:right;">
-0.015589
</td>
<td style="text-align:right;">
-0.043773
</td>
<td style="text-align:right;">
0.029402
</td>
<td style="text-align:right;">
0.028622
</td>
<td style="text-align:right;">
0.011031
</td>
<td style="text-align:right;">
-0.014008
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Unique.Nonstop
</td>
<td style="text-align:right;">
-0.015002
</td>
<td style="text-align:right;">
0.021038
</td>
<td style="text-align:right;">
0.999981
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.001599
</td>
<td style="text-align:right;">
0.029872
</td>
<td style="text-align:right;">
0.048812
</td>
<td style="text-align:right;">
-0.006286
</td>
<td style="text-align:right;">
0.015603
</td>
<td style="text-align:right;">
-0.000998
</td>
<td style="text-align:right;">
-0.003612
</td>
<td style="text-align:right;">
-0.001068
</td>
<td style="text-align:right;">
-0.002455
</td>
<td style="text-align:right;">
0.003277
</td>
<td style="text-align:right;">
0.003896
</td>
<td style="text-align:right;">
0.009095
</td>
<td style="text-align:right;">
0.015134
</td>
<td style="text-align:right;">
-0.006441
</td>
<td style="text-align:right;">
-0.004464
</td>
<td style="text-align:right;">
-0.002342
</td>
<td style="text-align:right;">
-0.004778
</td>
<td style="text-align:right;">
0.003528
</td>
<td style="text-align:right;">
-0.010565
</td>
<td style="text-align:right;">
-0.016344
</td>
<td style="text-align:right;">
-0.010711
</td>
<td style="text-align:right;">
-0.017172
</td>
<td style="text-align:right;">
-0.009482
</td>
<td style="text-align:right;">
-0.047616
</td>
<td style="text-align:right;">
-0.012570
</td>
<td style="text-align:right;">
-0.027221
</td>
<td style="text-align:right;">
-0.018677
</td>
<td style="text-align:right;">
-0.042074
</td>
<td style="text-align:right;">
-0.024820
</td>
<td style="text-align:right;">
-0.042500
</td>
<td style="text-align:right;">
-0.015560
</td>
<td style="text-align:right;">
-0.041201
</td>
<td style="text-align:right;">
0.027843
</td>
<td style="text-align:right;">
0.026426
</td>
<td style="text-align:right;">
0.011210
</td>
<td style="text-align:right;">
-0.014138
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Links
</td>
<td style="text-align:right;">
0.023271
</td>
<td style="text-align:right;">
0.377318
</td>
<td style="text-align:right;">
-0.001289
</td>
<td style="text-align:right;">
-0.001599
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.415860
</td>
<td style="text-align:right;">
0.260316
</td>
<td style="text-align:right;">
0.174609
</td>
<td style="text-align:right;">
0.215949
</td>
<td style="text-align:right;">
0.045499
</td>
<td style="text-align:right;">
-0.054173
</td>
<td style="text-align:right;">
-0.007907
</td>
<td style="text-align:right;">
-0.016210
</td>
<td style="text-align:right;">
0.003090
</td>
<td style="text-align:right;">
0.072270
</td>
<td style="text-align:right;">
0.006532
</td>
<td style="text-align:right;">
0.066465
</td>
<td style="text-align:right;">
-0.008576
</td>
<td style="text-align:right;">
0.020434
</td>
<td style="text-align:right;">
-0.023395
</td>
<td style="text-align:right;">
0.131802
</td>
<td style="text-align:right;">
0.078040
</td>
<td style="text-align:right;">
0.033342
</td>
<td style="text-align:right;">
-0.098662
</td>
<td style="text-align:right;">
0.065453
</td>
<td style="text-align:right;">
0.062237
</td>
<td style="text-align:right;">
-0.006073
</td>
<td style="text-align:right;">
0.125892
</td>
<td style="text-align:right;">
0.053643
</td>
<td style="text-align:right;">
-0.014674
</td>
<td style="text-align:right;">
-0.025759
</td>
<td style="text-align:right;">
0.086921
</td>
<td style="text-align:right;">
0.052987
</td>
<td style="text-align:right;">
0.120099
</td>
<td style="text-align:right;">
-0.108779
</td>
<td style="text-align:right;">
0.222730
</td>
<td style="text-align:right;">
-0.102084
</td>
<td style="text-align:right;">
-0.219638
</td>
<td style="text-align:right;">
0.074829
</td>
<td style="text-align:right;">
-0.016109
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Other
</td>
<td style="text-align:right;">
0.011947
</td>
<td style="text-align:right;">
0.387726
</td>
<td style="text-align:right;">
0.029544
</td>
<td style="text-align:right;">
0.029872
</td>
<td style="text-align:right;">
0.415860
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.257050
</td>
<td style="text-align:right;">
0.165901
</td>
<td style="text-align:right;">
0.208372
</td>
<td style="text-align:right;">
-0.021675
</td>
<td style="text-align:right;">
-0.046423
</td>
<td style="text-align:right;">
-0.013503
</td>
<td style="text-align:right;">
-0.014649
</td>
<td style="text-align:right;">
0.039141
</td>
<td style="text-align:right;">
0.054837
</td>
<td style="text-align:right;">
0.023245
</td>
<td style="text-align:right;">
0.120625
</td>
<td style="text-align:right;">
-0.029780
</td>
<td style="text-align:right;">
-0.007110
</td>
<td style="text-align:right;">
-0.054694
</td>
<td style="text-align:right;">
0.166907
</td>
<td style="text-align:right;">
0.055602
</td>
<td style="text-align:right;">
0.027933
</td>
<td style="text-align:right;">
-0.101452
</td>
<td style="text-align:right;">
0.112069
</td>
<td style="text-align:right;">
0.044059
</td>
<td style="text-align:right;">
0.003766
</td>
<td style="text-align:right;">
0.129344
</td>
<td style="text-align:right;">
0.055643
</td>
<td style="text-align:right;">
0.100075
</td>
<td style="text-align:right;">
0.047307
</td>
<td style="text-align:right;">
0.130231
</td>
<td style="text-align:right;">
0.058484
</td>
<td style="text-align:right;">
0.151816
</td>
<td style="text-align:right;">
-0.099542
</td>
<td style="text-align:right;">
0.243580
</td>
<td style="text-align:right;">
-0.107893
</td>
<td style="text-align:right;">
-0.221267
</td>
<td style="text-align:right;">
0.060164
</td>
<td style="text-align:right;">
-0.033539
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Images
</td>
<td style="text-align:right;">
0.036683
</td>
<td style="text-align:right;">
0.476333
</td>
<td style="text-align:right;">
0.050518
</td>
<td style="text-align:right;">
0.048812
</td>
<td style="text-align:right;">
0.260316
</td>
<td style="text-align:right;">
0.257050
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.092045
</td>
<td style="text-align:right;">
0.073842
</td>
<td style="text-align:right;">
0.032053
</td>
<td style="text-align:right;">
-0.066108
</td>
<td style="text-align:right;">
0.000523
</td>
<td style="text-align:right;">
-0.008757
</td>
<td style="text-align:right;">
0.045209
</td>
<td style="text-align:right;">
0.087188
</td>
<td style="text-align:right;">
0.052757
</td>
<td style="text-align:right;">
0.078803
</td>
<td style="text-align:right;">
0.008094
</td>
<td style="text-align:right;">
0.040841
</td>
<td style="text-align:right;">
-0.014736
</td>
<td style="text-align:right;">
0.057388
</td>
<td style="text-align:right;">
0.038989
</td>
<td style="text-align:right;">
-0.009761
</td>
<td style="text-align:right;">
-0.069591
</td>
<td style="text-align:right;">
0.069884
</td>
<td style="text-align:right;">
0.049318
</td>
<td style="text-align:right;">
-0.035778
</td>
<td style="text-align:right;">
-0.040391
</td>
<td style="text-align:right;">
-0.088776
</td>
<td style="text-align:right;">
-0.180249
</td>
<td style="text-align:right;">
0.014267
</td>
<td style="text-align:right;">
-0.088639
</td>
<td style="text-align:right;">
0.131297
</td>
<td style="text-align:right;">
0.053317
</td>
<td style="text-align:right;">
-0.092352
</td>
<td style="text-align:right;">
0.151287
</td>
<td style="text-align:right;">
-0.029599
</td>
<td style="text-align:right;">
-0.158060
</td>
<td style="text-align:right;">
0.093962
</td>
<td style="text-align:right;">
0.000734
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Videos
</td>
<td style="text-align:right;">
0.077542
</td>
<td style="text-align:right;">
0.220512
</td>
<td style="text-align:right;">
-0.006587
</td>
<td style="text-align:right;">
-0.006286
</td>
<td style="text-align:right;">
0.174609
</td>
<td style="text-align:right;">
0.165901
</td>
<td style="text-align:right;">
-0.092045
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.042930
</td>
<td style="text-align:right;">
-0.057520
</td>
<td style="text-align:right;">
-0.039203
</td>
<td style="text-align:right;">
-0.030074
</td>
<td style="text-align:right;">
-0.038364
</td>
<td style="text-align:right;">
0.054433
</td>
<td style="text-align:right;">
0.051413
</td>
<td style="text-align:right;">
0.193138
</td>
<td style="text-align:right;">
0.060230
</td>
<td style="text-align:right;">
-0.003454
</td>
<td style="text-align:right;">
0.035524
</td>
<td style="text-align:right;">
-0.015131
</td>
<td style="text-align:right;">
0.102127
</td>
<td style="text-align:right;">
0.066432
</td>
<td style="text-align:right;">
-0.029521
</td>
<td style="text-align:right;">
-0.197524
</td>
<td style="text-align:right;">
0.008305
</td>
<td style="text-align:right;">
0.213303
</td>
<td style="text-align:right;">
-0.067658
</td>
<td style="text-align:right;">
0.050824
</td>
<td style="text-align:right;">
-0.065404
</td>
<td style="text-align:right;">
0.113715
</td>
<td style="text-align:right;">
0.251790
</td>
<td style="text-align:right;">
-0.051550
</td>
<td style="text-align:right;">
0.113680
</td>
<td style="text-align:right;">
0.110954
</td>
<td style="text-align:right;">
-0.091255
</td>
<td style="text-align:right;">
0.171234
</td>
<td style="text-align:right;">
-0.124853
</td>
<td style="text-align:right;">
-0.212453
</td>
<td style="text-align:right;">
0.093116
</td>
<td style="text-align:right;">
0.035936
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Words
</td>
<td style="text-align:right;">
-0.067106
</td>
<td style="text-align:right;">
0.163794
</td>
<td style="text-align:right;">
0.013048
</td>
<td style="text-align:right;">
0.015603
</td>
<td style="text-align:right;">
0.215949
</td>
<td style="text-align:right;">
0.208372
</td>
<td style="text-align:right;">
0.073842
</td>
<td style="text-align:right;">
0.042930
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.072719
</td>
<td style="text-align:right;">
0.040882
</td>
<td style="text-align:right;">
-0.008144
</td>
<td style="text-align:right;">
0.007109
</td>
<td style="text-align:right;">
-0.008178
</td>
<td style="text-align:right;">
-0.049131
</td>
<td style="text-align:right;">
-0.158402
</td>
<td style="text-align:right;">
-0.057943
</td>
<td style="text-align:right;">
-0.044817
</td>
<td style="text-align:right;">
-0.112763
</td>
<td style="text-align:right;">
0.050214
</td>
<td style="text-align:right;">
0.059059
</td>
<td style="text-align:right;">
0.075147
</td>
<td style="text-align:right;">
0.019250
</td>
<td style="text-align:right;">
0.093583
</td>
<td style="text-align:right;">
0.024710
</td>
<td style="text-align:right;">
-0.114299
</td>
<td style="text-align:right;">
0.033978
</td>
<td style="text-align:right;">
0.642421
</td>
<td style="text-align:right;">
0.190679
</td>
<td style="text-align:right;">
0.349747
</td>
<td style="text-align:right;">
0.208540
</td>
<td style="text-align:right;">
0.600476
</td>
<td style="text-align:right;">
0.318678
</td>
<td style="text-align:right;">
0.571425
</td>
<td style="text-align:right;">
0.243124
</td>
<td style="text-align:right;">
0.511031
</td>
<td style="text-align:right;">
-0.361958
</td>
<td style="text-align:right;">
-0.296738
</td>
<td style="text-align:right;">
-0.197925
</td>
<td style="text-align:right;">
-0.040130
</td>
</tr>
<tr>
<td style="text-align:left;">
n.Key
</td>
<td style="text-align:right;">
-0.012268
</td>
<td style="text-align:right;">
-0.081501
</td>
<td style="text-align:right;">
-0.000159
</td>
<td style="text-align:right;">
-0.000998
</td>
<td style="text-align:right;">
0.045499
</td>
<td style="text-align:right;">
-0.021675
</td>
<td style="text-align:right;">
0.032053
</td>
<td style="text-align:right;">
-0.057520
</td>
<td style="text-align:right;">
-0.072719
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.056121
</td>
<td style="text-align:right;">
0.092794
</td>
<td style="text-align:right;">
0.090607
</td>
<td style="text-align:right;">
-0.354891
</td>
<td style="text-align:right;">
-0.050129
</td>
<td style="text-align:right;">
-0.281725
</td>
<td style="text-align:right;">
-0.330625
</td>
<td style="text-align:right;">
0.181277
</td>
<td style="text-align:right;">
0.076351
</td>
<td style="text-align:right;">
0.008928
</td>
<td style="text-align:right;">
0.056484
</td>
<td style="text-align:right;">
0.057222
</td>
<td style="text-align:right;">
0.085105
</td>
<td style="text-align:right;">
-0.081919
</td>
<td style="text-align:right;">
-0.084612
</td>
<td style="text-align:right;">
0.061684
</td>
<td style="text-align:right;">
0.106625
</td>
<td style="text-align:right;">
-0.058944
</td>
<td style="text-align:right;">
0.044189
</td>
<td style="text-align:right;">
-0.056011
</td>
<td style="text-align:right;">
-0.112516
</td>
<td style="text-align:right;">
-0.011337
</td>
<td style="text-align:right;">
-0.102865
</td>
<td style="text-align:right;">
-0.076176
</td>
<td style="text-align:right;">
0.030341
</td>
<td style="text-align:right;">
-0.101028
</td>
<td style="text-align:right;">
0.089525
</td>
<td style="text-align:right;">
0.108599
</td>
<td style="text-align:right;">
-0.001958
</td>
<td style="text-align:right;">
0.006300
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Worst.Key
</td>
<td style="text-align:right;">
-0.151072
</td>
<td style="text-align:right;">
-0.084185
</td>
<td style="text-align:right;">
-0.003670
</td>
<td style="text-align:right;">
-0.003612
</td>
<td style="text-align:right;">
-0.054173
</td>
<td style="text-align:right;">
-0.046423
</td>
<td style="text-align:right;">
-0.066108
</td>
<td style="text-align:right;">
-0.039203
</td>
<td style="text-align:right;">
0.040882
</td>
<td style="text-align:right;">
0.056121
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.001190
</td>
<td style="text-align:right;">
0.079099
</td>
<td style="text-align:right;">
-0.097067
</td>
<td style="text-align:right;">
-0.875376
</td>
<td style="text-align:right;">
-0.535278
</td>
<td style="text-align:right;">
-0.154376
</td>
<td style="text-align:right;">
-0.063632
</td>
<td style="text-align:right;">
-0.201076
</td>
<td style="text-align:right;">
-0.021991
</td>
<td style="text-align:right;">
-0.057648
</td>
<td style="text-align:right;">
-0.057516
</td>
<td style="text-align:right;">
0.056363
</td>
<td style="text-align:right;">
0.039650
</td>
<td style="text-align:right;">
-0.050606
</td>
<td style="text-align:right;">
-0.041931
</td>
<td style="text-align:right;">
0.034100
</td>
<td style="text-align:right;">
0.008122
</td>
<td style="text-align:right;">
0.094646
</td>
<td style="text-align:right;">
0.112629
</td>
<td style="text-align:right;">
-0.029882
</td>
<td style="text-align:right;">
0.099875
</td>
<td style="text-align:right;">
-0.082273
</td>
<td style="text-align:right;">
-0.004230
</td>
<td style="text-align:right;">
-0.011315
</td>
<td style="text-align:right;">
-0.040097
</td>
<td style="text-align:right;">
0.077188
</td>
<td style="text-align:right;">
0.109383
</td>
<td style="text-align:right;">
-0.014148
</td>
<td style="text-align:right;">
-0.018426
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Worst.Key
</td>
<td style="text-align:right;">
0.025923
</td>
<td style="text-align:right;">
-0.029344
</td>
<td style="text-align:right;">
-0.001078
</td>
<td style="text-align:right;">
-0.001068
</td>
<td style="text-align:right;">
-0.007907
</td>
<td style="text-align:right;">
-0.013503
</td>
<td style="text-align:right;">
0.000523
</td>
<td style="text-align:right;">
-0.030074
</td>
<td style="text-align:right;">
-0.008144
</td>
<td style="text-align:right;">
0.092794
</td>
<td style="text-align:right;">
0.001190
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.965225
</td>
<td style="text-align:right;">
-0.056489
</td>
<td style="text-align:right;">
0.005109
</td>
<td style="text-align:right;">
-0.054010
</td>
<td style="text-align:right;">
-0.009694
</td>
<td style="text-align:right;">
0.711832
</td>
<td style="text-align:right;">
0.572109
</td>
<td style="text-align:right;">
0.032501
</td>
<td style="text-align:right;">
0.066085
</td>
<td style="text-align:right;">
0.068632
</td>
<td style="text-align:right;">
-0.017893
</td>
<td style="text-align:right;">
-0.010377
</td>
<td style="text-align:right;">
0.011526
</td>
<td style="text-align:right;">
0.012241
</td>
<td style="text-align:right;">
-0.007671
</td>
<td style="text-align:right;">
-0.016100
</td>
<td style="text-align:right;">
0.001244
</td>
<td style="text-align:right;">
-0.008662
</td>
<td style="text-align:right;">
-0.017480
</td>
<td style="text-align:right;">
0.007240
</td>
<td style="text-align:right;">
-0.015448
</td>
<td style="text-align:right;">
-0.008878
</td>
<td style="text-align:right;">
0.032172
</td>
<td style="text-align:right;">
-0.026470
</td>
<td style="text-align:right;">
0.000491
</td>
<td style="text-align:right;">
0.015810
</td>
<td style="text-align:right;">
-0.013485
</td>
<td style="text-align:right;">
0.021930
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Worst.Key
</td>
<td style="text-align:right;">
-0.000795
</td>
<td style="text-align:right;">
-0.037321
</td>
<td style="text-align:right;">
-0.002512
</td>
<td style="text-align:right;">
-0.002455
</td>
<td style="text-align:right;">
-0.016210
</td>
<td style="text-align:right;">
-0.014649
</td>
<td style="text-align:right;">
-0.008757
</td>
<td style="text-align:right;">
-0.038364
</td>
<td style="text-align:right;">
0.007109
</td>
<td style="text-align:right;">
0.090607
</td>
<td style="text-align:right;">
0.079099
</td>
<td style="text-align:right;">
0.965225
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.079299
</td>
<td style="text-align:right;">
-0.077080
</td>
<td style="text-align:right;">
-0.135899
</td>
<td style="text-align:right;">
-0.034937
</td>
<td style="text-align:right;">
0.688782
</td>
<td style="text-align:right;">
0.561033
</td>
<td style="text-align:right;">
0.037104
</td>
<td style="text-align:right;">
0.066622
</td>
<td style="text-align:right;">
0.076828
</td>
<td style="text-align:right;">
-0.010751
</td>
<td style="text-align:right;">
0.002155
</td>
<td style="text-align:right;">
0.009963
</td>
<td style="text-align:right;">
-0.001337
</td>
<td style="text-align:right;">
-0.006531
</td>
<td style="text-align:right;">
-0.006473
</td>
<td style="text-align:right;">
0.013221
</td>
<td style="text-align:right;">
0.012457
</td>
<td style="text-align:right;">
-0.018083
</td>
<td style="text-align:right;">
0.024610
</td>
<td style="text-align:right;">
-0.020493
</td>
<td style="text-align:right;">
-0.004585
</td>
<td style="text-align:right;">
0.031879
</td>
<td style="text-align:right;">
-0.025497
</td>
<td style="text-align:right;">
0.002884
</td>
<td style="text-align:right;">
0.025361
</td>
<td style="text-align:right;">
-0.019434
</td>
<td style="text-align:right;">
0.019576
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Best.Key
</td>
<td style="text-align:right;">
-0.004444
</td>
<td style="text-align:right;">
0.053902
</td>
<td style="text-align:right;">
0.003504
</td>
<td style="text-align:right;">
0.003277
</td>
<td style="text-align:right;">
0.003090
</td>
<td style="text-align:right;">
0.039141
</td>
<td style="text-align:right;">
0.045209
</td>
<td style="text-align:right;">
0.054433
</td>
<td style="text-align:right;">
-0.008178
</td>
<td style="text-align:right;">
-0.354891
</td>
<td style="text-align:right;">
-0.097067
</td>
<td style="text-align:right;">
-0.056489
</td>
<td style="text-align:right;">
-0.079299
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.107838
</td>
<td style="text-align:right;">
0.482124
</td>
<td style="text-align:right;">
0.532063
</td>
<td style="text-align:right;">
0.018338
</td>
<td style="text-align:right;">
0.226245
</td>
<td style="text-align:right;">
0.005014
</td>
<td style="text-align:right;">
0.016471
</td>
<td style="text-align:right;">
0.008986
</td>
<td style="text-align:right;">
0.010164
</td>
<td style="text-align:right;">
-0.079871
</td>
<td style="text-align:right;">
-0.003032
</td>
<td style="text-align:right;">
0.078755
</td>
<td style="text-align:right;">
-0.012498
</td>
<td style="text-align:right;">
-0.023834
</td>
<td style="text-align:right;">
-0.033401
</td>
<td style="text-align:right;">
-0.027311
</td>
<td style="text-align:right;">
0.018019
</td>
<td style="text-align:right;">
-0.035932
</td>
<td style="text-align:right;">
0.033118
</td>
<td style="text-align:right;">
0.020747
</td>
<td style="text-align:right;">
-0.014794
</td>
<td style="text-align:right;">
0.024696
</td>
<td style="text-align:right;">
-0.016956
</td>
<td style="text-align:right;">
-0.042243
</td>
<td style="text-align:right;">
0.019293
</td>
<td style="text-align:right;">
-0.001003
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Best.Key
</td>
<td style="text-align:right;">
0.164371
</td>
<td style="text-align:right;">
0.109493
</td>
<td style="text-align:right;">
0.003923
</td>
<td style="text-align:right;">
0.003896
</td>
<td style="text-align:right;">
0.072270
</td>
<td style="text-align:right;">
0.054837
</td>
<td style="text-align:right;">
0.087188
</td>
<td style="text-align:right;">
0.051413
</td>
<td style="text-align:right;">
-0.049131
</td>
<td style="text-align:right;">
-0.050129
</td>
<td style="text-align:right;">
-0.875376
</td>
<td style="text-align:right;">
0.005109
</td>
<td style="text-align:right;">
-0.077080
</td>
<td style="text-align:right;">
0.107838
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.606035
</td>
<td style="text-align:right;">
0.166839
</td>
<td style="text-align:right;">
0.085338
</td>
<td style="text-align:right;">
0.237304
</td>
<td style="text-align:right;">
0.020276
</td>
<td style="text-align:right;">
0.064176
</td>
<td style="text-align:right;">
0.061818
</td>
<td style="text-align:right;">
-0.068769
</td>
<td style="text-align:right;">
-0.033433
</td>
<td style="text-align:right;">
0.054341
</td>
<td style="text-align:right;">
0.039706
</td>
<td style="text-align:right;">
-0.041688
</td>
<td style="text-align:right;">
-0.015055
</td>
<td style="text-align:right;">
-0.112997
</td>
<td style="text-align:right;">
-0.126159
</td>
<td style="text-align:right;">
0.054773
</td>
<td style="text-align:right;">
-0.122321
</td>
<td style="text-align:right;">
0.104672
</td>
<td style="text-align:right;">
0.000739
</td>
<td style="text-align:right;">
0.014036
</td>
<td style="text-align:right;">
0.041788
</td>
<td style="text-align:right;">
-0.068410
</td>
<td style="text-align:right;">
-0.120472
</td>
<td style="text-align:right;">
0.038279
</td>
<td style="text-align:right;">
0.026432
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Best.Key
</td>
<td style="text-align:right;">
0.129565
</td>
<td style="text-align:right;">
0.057758
</td>
<td style="text-align:right;">
0.009712
</td>
<td style="text-align:right;">
0.009095
</td>
<td style="text-align:right;">
0.006532
</td>
<td style="text-align:right;">
0.023245
</td>
<td style="text-align:right;">
0.052757
</td>
<td style="text-align:right;">
0.193138
</td>
<td style="text-align:right;">
-0.158402
</td>
<td style="text-align:right;">
-0.281725
</td>
<td style="text-align:right;">
-0.535278
</td>
<td style="text-align:right;">
-0.054010
</td>
<td style="text-align:right;">
-0.135899
</td>
<td style="text-align:right;">
0.482124
</td>
<td style="text-align:right;">
0.606035
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.444008
</td>
<td style="text-align:right;">
0.087157
</td>
<td style="text-align:right;">
0.373897
</td>
<td style="text-align:right;">
0.034653
</td>
<td style="text-align:right;">
0.055367
</td>
<td style="text-align:right;">
0.062323
</td>
<td style="text-align:right;">
-0.068838
</td>
<td style="text-align:right;">
-0.242088
</td>
<td style="text-align:right;">
0.050257
</td>
<td style="text-align:right;">
0.262958
</td>
<td style="text-align:right;">
-0.118845
</td>
<td style="text-align:right;">
-0.088549
</td>
<td style="text-align:right;">
-0.118940
</td>
<td style="text-align:right;">
-0.130456
</td>
<td style="text-align:right;">
0.056105
</td>
<td style="text-align:right;">
-0.191420
</td>
<td style="text-align:right;">
0.059070
</td>
<td style="text-align:right;">
-0.025027
</td>
<td style="text-align:right;">
-0.020444
</td>
<td style="text-align:right;">
-0.010054
</td>
<td style="text-align:right;">
-0.037530
</td>
<td style="text-align:right;">
-0.073738
</td>
<td style="text-align:right;">
0.025348
</td>
<td style="text-align:right;">
0.027677
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Min.Key
</td>
<td style="text-align:right;">
0.009346
</td>
<td style="text-align:right;">
0.085725
</td>
<td style="text-align:right;">
0.015545
</td>
<td style="text-align:right;">
0.015134
</td>
<td style="text-align:right;">
0.066465
</td>
<td style="text-align:right;">
0.120625
</td>
<td style="text-align:right;">
0.078803
</td>
<td style="text-align:right;">
0.060230
</td>
<td style="text-align:right;">
-0.057943
</td>
<td style="text-align:right;">
-0.330625
</td>
<td style="text-align:right;">
-0.154376
</td>
<td style="text-align:right;">
-0.009694
</td>
<td style="text-align:right;">
-0.034937
</td>
<td style="text-align:right;">
0.532063
</td>
<td style="text-align:right;">
0.166839
</td>
<td style="text-align:right;">
0.444008
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.057203
</td>
<td style="text-align:right;">
0.416009
</td>
<td style="text-align:right;">
0.017119
</td>
<td style="text-align:right;">
0.036408
</td>
<td style="text-align:right;">
0.034637
</td>
<td style="text-align:right;">
-0.020613
</td>
<td style="text-align:right;">
-0.098456
</td>
<td style="text-align:right;">
0.053055
</td>
<td style="text-align:right;">
0.080751
</td>
<td style="text-align:right;">
-0.007481
</td>
<td style="text-align:right;">
-0.021499
</td>
<td style="text-align:right;">
-0.025151
</td>
<td style="text-align:right;">
-0.051786
</td>
<td style="text-align:right;">
0.021424
</td>
<td style="text-align:right;">
-0.068211
</td>
<td style="text-align:right;">
0.028538
</td>
<td style="text-align:right;">
0.035750
</td>
<td style="text-align:right;">
-0.026934
</td>
<td style="text-align:right;">
0.042856
</td>
<td style="text-align:right;">
-0.041021
</td>
<td style="text-align:right;">
-0.068967
</td>
<td style="text-align:right;">
0.020505
</td>
<td style="text-align:right;">
0.019656
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Max.Key
</td>
<td style="text-align:right;">
0.017950
</td>
<td style="text-align:right;">
-0.055541
</td>
<td style="text-align:right;">
-0.006140
</td>
<td style="text-align:right;">
-0.006441
</td>
<td style="text-align:right;">
-0.008576
</td>
<td style="text-align:right;">
-0.029780
</td>
<td style="text-align:right;">
0.008094
</td>
<td style="text-align:right;">
-0.003454
</td>
<td style="text-align:right;">
-0.044817
</td>
<td style="text-align:right;">
0.181277
</td>
<td style="text-align:right;">
-0.063632
</td>
<td style="text-align:right;">
0.711832
</td>
<td style="text-align:right;">
0.688782
</td>
<td style="text-align:right;">
0.018338
</td>
<td style="text-align:right;">
0.085338
</td>
<td style="text-align:right;">
0.087157
</td>
<td style="text-align:right;">
0.057203
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.833061
</td>
<td style="text-align:right;">
0.058664
</td>
<td style="text-align:right;">
0.122793
</td>
<td style="text-align:right;">
0.143177
</td>
<td style="text-align:right;">
-0.027500
</td>
<td style="text-align:right;">
-0.082880
</td>
<td style="text-align:right;">
-0.051256
</td>
<td style="text-align:right;">
0.116664
</td>
<td style="text-align:right;">
-0.033866
</td>
<td style="text-align:right;">
-0.025358
</td>
<td style="text-align:right;">
-0.006580
</td>
<td style="text-align:right;">
-0.049908
</td>
<td style="text-align:right;">
-0.030322
</td>
<td style="text-align:right;">
-0.029354
</td>
<td style="text-align:right;">
-0.017563
</td>
<td style="text-align:right;">
-0.008911
</td>
<td style="text-align:right;">
0.039030
</td>
<td style="text-align:right;">
-0.044781
</td>
<td style="text-align:right;">
0.001110
</td>
<td style="text-align:right;">
0.016887
</td>
<td style="text-align:right;">
-0.024474
</td>
<td style="text-align:right;">
0.021324
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Avg.Key
</td>
<td style="text-align:right;">
0.036603
</td>
<td style="text-align:right;">
-0.042816
</td>
<td style="text-align:right;">
-0.003785
</td>
<td style="text-align:right;">
-0.004464
</td>
<td style="text-align:right;">
0.020434
</td>
<td style="text-align:right;">
-0.007110
</td>
<td style="text-align:right;">
0.040841
</td>
<td style="text-align:right;">
0.035524
</td>
<td style="text-align:right;">
-0.112763
</td>
<td style="text-align:right;">
0.076351
</td>
<td style="text-align:right;">
-0.201076
</td>
<td style="text-align:right;">
0.572109
</td>
<td style="text-align:right;">
0.561033
</td>
<td style="text-align:right;">
0.226245
</td>
<td style="text-align:right;">
0.237304
</td>
<td style="text-align:right;">
0.373897
</td>
<td style="text-align:right;">
0.416009
</td>
<td style="text-align:right;">
0.833061
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.080898
</td>
<td style="text-align:right;">
0.145038
</td>
<td style="text-align:right;">
0.174808
</td>
<td style="text-align:right;">
-0.047316
</td>
<td style="text-align:right;">
-0.127711
</td>
<td style="text-align:right;">
-0.081656
</td>
<td style="text-align:right;">
0.186483
</td>
<td style="text-align:right;">
-0.069566
</td>
<td style="text-align:right;">
-0.045873
</td>
<td style="text-align:right;">
-0.033233
</td>
<td style="text-align:right;">
-0.095431
</td>
<td style="text-align:right;">
-0.030342
</td>
<td style="text-align:right;">
-0.092884
</td>
<td style="text-align:right;">
-0.013352
</td>
<td style="text-align:right;">
-0.022992
</td>
<td style="text-align:right;">
0.028275
</td>
<td style="text-align:right;">
-0.052130
</td>
<td style="text-align:right;">
-0.010945
</td>
<td style="text-align:right;">
0.001073
</td>
<td style="text-align:right;">
-0.025406
</td>
<td style="text-align:right;">
0.038152
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Ref
</td>
<td style="text-align:right;">
-0.032059
</td>
<td style="text-align:right;">
-0.046932
</td>
<td style="text-align:right;">
-0.002379
</td>
<td style="text-align:right;">
-0.002342
</td>
<td style="text-align:right;">
-0.023395
</td>
<td style="text-align:right;">
-0.054694
</td>
<td style="text-align:right;">
-0.014736
</td>
<td style="text-align:right;">
-0.015131
</td>
<td style="text-align:right;">
0.050214
</td>
<td style="text-align:right;">
0.008928
</td>
<td style="text-align:right;">
-0.021991
</td>
<td style="text-align:right;">
0.032501
</td>
<td style="text-align:right;">
0.037104
</td>
<td style="text-align:right;">
0.005014
</td>
<td style="text-align:right;">
0.020276
</td>
<td style="text-align:right;">
0.034653
</td>
<td style="text-align:right;">
0.017119
</td>
<td style="text-align:right;">
0.058664
</td>
<td style="text-align:right;">
0.080898
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.302531
</td>
<td style="text-align:right;">
0.743695
</td>
<td style="text-align:right;">
-0.008135
</td>
<td style="text-align:right;">
0.016014
</td>
<td style="text-align:right;">
-0.022087
</td>
<td style="text-align:right;">
-0.010245
</td>
<td style="text-align:right;">
0.021728
</td>
<td style="text-align:right;">
0.051547
</td>
<td style="text-align:right;">
0.028372
</td>
<td style="text-align:right;">
0.026607
</td>
<td style="text-align:right;">
-0.000853
</td>
<td style="text-align:right;">
0.046789
</td>
<td style="text-align:right;">
0.011169
</td>
<td style="text-align:right;">
0.047079
</td>
<td style="text-align:right;">
0.047951
</td>
<td style="text-align:right;">
0.015913
</td>
<td style="text-align:right;">
-0.022396
</td>
<td style="text-align:right;">
0.009731
</td>
<td style="text-align:right;">
-0.044003
</td>
<td style="text-align:right;">
-0.000688
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Ref
</td>
<td style="text-align:right;">
0.034275
</td>
<td style="text-align:right;">
0.050144
</td>
<td style="text-align:right;">
-0.004688
</td>
<td style="text-align:right;">
-0.004778
</td>
<td style="text-align:right;">
0.131802
</td>
<td style="text-align:right;">
0.166907
</td>
<td style="text-align:right;">
0.057388
</td>
<td style="text-align:right;">
0.102127
</td>
<td style="text-align:right;">
0.059059
</td>
<td style="text-align:right;">
0.056484
</td>
<td style="text-align:right;">
-0.057648
</td>
<td style="text-align:right;">
0.066085
</td>
<td style="text-align:right;">
0.066622
</td>
<td style="text-align:right;">
0.016471
</td>
<td style="text-align:right;">
0.064176
</td>
<td style="text-align:right;">
0.055367
</td>
<td style="text-align:right;">
0.036408
</td>
<td style="text-align:right;">
0.122793
</td>
<td style="text-align:right;">
0.145038
</td>
<td style="text-align:right;">
0.302531
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.771292
</td>
<td style="text-align:right;">
-0.019744
</td>
<td style="text-align:right;">
-0.056307
</td>
<td style="text-align:right;">
-0.013734
</td>
<td style="text-align:right;">
0.069932
</td>
<td style="text-align:right;">
-0.018020
</td>
<td style="text-align:right;">
0.067803
</td>
<td style="text-align:right;">
0.024671
</td>
<td style="text-align:right;">
-0.001641
</td>
<td style="text-align:right;">
0.004904
</td>
<td style="text-align:right;">
0.030090
</td>
<td style="text-align:right;">
0.032365
</td>
<td style="text-align:right;">
0.062154
</td>
<td style="text-align:right;">
0.004278
</td>
<td style="text-align:right;">
0.068863
</td>
<td style="text-align:right;">
-0.029914
</td>
<td style="text-align:right;">
-0.057880
</td>
<td style="text-align:right;">
0.010668
</td>
<td style="text-align:right;">
0.022560
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Ref
</td>
<td style="text-align:right;">
0.009128
</td>
<td style="text-align:right;">
0.001865
</td>
<td style="text-align:right;">
0.003597
</td>
<td style="text-align:right;">
0.003528
</td>
<td style="text-align:right;">
0.078040
</td>
<td style="text-align:right;">
0.055602
</td>
<td style="text-align:right;">
0.038989
</td>
<td style="text-align:right;">
0.066432
</td>
<td style="text-align:right;">
0.075147
</td>
<td style="text-align:right;">
0.057222
</td>
<td style="text-align:right;">
-0.057516
</td>
<td style="text-align:right;">
0.068632
</td>
<td style="text-align:right;">
0.076828
</td>
<td style="text-align:right;">
0.008986
</td>
<td style="text-align:right;">
0.061818
</td>
<td style="text-align:right;">
0.062323
</td>
<td style="text-align:right;">
0.034637
</td>
<td style="text-align:right;">
0.143177
</td>
<td style="text-align:right;">
0.174808
</td>
<td style="text-align:right;">
0.743695
</td>
<td style="text-align:right;">
0.771292
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.016165
</td>
<td style="text-align:right;">
-0.027648
</td>
<td style="text-align:right;">
-0.033751
</td>
<td style="text-align:right;">
0.043970
</td>
<td style="text-align:right;">
0.000543
</td>
<td style="text-align:right;">
0.084248
</td>
<td style="text-align:right;">
0.035268
</td>
<td style="text-align:right;">
0.011347
</td>
<td style="text-align:right;">
0.007659
</td>
<td style="text-align:right;">
0.046920
</td>
<td style="text-align:right;">
0.034169
</td>
<td style="text-align:right;">
0.078828
</td>
<td style="text-align:right;">
0.031153
</td>
<td style="text-align:right;">
0.059482
</td>
<td style="text-align:right;">
-0.038685
</td>
<td style="text-align:right;">
-0.035866
</td>
<td style="text-align:right;">
-0.021273
</td>
<td style="text-align:right;">
0.015494
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_00
</td>
<td style="text-align:right;">
-0.034806
</td>
<td style="text-align:right;">
0.010498
</td>
<td style="text-align:right;">
-0.010726
</td>
<td style="text-align:right;">
-0.010565
</td>
<td style="text-align:right;">
0.033342
</td>
<td style="text-align:right;">
0.027933
</td>
<td style="text-align:right;">
-0.009761
</td>
<td style="text-align:right;">
-0.029521
</td>
<td style="text-align:right;">
0.019250
</td>
<td style="text-align:right;">
0.085105
</td>
<td style="text-align:right;">
0.056363
</td>
<td style="text-align:right;">
-0.017893
</td>
<td style="text-align:right;">
-0.010751
</td>
<td style="text-align:right;">
0.010164
</td>
<td style="text-align:right;">
-0.068769
</td>
<td style="text-align:right;">
-0.068838
</td>
<td style="text-align:right;">
-0.020613
</td>
<td style="text-align:right;">
-0.027500
</td>
<td style="text-align:right;">
-0.047316
</td>
<td style="text-align:right;">
-0.008135
</td>
<td style="text-align:right;">
-0.019744
</td>
<td style="text-align:right;">
-0.016165
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.111633
</td>
<td style="text-align:right;">
-0.053994
</td>
<td style="text-align:right;">
-0.121670
</td>
<td style="text-align:right;">
-0.040561
</td>
<td style="text-align:right;">
0.001622
</td>
<td style="text-align:right;">
0.051973
</td>
<td style="text-align:right;">
0.052845
</td>
<td style="text-align:right;">
-0.060456
</td>
<td style="text-align:right;">
0.069942
</td>
<td style="text-align:right;">
-0.073754
</td>
<td style="text-align:right;">
-0.028909
</td>
<td style="text-align:right;">
-0.058983
</td>
<td style="text-align:right;">
0.003639
</td>
<td style="text-align:right;">
0.033611
</td>
<td style="text-align:right;">
0.036857
</td>
<td style="text-align:right;">
-0.004357
</td>
<td style="text-align:right;">
-0.006321
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_01
</td>
<td style="text-align:right;">
-0.059736
</td>
<td style="text-align:right;">
-0.024317
</td>
<td style="text-align:right;">
-0.017545
</td>
<td style="text-align:right;">
-0.016344
</td>
<td style="text-align:right;">
-0.098662
</td>
<td style="text-align:right;">
-0.101452
</td>
<td style="text-align:right;">
-0.069591
</td>
<td style="text-align:right;">
-0.197524
</td>
<td style="text-align:right;">
0.093583
</td>
<td style="text-align:right;">
-0.081919
</td>
<td style="text-align:right;">
0.039650
</td>
<td style="text-align:right;">
-0.010377
</td>
<td style="text-align:right;">
0.002155
</td>
<td style="text-align:right;">
-0.079871
</td>
<td style="text-align:right;">
-0.033433
</td>
<td style="text-align:right;">
-0.242088
</td>
<td style="text-align:right;">
-0.098456
</td>
<td style="text-align:right;">
-0.082880
</td>
<td style="text-align:right;">
-0.127711
</td>
<td style="text-align:right;">
0.016014
</td>
<td style="text-align:right;">
-0.056307
</td>
<td style="text-align:right;">
-0.027648
</td>
<td style="text-align:right;">
-0.111633
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.143379
</td>
<td style="text-align:right;">
-0.862869
</td>
<td style="text-align:right;">
-0.057529
</td>
<td style="text-align:right;">
0.071902
</td>
<td style="text-align:right;">
-0.038869
</td>
<td style="text-align:right;">
0.075383
</td>
<td style="text-align:right;">
0.086706
</td>
<td style="text-align:right;">
0.042382
</td>
<td style="text-align:right;">
0.065976
</td>
<td style="text-align:right;">
0.017374
</td>
<td style="text-align:right;">
0.028221
</td>
<td style="text-align:right;">
0.022201
</td>
<td style="text-align:right;">
-0.053361
</td>
<td style="text-align:right;">
-0.041691
</td>
<td style="text-align:right;">
-0.007019
</td>
<td style="text-align:right;">
0.014400
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_02
</td>
<td style="text-align:right;">
0.020061
</td>
<td style="text-align:right;">
0.136332
</td>
<td style="text-align:right;">
-0.010825
</td>
<td style="text-align:right;">
-0.010711
</td>
<td style="text-align:right;">
0.065453
</td>
<td style="text-align:right;">
0.112069
</td>
<td style="text-align:right;">
0.069884
</td>
<td style="text-align:right;">
0.008305
</td>
<td style="text-align:right;">
0.024710
</td>
<td style="text-align:right;">
-0.084612
</td>
<td style="text-align:right;">
-0.050606
</td>
<td style="text-align:right;">
0.011526
</td>
<td style="text-align:right;">
0.009963
</td>
<td style="text-align:right;">
-0.003032
</td>
<td style="text-align:right;">
0.054341
</td>
<td style="text-align:right;">
0.050257
</td>
<td style="text-align:right;">
0.053055
</td>
<td style="text-align:right;">
-0.051256
</td>
<td style="text-align:right;">
-0.081656
</td>
<td style="text-align:right;">
-0.022087
</td>
<td style="text-align:right;">
-0.013734
</td>
<td style="text-align:right;">
-0.033751
</td>
<td style="text-align:right;">
-0.053994
</td>
<td style="text-align:right;">
-0.143379
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.224413
</td>
<td style="text-align:right;">
-0.064641
</td>
<td style="text-align:right;">
-0.024985
</td>
<td style="text-align:right;">
-0.028471
</td>
<td style="text-align:right;">
-0.045725
</td>
<td style="text-align:right;">
0.012161
</td>
<td style="text-align:right;">
-0.026057
</td>
<td style="text-align:right;">
0.048419
</td>
<td style="text-align:right;">
0.032224
</td>
<td style="text-align:right;">
-0.025578
</td>
<td style="text-align:right;">
0.035828
</td>
<td style="text-align:right;">
-0.003964
</td>
<td style="text-align:right;">
-0.069670
</td>
<td style="text-align:right;">
0.054906
</td>
<td style="text-align:right;">
-0.028325
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_03
</td>
<td style="text-align:right;">
0.078133
</td>
<td style="text-align:right;">
-0.038886
</td>
<td style="text-align:right;">
-0.015792
</td>
<td style="text-align:right;">
-0.017172
</td>
<td style="text-align:right;">
0.062237
</td>
<td style="text-align:right;">
0.044059
</td>
<td style="text-align:right;">
0.049318
</td>
<td style="text-align:right;">
0.213303
</td>
<td style="text-align:right;">
-0.114299
</td>
<td style="text-align:right;">
0.061684
</td>
<td style="text-align:right;">
-0.041931
</td>
<td style="text-align:right;">
0.012241
</td>
<td style="text-align:right;">
-0.001337
</td>
<td style="text-align:right;">
0.078755
</td>
<td style="text-align:right;">
0.039706
</td>
<td style="text-align:right;">
0.262958
</td>
<td style="text-align:right;">
0.080751
</td>
<td style="text-align:right;">
0.116664
</td>
<td style="text-align:right;">
0.186483
</td>
<td style="text-align:right;">
-0.010245
</td>
<td style="text-align:right;">
0.069932
</td>
<td style="text-align:right;">
0.043970
</td>
<td style="text-align:right;">
-0.121670
</td>
<td style="text-align:right;">
-0.862869
</td>
<td style="text-align:right;">
-0.224413
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.174371
</td>
<td style="text-align:right;">
-0.065178
</td>
<td style="text-align:right;">
0.025081
</td>
<td style="text-align:right;">
-0.086819
</td>
<td style="text-align:right;">
-0.068748
</td>
<td style="text-align:right;">
-0.066113
</td>
<td style="text-align:right;">
-0.051454
</td>
<td style="text-align:right;">
-0.019418
</td>
<td style="text-align:right;">
0.005056
</td>
<td style="text-align:right;">
-0.040406
</td>
<td style="text-align:right;">
0.034135
</td>
<td style="text-align:right;">
0.046694
</td>
<td style="text-align:right;">
-0.017939
</td>
<td style="text-align:right;">
0.004442
</td>
</tr>
<tr>
<td style="text-align:left;">
LDA_04
</td>
<td style="text-align:right;">
-0.070240
</td>
<td style="text-align:right;">
0.017767
</td>
<td style="text-align:right;">
-0.009995
</td>
<td style="text-align:right;">
-0.009482
</td>
<td style="text-align:right;">
-0.006073
</td>
<td style="text-align:right;">
0.003766
</td>
<td style="text-align:right;">
-0.035778
</td>
<td style="text-align:right;">
-0.067658
</td>
<td style="text-align:right;">
0.033978
</td>
<td style="text-align:right;">
0.106625
</td>
<td style="text-align:right;">
0.034100
</td>
<td style="text-align:right;">
-0.007671
</td>
<td style="text-align:right;">
-0.006531
</td>
<td style="text-align:right;">
-0.012498
</td>
<td style="text-align:right;">
-0.041688
</td>
<td style="text-align:right;">
-0.118845
</td>
<td style="text-align:right;">
-0.007481
</td>
<td style="text-align:right;">
-0.033866
</td>
<td style="text-align:right;">
-0.069566
</td>
<td style="text-align:right;">
0.021728
</td>
<td style="text-align:right;">
-0.018020
</td>
<td style="text-align:right;">
0.000543
</td>
<td style="text-align:right;">
-0.040561
</td>
<td style="text-align:right;">
-0.057529
</td>
<td style="text-align:right;">
-0.064641
</td>
<td style="text-align:right;">
-0.174371
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.028771
</td>
<td style="text-align:right;">
0.040200
</td>
<td style="text-align:right;">
0.074045
</td>
<td style="text-align:right;">
-0.011084
</td>
<td style="text-align:right;">
0.072620
</td>
<td style="text-align:right;">
-0.040509
</td>
<td style="text-align:right;">
-0.001396
</td>
<td style="text-align:right;">
-0.021948
</td>
<td style="text-align:right;">
0.021952
</td>
<td style="text-align:right;">
0.032770
</td>
<td style="text-align:right;">
0.038164
</td>
<td style="text-align:right;">
0.013210
</td>
<td style="text-align:right;">
-0.018464
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Subj
</td>
<td style="text-align:right;">
-0.033672
</td>
<td style="text-align:right;">
0.123865
</td>
<td style="text-align:right;">
-0.050145
</td>
<td style="text-align:right;">
-0.047616
</td>
<td style="text-align:right;">
0.125892
</td>
<td style="text-align:right;">
0.129344
</td>
<td style="text-align:right;">
-0.040391
</td>
<td style="text-align:right;">
0.050824
</td>
<td style="text-align:right;">
0.642421
</td>
<td style="text-align:right;">
-0.058944
</td>
<td style="text-align:right;">
0.008122
</td>
<td style="text-align:right;">
-0.016100
</td>
<td style="text-align:right;">
-0.006473
</td>
<td style="text-align:right;">
-0.023834
</td>
<td style="text-align:right;">
-0.015055
</td>
<td style="text-align:right;">
-0.088549
</td>
<td style="text-align:right;">
-0.021499
</td>
<td style="text-align:right;">
-0.025358
</td>
<td style="text-align:right;">
-0.045873
</td>
<td style="text-align:right;">
0.051547
</td>
<td style="text-align:right;">
0.067803
</td>
<td style="text-align:right;">
0.084248
</td>
<td style="text-align:right;">
0.001622
</td>
<td style="text-align:right;">
0.071902
</td>
<td style="text-align:right;">
-0.024985
</td>
<td style="text-align:right;">
-0.065178
</td>
<td style="text-align:right;">
0.028771
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.239956
</td>
<td style="text-align:right;">
0.463402
</td>
<td style="text-align:right;">
0.257877
</td>
<td style="text-align:right;">
0.478553
</td>
<td style="text-align:right;">
0.197457
</td>
<td style="text-align:right;">
0.571707
</td>
<td style="text-align:right;">
0.222128
</td>
<td style="text-align:right;">
0.496662
</td>
<td style="text-align:right;">
-0.479637
</td>
<td style="text-align:right;">
-0.379408
</td>
<td style="text-align:right;">
-0.218913
</td>
<td style="text-align:right;">
0.088746
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Pol
</td>
<td style="text-align:right;">
-0.067251
</td>
<td style="text-align:right;">
-0.033395
</td>
<td style="text-align:right;">
-0.013104
</td>
<td style="text-align:right;">
-0.012570
</td>
<td style="text-align:right;">
0.053643
</td>
<td style="text-align:right;">
0.055643
</td>
<td style="text-align:right;">
-0.088776
</td>
<td style="text-align:right;">
-0.065404
</td>
<td style="text-align:right;">
0.190679
</td>
<td style="text-align:right;">
0.044189
</td>
<td style="text-align:right;">
0.094646
</td>
<td style="text-align:right;">
0.001244
</td>
<td style="text-align:right;">
0.013221
</td>
<td style="text-align:right;">
-0.033401
</td>
<td style="text-align:right;">
-0.112997
</td>
<td style="text-align:right;">
-0.118940
</td>
<td style="text-align:right;">
-0.025151
</td>
<td style="text-align:right;">
-0.006580
</td>
<td style="text-align:right;">
-0.033233
</td>
<td style="text-align:right;">
0.028372
</td>
<td style="text-align:right;">
0.024671
</td>
<td style="text-align:right;">
0.035268
</td>
<td style="text-align:right;">
0.051973
</td>
<td style="text-align:right;">
-0.038869
</td>
<td style="text-align:right;">
-0.028471
</td>
<td style="text-align:right;">
0.025081
</td>
<td style="text-align:right;">
0.040200
</td>
<td style="text-align:right;">
0.239956
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.473899
</td>
<td style="text-align:right;">
-0.540972
</td>
<td style="text-align:right;">
0.736400
</td>
<td style="text-align:right;">
-0.673328
</td>
<td style="text-align:right;">
0.442518
</td>
<td style="text-align:right;">
0.088313
</td>
<td style="text-align:right;">
0.360203
</td>
<td style="text-align:right;">
0.272885
</td>
<td style="text-align:right;">
0.298437
</td>
<td style="text-align:right;">
-0.054352
</td>
<td style="text-align:right;">
-0.026829
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Pos.Rate
</td>
<td style="text-align:right;">
-0.065763
</td>
<td style="text-align:right;">
0.107035
</td>
<td style="text-align:right;">
-0.029614
</td>
<td style="text-align:right;">
-0.027221
</td>
<td style="text-align:right;">
-0.014674
</td>
<td style="text-align:right;">
0.100075
</td>
<td style="text-align:right;">
-0.180249
</td>
<td style="text-align:right;">
0.113715
</td>
<td style="text-align:right;">
0.349747
</td>
<td style="text-align:right;">
-0.056011
</td>
<td style="text-align:right;">
0.112629
</td>
<td style="text-align:right;">
-0.008662
</td>
<td style="text-align:right;">
0.012457
</td>
<td style="text-align:right;">
-0.027311
</td>
<td style="text-align:right;">
-0.126159
</td>
<td style="text-align:right;">
-0.130456
</td>
<td style="text-align:right;">
-0.051786
</td>
<td style="text-align:right;">
-0.049908
</td>
<td style="text-align:right;">
-0.095431
</td>
<td style="text-align:right;">
0.026607
</td>
<td style="text-align:right;">
-0.001641
</td>
<td style="text-align:right;">
0.011347
</td>
<td style="text-align:right;">
0.052845
</td>
<td style="text-align:right;">
0.075383
</td>
<td style="text-align:right;">
-0.045725
</td>
<td style="text-align:right;">
-0.086819
</td>
<td style="text-align:right;">
0.074045
</td>
<td style="text-align:right;">
0.463402
</td>
<td style="text-align:right;">
0.473899
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.191531
</td>
<td style="text-align:right;">
0.574023
</td>
<td style="text-align:right;">
-0.232291
</td>
<td style="text-align:right;">
0.322832
</td>
<td style="text-align:right;">
-0.097316
</td>
<td style="text-align:right;">
0.437572
</td>
<td style="text-align:right;">
-0.187349
</td>
<td style="text-align:right;">
-0.183152
</td>
<td style="text-align:right;">
-0.052942
</td>
<td style="text-align:right;">
0.088136
</td>
</tr>
<tr>
<td style="text-align:left;">
Global.Neg.Rate
</td>
<td style="text-align:right;">
0.003340
</td>
<td style="text-align:right;">
0.182649
</td>
<td style="text-align:right;">
-0.020239
</td>
<td style="text-align:right;">
-0.018677
</td>
<td style="text-align:right;">
-0.025759
</td>
<td style="text-align:right;">
0.047307
</td>
<td style="text-align:right;">
0.014267
</td>
<td style="text-align:right;">
0.251790
</td>
<td style="text-align:right;">
0.208540
</td>
<td style="text-align:right;">
-0.112516
</td>
<td style="text-align:right;">
-0.029882
</td>
<td style="text-align:right;">
-0.017480
</td>
<td style="text-align:right;">
-0.018083
</td>
<td style="text-align:right;">
0.018019
</td>
<td style="text-align:right;">
0.054773
</td>
<td style="text-align:right;">
0.056105
</td>
<td style="text-align:right;">
0.021424
</td>
<td style="text-align:right;">
-0.030322
</td>
<td style="text-align:right;">
-0.030342
</td>
<td style="text-align:right;">
-0.000853
</td>
<td style="text-align:right;">
0.004904
</td>
<td style="text-align:right;">
0.007659
</td>
<td style="text-align:right;">
-0.060456
</td>
<td style="text-align:right;">
0.086706
</td>
<td style="text-align:right;">
0.012161
</td>
<td style="text-align:right;">
-0.068748
</td>
<td style="text-align:right;">
-0.011084
</td>
<td style="text-align:right;">
0.257877
</td>
<td style="text-align:right;">
-0.540972
</td>
<td style="text-align:right;">
0.191531
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.416919
</td>
<td style="text-align:right;">
0.794680
</td>
<td style="text-align:right;">
0.197575
</td>
<td style="text-align:right;">
0.003578
</td>
<td style="text-align:right;">
0.209045
</td>
<td style="text-align:right;">
-0.378526
</td>
<td style="text-align:right;">
-0.484382
</td>
<td style="text-align:right;">
0.130702
</td>
<td style="text-align:right;">
0.106005
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Pos
</td>
<td style="text-align:right;">
-0.063504
</td>
<td style="text-align:right;">
0.035816
</td>
<td style="text-align:right;">
-0.044156
</td>
<td style="text-align:right;">
-0.042074
</td>
<td style="text-align:right;">
0.086921
</td>
<td style="text-align:right;">
0.130231
</td>
<td style="text-align:right;">
-0.088639
</td>
<td style="text-align:right;">
-0.051550
</td>
<td style="text-align:right;">
0.600476
</td>
<td style="text-align:right;">
-0.011337
</td>
<td style="text-align:right;">
0.099875
</td>
<td style="text-align:right;">
0.007240
</td>
<td style="text-align:right;">
0.024610
</td>
<td style="text-align:right;">
-0.035932
</td>
<td style="text-align:right;">
-0.122321
</td>
<td style="text-align:right;">
-0.191420
</td>
<td style="text-align:right;">
-0.068211
</td>
<td style="text-align:right;">
-0.029354
</td>
<td style="text-align:right;">
-0.092884
</td>
<td style="text-align:right;">
0.046789
</td>
<td style="text-align:right;">
0.030090
</td>
<td style="text-align:right;">
0.046920
</td>
<td style="text-align:right;">
0.069942
</td>
<td style="text-align:right;">
0.042382
</td>
<td style="text-align:right;">
-0.026057
</td>
<td style="text-align:right;">
-0.066113
</td>
<td style="text-align:right;">
0.072620
</td>
<td style="text-align:right;">
0.478553
</td>
<td style="text-align:right;">
0.736400
</td>
<td style="text-align:right;">
0.574023
</td>
<td style="text-align:right;">
-0.416919
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.514523
</td>
<td style="text-align:right;">
0.397222
</td>
<td style="text-align:right;">
0.107540
</td>
<td style="text-align:right;">
0.416478
</td>
<td style="text-align:right;">
-0.051168
</td>
<td style="text-align:right;">
0.070150
</td>
<td style="text-align:right;">
-0.239487
</td>
<td style="text-align:right;">
-0.050102
</td>
</tr>
<tr>
<td style="text-align:left;">
Rate.Neg
</td>
<td style="text-align:right;">
0.028120
</td>
<td style="text-align:right;">
0.171411
</td>
<td style="text-align:right;">
-0.026000
</td>
<td style="text-align:right;">
-0.024820
</td>
<td style="text-align:right;">
0.052987
</td>
<td style="text-align:right;">
0.058484
</td>
<td style="text-align:right;">
0.131297
</td>
<td style="text-align:right;">
0.113680
</td>
<td style="text-align:right;">
0.318678
</td>
<td style="text-align:right;">
-0.102865
</td>
<td style="text-align:right;">
-0.082273
</td>
<td style="text-align:right;">
-0.015448
</td>
<td style="text-align:right;">
-0.020493
</td>
<td style="text-align:right;">
0.033118
</td>
<td style="text-align:right;">
0.104672
</td>
<td style="text-align:right;">
0.059070
</td>
<td style="text-align:right;">
0.028538
</td>
<td style="text-align:right;">
-0.017563
</td>
<td style="text-align:right;">
-0.013352
</td>
<td style="text-align:right;">
0.011169
</td>
<td style="text-align:right;">
0.032365
</td>
<td style="text-align:right;">
0.034169
</td>
<td style="text-align:right;">
-0.073754
</td>
<td style="text-align:right;">
0.065976
</td>
<td style="text-align:right;">
0.048419
</td>
<td style="text-align:right;">
-0.051454
</td>
<td style="text-align:right;">
-0.040509
</td>
<td style="text-align:right;">
0.197457
</td>
<td style="text-align:right;">
-0.673328
</td>
<td style="text-align:right;">
-0.232291
</td>
<td style="text-align:right;">
0.794680
</td>
<td style="text-align:right;">
-0.514523
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.195950
</td>
<td style="text-align:right;">
0.156028
</td>
<td style="text-align:right;">
0.120448
</td>
<td style="text-align:right;">
-0.377885
</td>
<td style="text-align:right;">
-0.465638
</td>
<td style="text-align:right;">
0.074616
</td>
<td style="text-align:right;">
0.027381
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Pos.Pol
</td>
<td style="text-align:right;">
-0.050488
</td>
<td style="text-align:right;">
0.182101
</td>
<td style="text-align:right;">
-0.044497
</td>
<td style="text-align:right;">
-0.042500
</td>
<td style="text-align:right;">
0.120099
</td>
<td style="text-align:right;">
0.151816
</td>
<td style="text-align:right;">
0.053317
</td>
<td style="text-align:right;">
0.110954
</td>
<td style="text-align:right;">
0.571425
</td>
<td style="text-align:right;">
-0.076176
</td>
<td style="text-align:right;">
-0.004230
</td>
<td style="text-align:right;">
-0.008878
</td>
<td style="text-align:right;">
-0.004585
</td>
<td style="text-align:right;">
0.020747
</td>
<td style="text-align:right;">
0.000739
</td>
<td style="text-align:right;">
-0.025027
</td>
<td style="text-align:right;">
0.035750
</td>
<td style="text-align:right;">
-0.008911
</td>
<td style="text-align:right;">
-0.022992
</td>
<td style="text-align:right;">
0.047079
</td>
<td style="text-align:right;">
0.062154
</td>
<td style="text-align:right;">
0.078828
</td>
<td style="text-align:right;">
-0.028909
</td>
<td style="text-align:right;">
0.017374
</td>
<td style="text-align:right;">
0.032224
</td>
<td style="text-align:right;">
-0.019418
</td>
<td style="text-align:right;">
-0.001396
</td>
<td style="text-align:right;">
0.571707
</td>
<td style="text-align:right;">
0.442518
</td>
<td style="text-align:right;">
0.322832
</td>
<td style="text-align:right;">
0.197575
</td>
<td style="text-align:right;">
0.397222
</td>
<td style="text-align:right;">
0.195950
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.407303
</td>
<td style="text-align:right;">
0.725481
</td>
<td style="text-align:right;">
-0.273989
</td>
<td style="text-align:right;">
-0.268862
</td>
<td style="text-align:right;">
-0.088163
</td>
<td style="text-align:right;">
0.025445
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Pos.Pol
</td>
<td style="text-align:right;">
-0.022095
</td>
<td style="text-align:right;">
-0.258499
</td>
<td style="text-align:right;">
-0.015589
</td>
<td style="text-align:right;">
-0.015560
</td>
<td style="text-align:right;">
-0.108779
</td>
<td style="text-align:right;">
-0.099542
</td>
<td style="text-align:right;">
-0.092352
</td>
<td style="text-align:right;">
-0.091255
</td>
<td style="text-align:right;">
0.243124
</td>
<td style="text-align:right;">
0.030341
</td>
<td style="text-align:right;">
-0.011315
</td>
<td style="text-align:right;">
0.032172
</td>
<td style="text-align:right;">
0.031879
</td>
<td style="text-align:right;">
-0.014794
</td>
<td style="text-align:right;">
0.014036
</td>
<td style="text-align:right;">
-0.020444
</td>
<td style="text-align:right;">
-0.026934
</td>
<td style="text-align:right;">
0.039030
</td>
<td style="text-align:right;">
0.028275
</td>
<td style="text-align:right;">
0.047951
</td>
<td style="text-align:right;">
0.004278
</td>
<td style="text-align:right;">
0.031153
</td>
<td style="text-align:right;">
-0.058983
</td>
<td style="text-align:right;">
0.028221
</td>
<td style="text-align:right;">
-0.025578
</td>
<td style="text-align:right;">
0.005056
</td>
<td style="text-align:right;">
-0.021948
</td>
<td style="text-align:right;">
0.222128
</td>
<td style="text-align:right;">
0.088313
</td>
<td style="text-align:right;">
-0.097316
</td>
<td style="text-align:right;">
0.003578
</td>
<td style="text-align:right;">
0.107540
</td>
<td style="text-align:right;">
0.156028
</td>
<td style="text-align:right;">
0.407303
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.014295
</td>
<td style="text-align:right;">
-0.031566
</td>
<td style="text-align:right;">
0.127585
</td>
<td style="text-align:right;">
-0.186335
</td>
<td style="text-align:right;">
-0.010259
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Pos.Pol
</td>
<td style="text-align:right;">
0.014117
</td>
<td style="text-align:right;">
0.433123
</td>
<td style="text-align:right;">
-0.043773
</td>
<td style="text-align:right;">
-0.041201
</td>
<td style="text-align:right;">
0.222730
</td>
<td style="text-align:right;">
0.243580
</td>
<td style="text-align:right;">
0.151287
</td>
<td style="text-align:right;">
0.171234
</td>
<td style="text-align:right;">
0.511031
</td>
<td style="text-align:right;">
-0.101028
</td>
<td style="text-align:right;">
-0.040097
</td>
<td style="text-align:right;">
-0.026470
</td>
<td style="text-align:right;">
-0.025497
</td>
<td style="text-align:right;">
0.024696
</td>
<td style="text-align:right;">
0.041788
</td>
<td style="text-align:right;">
-0.010054
</td>
<td style="text-align:right;">
0.042856
</td>
<td style="text-align:right;">
-0.044781
</td>
<td style="text-align:right;">
-0.052130
</td>
<td style="text-align:right;">
0.015913
</td>
<td style="text-align:right;">
0.068863
</td>
<td style="text-align:right;">
0.059482
</td>
<td style="text-align:right;">
0.003639
</td>
<td style="text-align:right;">
0.022201
</td>
<td style="text-align:right;">
0.035828
</td>
<td style="text-align:right;">
-0.040406
</td>
<td style="text-align:right;">
0.021952
</td>
<td style="text-align:right;">
0.496662
</td>
<td style="text-align:right;">
0.360203
</td>
<td style="text-align:right;">
0.437572
</td>
<td style="text-align:right;">
0.209045
</td>
<td style="text-align:right;">
0.416478
</td>
<td style="text-align:right;">
0.120448
</td>
<td style="text-align:right;">
0.725481
</td>
<td style="text-align:right;">
0.014295
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.285150
</td>
<td style="text-align:right;">
-0.406670
</td>
<td style="text-align:right;">
0.037476
</td>
<td style="text-align:right;">
0.020912
</td>
</tr>
<tr>
<td style="text-align:left;">
Avg.Neg.Pol
</td>
<td style="text-align:right;">
-0.018277
</td>
<td style="text-align:right;">
-0.155826
</td>
<td style="text-align:right;">
0.029402
</td>
<td style="text-align:right;">
0.027843
</td>
<td style="text-align:right;">
-0.102084
</td>
<td style="text-align:right;">
-0.107893
</td>
<td style="text-align:right;">
-0.029599
</td>
<td style="text-align:right;">
-0.124853
</td>
<td style="text-align:right;">
-0.361958
</td>
<td style="text-align:right;">
0.089525
</td>
<td style="text-align:right;">
0.077188
</td>
<td style="text-align:right;">
0.000491
</td>
<td style="text-align:right;">
0.002884
</td>
<td style="text-align:right;">
-0.016956
</td>
<td style="text-align:right;">
-0.068410
</td>
<td style="text-align:right;">
-0.037530
</td>
<td style="text-align:right;">
-0.041021
</td>
<td style="text-align:right;">
0.001110
</td>
<td style="text-align:right;">
-0.010945
</td>
<td style="text-align:right;">
-0.022396
</td>
<td style="text-align:right;">
-0.029914
</td>
<td style="text-align:right;">
-0.038685
</td>
<td style="text-align:right;">
0.033611
</td>
<td style="text-align:right;">
-0.053361
</td>
<td style="text-align:right;">
-0.003964
</td>
<td style="text-align:right;">
0.034135
</td>
<td style="text-align:right;">
0.032770
</td>
<td style="text-align:right;">
-0.479637
</td>
<td style="text-align:right;">
0.272885
</td>
<td style="text-align:right;">
-0.187349
</td>
<td style="text-align:right;">
-0.378526
</td>
<td style="text-align:right;">
-0.051168
</td>
<td style="text-align:right;">
-0.377885
</td>
<td style="text-align:right;">
-0.273989
</td>
<td style="text-align:right;">
-0.031566
</td>
<td style="text-align:right;">
-0.285150
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.731844
</td>
<td style="text-align:right;">
0.533210
</td>
<td style="text-align:right;">
-0.086476
</td>
</tr>
<tr>
<td style="text-align:left;">
Min.Neg.Pol
</td>
<td style="text-align:right;">
-0.048041
</td>
<td style="text-align:right;">
-0.494800
</td>
<td style="text-align:right;">
0.028622
</td>
<td style="text-align:right;">
0.026426
</td>
<td style="text-align:right;">
-0.219638
</td>
<td style="text-align:right;">
-0.221267
</td>
<td style="text-align:right;">
-0.158060
</td>
<td style="text-align:right;">
-0.212453
</td>
<td style="text-align:right;">
-0.296738
</td>
<td style="text-align:right;">
0.108599
</td>
<td style="text-align:right;">
0.109383
</td>
<td style="text-align:right;">
0.015810
</td>
<td style="text-align:right;">
0.025361
</td>
<td style="text-align:right;">
-0.042243
</td>
<td style="text-align:right;">
-0.120472
</td>
<td style="text-align:right;">
-0.073738
</td>
<td style="text-align:right;">
-0.068967
</td>
<td style="text-align:right;">
0.016887
</td>
<td style="text-align:right;">
0.001073
</td>
<td style="text-align:right;">
0.009731
</td>
<td style="text-align:right;">
-0.057880
</td>
<td style="text-align:right;">
-0.035866
</td>
<td style="text-align:right;">
0.036857
</td>
<td style="text-align:right;">
-0.041691
</td>
<td style="text-align:right;">
-0.069670
</td>
<td style="text-align:right;">
0.046694
</td>
<td style="text-align:right;">
0.038164
</td>
<td style="text-align:right;">
-0.379408
</td>
<td style="text-align:right;">
0.298437
</td>
<td style="text-align:right;">
-0.183152
</td>
<td style="text-align:right;">
-0.484382
</td>
<td style="text-align:right;">
0.070150
</td>
<td style="text-align:right;">
-0.465638
</td>
<td style="text-align:right;">
-0.268862
</td>
<td style="text-align:right;">
0.127585
</td>
<td style="text-align:right;">
-0.406670
</td>
<td style="text-align:right;">
0.731844
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
0.008805
</td>
<td style="text-align:right;">
-0.058170
</td>
</tr>
<tr>
<td style="text-align:left;">
Max.Neg.Pol
</td>
<td style="text-align:right;">
0.025921
</td>
<td style="text-align:right;">
0.258640
</td>
<td style="text-align:right;">
0.011031
</td>
<td style="text-align:right;">
0.011210
</td>
<td style="text-align:right;">
0.074829
</td>
<td style="text-align:right;">
0.060164
</td>
<td style="text-align:right;">
0.093962
</td>
<td style="text-align:right;">
0.093116
</td>
<td style="text-align:right;">
-0.197925
</td>
<td style="text-align:right;">
-0.001958
</td>
<td style="text-align:right;">
-0.014148
</td>
<td style="text-align:right;">
-0.013485
</td>
<td style="text-align:right;">
-0.019434
</td>
<td style="text-align:right;">
0.019293
</td>
<td style="text-align:right;">
0.038279
</td>
<td style="text-align:right;">
0.025348
</td>
<td style="text-align:right;">
0.020505
</td>
<td style="text-align:right;">
-0.024474
</td>
<td style="text-align:right;">
-0.025406
</td>
<td style="text-align:right;">
-0.044003
</td>
<td style="text-align:right;">
0.010668
</td>
<td style="text-align:right;">
-0.021273
</td>
<td style="text-align:right;">
-0.004357
</td>
<td style="text-align:right;">
-0.007019
</td>
<td style="text-align:right;">
0.054906
</td>
<td style="text-align:right;">
-0.017939
</td>
<td style="text-align:right;">
0.013210
</td>
<td style="text-align:right;">
-0.218913
</td>
<td style="text-align:right;">
-0.054352
</td>
<td style="text-align:right;">
-0.052942
</td>
<td style="text-align:right;">
0.130702
</td>
<td style="text-align:right;">
-0.239487
</td>
<td style="text-align:right;">
0.074616
</td>
<td style="text-align:right;">
-0.088163
</td>
<td style="text-align:right;">
-0.186335
</td>
<td style="text-align:right;">
0.037476
</td>
<td style="text-align:right;">
0.533210
</td>
<td style="text-align:right;">
0.008805
</td>
<td style="text-align:right;">
1.000000
</td>
<td style="text-align:right;">
-0.028663
</td>
</tr>
<tr>
<td style="text-align:left;">
Title.Subj
</td>
<td style="text-align:right;">
0.110429
</td>
<td style="text-align:right;">
-0.010097
</td>
<td style="text-align:right;">
-0.014008
</td>
<td style="text-align:right;">
-0.014138
</td>
<td style="text-align:right;">
-0.016109
</td>
<td style="text-align:right;">
-0.033539
</td>
<td style="text-align:right;">
0.000734
</td>
<td style="text-align:right;">
0.035936
</td>
<td style="text-align:right;">
-0.040130
</td>
<td style="text-align:right;">
0.006300
</td>
<td style="text-align:right;">
-0.018426
</td>
<td style="text-align:right;">
0.021930
</td>
<td style="text-align:right;">
0.019576
</td>
<td style="text-align:right;">
-0.001003
</td>
<td style="text-align:right;">
0.026432
</td>
<td style="text-align:right;">
0.027677
</td>
<td style="text-align:right;">
0.019656
</td>
<td style="text-align:right;">
0.021324
</td>
<td style="text-align:right;">
0.038152
</td>
<td style="text-align:right;">
-0.000688
</td>
<td style="text-align:right;">
0.022560
</td>
<td style="text-align:right;">
0.015494
</td>
<td style="text-align:right;">
-0.006321
</td>
<td style="text-align:right;">
0.014400
</td>
<td style="text-align:right;">
-0.028325
</td>
<td style="text-align:right;">
0.004442
</td>
<td style="text-align:right;">
-0.018464
</td>
<td style="text-align:right;">
0.088746
</td>
<td style="text-align:right;">
-0.026829
</td>
<td style="text-align:right;">
0.088136
</td>
<td style="text-align:right;">
0.106005
</td>
<td style="text-align:right;">
-0.050102
</td>
<td style="text-align:right;">
0.027381
</td>
<td style="text-align:right;">
0.025445
</td>
<td style="text-align:right;">
-0.010259
</td>
<td style="text-align:right;">
0.020912
</td>
<td style="text-align:right;">
-0.086476
</td>
<td style="text-align:right;">
-0.058170
</td>
<td style="text-align:right;">
-0.028663
</td>
<td style="text-align:right;">
1.000000
</td>
</tr>
</tbody>
</table>

The above table gives the correlations between all variables in the
Entertainment data set. This allows us to see which two variables have
strong correlation. If we have two variables with a high correlation, we
might want to remove one of them to avoid too much multicollinearity.

``` r
#Correlation graph for lifestyle_train
correlation_graph(data_channel_train)
```

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/r%20params$DataChannel%20corr_graph-1.png)<!-- -->

Because the correlation table above is large, it can be difficult to
read. The correlation graph above gives a visual summary of the table.
Using the legend, we are able to see the correlations between variables,
how strong the correlation is, and in what direction.

``` r
ggplot(shareshigh, aes(x=Rate.Pos, y=Rate.Neg,
                       color=Days_of_Week)) +
    geom_point(size=2)
```

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/scatterplot-1.png)<!-- -->

Once seeing the correlation table and graph, it is possible to graph two
variables on a scatterplot. This provides a visual of the linear
relationship. A scatterplot of two variables in the Entertainment
dataset has been created above.

``` r
## mean of shares 
mean(data_channel_train$shares)
```

    ## [1] 2919.18

``` r
## sd of shares 
sd(data_channel_train$shares)
```

    ## [1] 7642.04

``` r
## creates a new column that is if shares is higher than average or not 
shareshigh <- data_channel_train %>% select(shares) %>% mutate (shareshigh = (shares> mean(shares)))

## creates a contingency table of shareshigh and whether it is the weekend 
table(shareshigh$shareshigh, data_channel_train$Weekend)
```

    ##        
    ##            0    1
    ##   FALSE 3583  465
    ##   TRUE   714  177

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
    ##   Weekday 0.7254505 0.1445637
    ##   Weekend 0.0941486 0.0358372

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
    ##   Mon     0.1589391 0.0346224
    ##   Tues    0.1573193 0.0283458
    ##   Wed     0.1540798 0.0305730
    ##   Thurs   0.1423365 0.0281433
    ##   Fri     0.1127759 0.0228791
    ##   Weekend 0.0941486 0.0358372

After comparing shareshigh with whether or not the day was a weekend or
weekday, the above contingency table compares shareshigh for each
specific day of the week. Again, the frequencies are displayed as
relative frequencies.

``` r
ggplot(shareshigh, aes(x = Weekday, fill = shareshigh)) +
  geom_bar(aes(y = (after_stat(count))/sum(after_stat(count)))) + xlab('Weekday or Weekend?') + 
  ylab('Relative Frequency')
```

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/weekday%20bar%20graph-1.png)<!-- -->

``` r
ggplot(shareshigh, aes(x = Days_of_Week, fill = shareshigh)) +
  geom_bar(aes(y = (after_stat(count))/sum(after_stat(count)))) + xlab('Day of the Week') + 
  ylab('Relative Frequency')
```

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/day%20of%20the%20week%20graph-1.png)<!-- -->

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

    ## [1] " For Entertainment Mon is the most frequent day of the week"

``` r
table(shareshigh$shareshigh, g$Most_Freq)
```

    ##        
    ##         Most Freq Day Not Most Freq Day
    ##   FALSE           785              3263
    ##   TRUE            171               720

The above contingency table compares shareshigh to the Entertainment day
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

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/shares%20histogram-1.png)<!-- -->

Above we can see the frequency distribution of shares of the
Entertainment data channel. We should always see a long tail to the
right because a small number of articles will get a very high number of
shares. But looking at by looking at the distribution we can say how
many shares most of these articles got.

``` r
## creates plotting object with number of words in title and shares
b<- ggplot(data_channel_train, aes(x=n.Title, y=shares))

## creates a bar chart with number of words in title and shares 
b+ geom_col(fill="blue")+ ggtitle("Number of words in title vs shares") + labs(x="Number of words in title")
```

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/col%20graph-1.png)<!-- -->

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

    ##              LDA_02     Global.Neg.Rate              LDA_01 
    ##        -0.043137169        -0.027956371        -0.023706556 
    ##              LDA_00         Avg.Neg.Pol         Max.Neg.Pol 
    ##        -0.021786223        -0.018906678        -0.017940456 
    ##     Global.Pos.Rate                Tues         Min.Neg.Pol 
    ##        -0.017639789        -0.016839728        -0.014349194 
    ##            Rate.Neg        Min.Best.Key            n.Videos 
    ##        -0.013614282        -0.013568725        -0.011839265 
    ##              LDA_04               Thurs                 Wed 
    ##        -0.009295792        -0.008134914        -0.007609304 
    ##                 Mon             n.Other       Min.Worst.Key 
    ##        -0.005835920        -0.002957724         0.000381826 
    ##            Abs.Subj Rate.Unique.Nonstop                 Sat 
    ##         0.003405179         0.005462666         0.005470390 
    ##        Rate.Nonstop         Rate.Unique           Avg.Words 
    ##         0.005559726         0.005621983         0.005878974 
    ##                 Fri        Max.Best.Key         Max.Pos.Pol 
    ##         0.005962531         0.007840847         0.011271346 
    ##            Rate.Pos           Title.Pol             n.Title 
    ##         0.011585188         0.014913258         0.016375291 
    ##          Title.Subj          Global.Pol           n.Content 
    ##         0.017563672         0.020379969         0.021028707 
    ##         Min.Pos.Pol         Avg.Pos.Pol             Abs.Pol 
    ##         0.022339809         0.024736530         0.025206903 
    ##        Avg.Best.Key         Avg.Min.Key         Global.Subj 
    ##         0.025259011         0.026494805         0.028773403 
    ##            n.Images             Weekend             n.Links 
    ##         0.036901750         0.038132801         0.043079563 
    ##               n.Key                 Sun              LDA_03 
    ##         0.043397051         0.044065083         0.047838356 
    ##             Min.Ref             Max.Ref             Avg.Ref 
    ##         0.081749929         0.090539063         0.114569533 
    ##       Avg.Worst.Key         Avg.Avg.Key         Avg.Max.Key 
    ##         0.155524125         0.177707637         0.180810010 
    ##       Max.Worst.Key              shares 
    ##         0.182658217         1.000000000

``` r
## take the name of the highest correlated variable
highest_cor <-shares_correlations[52]  %>% names()

highest_cor
```

    ## [1] "Max.Worst.Key"

``` r
## creats scatter plot looking at shares vs highest correlated variable
g <-ggplot(data_channel_train,  aes(y=shares, x= data_channel_train[[highest_cor]])) 


g+ geom_point(aes(color=as.factor(Weekend))) +geom_smooth(method = lm) + ggtitle(" Highest correlated variable with shares") + labs(x="Highest correlated variable vs shares", color="Weekend")
```

    ## `geom_smooth()` using formula = 'y ~ x'

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/graph%20of%20shares%20with%20highest%20correlated%20var-1.png)<!-- -->

The above graph looks at the relationship between shares and the
variable with the highest correlation for the Entertainment data
channel, and colored based on whether or not it is the weekend. because
this is the most positively correlated variable we should always see an
upward trend but the more correlated they are the more the dots will
fall onto the line of best fit.

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

![](C:/Documents/Github/ST_558_Project_2/_Rmd/automations_test2_md/Entertainment_files/figure-gfm/boosted%20tree%20tuning-1.png)<!-- -->

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

    ## [1] " For Entertainment boosted_tree_RMSE is the winner"

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
