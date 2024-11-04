#setwd("~/GitHub/apuntes-bioinfo/Estadistica-R/ejercicios")
#PRSTR. Exercises R. Graphics

##1. The data
#Read the data “leukemia.data.txt”. Note that there are row names that we do not want to be
#part of the data itself (look at the help for “read.table”, the option “row.names”). Call this
#“leuk.dat” These data are based on the famous leukemia data from Golub et al. I’ve modified a few things
#below: the gene names and the sex of the patients are invented.
leuk.dat <- read.table('data/leukemia.data.txt', row.names = NULL)
leuk.dat

#That will be a data frame. For most figures your life will be simpler if you convert it into a
#matrix, using data.matrix. Call this data matrix “leuk.dat.m”. (Why is it a good idea to
#convert this into a matrix? Try doing the figures using the original data frame.)
leuk.dat.m <- data.matrix(leuk.dat)

#Read the data “leukemia.class.txt”. Use scan for this (do you know why?). Since these are
#labels, use what = "" or what = character() in the call to scan. Convert the
#classes into a factor. Call this “leuk.class”.
leuk.class <- scan(file = 'data/leukemia.class.txt' , what = "")
leuk.class <- factor(leuk.class)
leuk.class

#Create a vector for the sex of the patients. The patients are Male and Female, alternating,
#and staring with Male. Call this factor “sex”.
sex <- factor(rep(c("Male", "Female"), length.out = 38))
sex

## 2. The boxplot of PTEN and scatterplots with lots of info
library(colorspace)
#The boxplot
pten_genedata <- leuk.dat.m[2124, 2:39]
boxplot(pten_genedata ~ leuk.class, col = c('orange', 'lightblue'),
        xlab = 'Patient groups', ylab = 'Gene expression (mRNA)')
title('Boxplot of PTEN by patient group')

#PTEN, HK-1, a third gene, patient status, and some sex
hk1_genedata <- leuk.dat.m[1, 2:39]
third_genedata <- leuk.dat.m[2600, 2:39]

for.cex <- leuk.dat.m[2600, 2:39] - min(leuk.dat.m[2600, 2:39]) + 1
the.cex <-   2 * for.cex/max(for.cex)

plot(hk1_genedata ~ pten_genedata, col = c('blue', 'red')[leuk.class],
     pch = c(21, 24)[sex], cex = the.cex,
     xlab = 'PTEN', ylab = 'HK-1',
     main ='HK-1 vs. PTEN; symbol size proportional to gene 2600')

lclass <- rep(levels(leuk.class), rep(2, 2))
lsex <- rep(levels(sex), 2)
text.legend <- paste(lclass, lsex, sep = ", ")

legend(-1, 1, legend = text.legend,
       pch = c(21, 24)[factor(lsex)],
       col = diverge_hcl(2)[factor(lclass)])

abline(lm(leuk.dat.m[1, 2:39] ~ leuk.dat.m[2124, 2:39]), lty = 2)

## 3. Conditioning plots

