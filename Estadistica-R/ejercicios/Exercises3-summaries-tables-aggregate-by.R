#setwd("~/GitHub/apuntes-bioinfo/Estadistica-R/ejercicios")
#PRSTR. Exercises R. Summaries, tables, aggregate, by.

## The data
leuk.dat <- read.table('data/leukemia.data.txt', row.names = 1)
leuk.dat.m <- data.matrix(leuk.dat)

leuk.class <- factor(scan('data/leukemia.class.txt', what = ''))

sex <- factor(rep(c('Male', 'Female'), length.out = 38))

complete_data <- as.data.frame(t(leuk.dat.m))
complete_data$sex <- sex  
complete_data$class <- leuk.class

rm(c(leuk.class, sex, leuk.dat, leuk.dat.m))

## 1. Tables and cross-tabs
table(complete_data$sex)

table(complete_data$sex, complete_data$class)

xtabs( ~ sex + class, data = complete_data)

(my_table <- as.data.frame(xtabs(~sex+class, data = complete_data)))

## 2. Two subsetting operations
pvals <- apply(leuk.dat, 1, function(x) t.test(x ~ leuk.class)$p.value)
mean(leuk.dat.m[pvals < 0.01, 3])

median(leuk.dat.m[2, sex == 'Male'])

## 3. Gene summaries by condition and sex
leuk.dat.t <- t(leuk.dat.m)

aggregate(leuk.dat.t[, c(1, 2124, 2600)], 
          list(type = leuk.class), median)

aggregate(leuk.dat.t[, c(1, 2124, 2600)], 
          list(type = leuk.class, sex = sex), median)

all.median <- aggregate(leuk.dat.t, 
                        list(type = leuk.class, sex = sex), median)
all.median[,1:10]

dim(all.median)

aggregate(leuk.dat.t[,c(1,2124,2600)],
          list(type = leuk.class, sex = sex), 
          function(x) c(mean = mean(x), sd = sd(x)))

by(leuk.dat.t[, c(1, 2124, 2600)], 
   list(type = leuk.class, sex = sex), summary)

aggregate(leuk.dat.t[,c(1,2124,2600)], 
          list(type = leuk.class, sex = sex), summary)
