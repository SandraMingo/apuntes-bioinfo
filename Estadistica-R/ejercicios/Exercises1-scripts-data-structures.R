#setwd("~/GitHub/apuntes-bioinfo/Estadistica-R/ejercicios")
#PRSTR. Exercises R. Scripts, data structures, reading data

## 1. Scripts
###1.1 Creating the script
#Create a variable x and assign to it a sequence of numbers from 123 to 297 in jumps of 3.5.
x <- seq(from = 123, to = 297, by = 3.5)

#Create a second variable y storing the 10 first elements of x
y <- x[1:10]

#Create a second variable y storing the numbers of x that are located in odd positions of vector x
y <- x[seq(1, length(x), by = 2)]

#Obtain a summary of that variable x
summary(x)

#Obtain a summary of variables x and y separately. Check if the mean of x is equal to the median of y
summary_x <- summary(x)
summary_y <- summary(y)
summary_x['Mean'] == summary_y['Median']

## 2. Some read.table operations
#Take the file AnotherDataSet.txt and create the following different data sets:
#d1.txt, which is like AnotherDataSet but with the 23.4 substituted by an NA 
d1 <- read.table('data/AnotherDataSet.txt', header = TRUE)
d1[d1 == 23.4] <- NA
write.table(d1, file = "d1.txt")
read.table('d1.txt', header = TRUE, na.strings = NA)

#d2.txt, which is like d1.txt, but leave a blank instead of 23.4 and a blank space instead of the 14 in the second row and column.
d2 <- read.table('data/AnotherDataSet.txt', header = TRUE)
d2[d2 == 23.4] <-  ''
d2[2, 2] <-  ''
write.table(d2, file = "d2.txt")
read.table('d1.txt', header = TRUE, na.strings = '')

#d3.txt, which is the same as d2.txt but without the header.
d3 <- read.table('data/AnotherDataSet.txt', header = FALSE)
d3[d3 == 23.4] <-  ''
d3[2, 2] <-  ''
write.table(d3, file = "d3.txt")
read.table('d1.txt', header = TRUE, na.strings = '')

## 5. Saving objects
#Create an object called x with value 97
x <- 97

#Create an object called y with value 95
y <- 95

#Save a binary R object, called oneObject.RData that will contain only object x
saveRDS(x, file = 'oneObject.RData')

#What would have been the difference of doing save.image()?
#save.image() saves all the object in the current R session

## 6. Vectors, data frames, etc
#Read file heart_rates.txt and store it on a variable named df1. Check the class of object of 
#df1 and convert it into a vector named hr
df1 <- read.table("data/heart_rates.txt", header = TRUE)
class(df1) #dataframe
hr <- df1[["Heart_Rates"]]

#Add at the begining of the vector “hr” the following heart rates: 87, 78, 86, 62, 69, 69, 68, 67, 75, 76.
hr <- c(87, 78, 86, 62, 69, 69, 68, 67, 75, 76, df1$Heart_Rates)

#You are told the first three observations are from individuals of age 11, then you have two
#observations from individuals of age 63, then you have four observations from individuals
#of age 40, one observation from an individual of age 47, ten observations from individuals
#of age 55, forty individuals of age 90, thirty individuals of age 22, fifteen individuals of age
#30 and then five individuals of age 74. Create a vector named age.
age <- c(rep(11, 3), rep(63, 2), rep(40, 4), 47, rep(55, 10), rep(90, 40), rep(22, 30), rep(30, 15), rep(74, 5))

#Show the code to obtain the percentage of individuals who are less than 45 years old. Use both vectors, age and hr, to do it.
length(age[age<45]) * 100 / length(age)

#Now, we will use, for age and hr, only the first ten observations. In other words, create a
#vector “hr” that contains the first observations of the former hr vector, and an “age” vector
#that contains the first 10 observations of the former “age” vector.
hr <- hr[1:10]
age <- age[1:10]

#Create, using “hr” and “age” (the newly created hr and age) a new vector that contains the
#observations of the individuals who are 63 and 47 years old; call this “hr2”. The names
#of these individuals are Juan, Ana, Carmen; place the names of the individuals in the vector.
hr2 <- c(hr[age == 63], hr[age==47])
hr2 <- c('Juan' = hr2[1], 'Ana' = hr2[2], 'Carmen' = hr2[3])
age
#Using the names, obtain the heart rates of Juan and Ana.
hr2['Juan']
hr2['Ana']

#Create a matrix for Juan, Ana, and Carmen that contains, as columns, their hear rate and
#their age, and use their names as row names.
cbind('HR' = hr2, 'Age' = c('Juan' = 63, 'Ana' = 63, 'Carmen' = 47))

#Repeat the above, but create a data frame, not a matrix, and create the data frame from
#scratch (not via a matrix)
df2 <- data.frame('HR' = hr2, 'Age'=c('Juan' = 63, 'Ana' = 63, 'Carmen' = 47), row.names = c('Juan', 'Ana', 'Carmen'))
df2
#Repeat the above, but do not use their names as row names, but just as another column.
#Could you do this with a matrix?
df3 <- data.frame('Names' = c('Juan', 'Ana', 'Carmen'),'HR' = hr2, 'Age'=c('Juan' = 63, 'Ana' = 63, 'Carmen' = 47), row.names = NULL)
matrix1 <- cbind('Names'= c('Juan', 'Ana', 'Carmen'), 'HR' = hr2, 'Age' = c('Juan' = 63, 'Ana' = 63, 'Carmen' = 47))

#Using the matrix, obtain the heart rate of Juan. Do it using indices and using row names
matrix1['Juan', 'HR']
matrix1[1,2]

#Using the matrix, obtain all the values associated to Juan (heart rate and age).
matrix1['Juan', ]

#Using the data frame with names as row names, obtain the heart rate of Ana.
df2[2, 1]
df2['Ana', 'HR']

#Using the data frame with names as another column, obtain the heart rate of Ana.
df3[df3$Names == 'Ana', 'HR']

#Using the data frame with names as another column, obtain all the values associate to Ana
#(her heart rate, her age, and her name).
df3[df3$Names == 'Ana', ]

#Using the data frame with names as another column, obtain all the values (i.e. another data
#frame) for all individuals older than 60.
df4 <- df3[df3$Age > 60, ]
df4

#Using the data frame with names as row names, obtain all the values (i.e. another data
#frame) for all individuals older than 60.
df5 <- df2[df2$Age > 60, ]
df5

#Create a matrix of heart rates and ages for all the original measures for individuals who are
#older than 15. The simplest thing might be to create a matrix for all, and then create another
#matrix by subsetting this one.
m1 <- cbind(hr, age)
m2 <- m1[m1[ ,2] > 15, ]

#Do you think those sexes can correspond to the individuals above? Look in particular to the
#individual of age 47, for which I gave you the name (and, thus, implicitly sex) above.
sex <- c("M", "F")[c(1, 2, 1, 1, 2, 2, 1, 1, 2, 1)]

df2[df2$Age == 47, ] #Carmen
sex[age == 47] #M
#The sexes do not correspond - Carmen should be F, not M

#Correct that value
sex[age == 47] <- "F"
sex

#Using the previous vector of sex, add it to the matrix of heart rates and ages 
#for the individuals that are older than 15 (see 6.18). I want you to combine (or 
#something of that kind); I do not want you to create a new matrix from scratch.
m3 <- cbind(m2, sex[m1[ ,2] > 15])

#Do the same thing with the data frame in 6.9 (df2); i.e., create a new column that will contain 
#the sexes of those individuals (so those individuals of ages 63 and 47).
df2$Sex <- c(sex[age == 63], sex[age == 47])
df2

#Now you have received the corresponding identifications numbers for the 10 patients 
#as follows: "A1,A2,A3,A4,A5,B1,B2,B3,B4,B5", you can copy it directly or do it programaticaly
#in a fancy way, take a look to paste0 function. Create a dataframe named df4, that contains 4
#columns named as follows: "ID", "Age","Sex","Heart_Rate", where age, sex, and heart rate
#are from the vectors “age”, “sex”, and “hr” we have just used (the ones for 10 individuals).
id <- c('A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5')
ddf4 <- data.frame('ID' = id, 'Age' = age, 'Sex' = sex, 'Heart_rate' = hr)
ddf4

#Now you have received the information about the oxygen consumption in ml of oxigen per
#minute, contained in file "Oxygen.txt". Read that table, and obtain a final dataframe named
#df_final with all the information: ID, age, sex, heart and oxygen consumption. Take a look
#to merge function or at function left_join in package dplyr
library(dplyr)
oxy <- read.table('data/Oxygen.txt', header = TRUE)
left_join(ddf4, oxy)
