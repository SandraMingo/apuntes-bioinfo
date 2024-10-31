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
d1 <- read.table('AnotherDataSet.txt', header = TRUE)
d1[d1 == 23.4] <- NA
write.table(d1, file = "d1.txt")
read.table('d1.txt', header = TRUE, na.strings = NA)

#d2.txt, which is like d1.txt, but leave a blank instead of 23.4 and a blank space instead of the 14 in the second row and column.
d2 <- read.table('AnotherDataSet.txt', header = TRUE)
d2[d2 == 23.4] <-  ''
d2[2, 2] <-  ''
write.table(d2, file = "d2.txt")
read.table('d1.txt', header = TRUE, na.strings = '')

#d3.txt, which is the same as d2.txt but without the header.
d3 <- read.table('AnotherDataSet.txt', header = FALSE)
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
