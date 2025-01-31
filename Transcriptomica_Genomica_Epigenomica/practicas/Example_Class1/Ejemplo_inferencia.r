#install.packages(c("downloader","dplyr","lattice","beeswarm","WDI","ggplot2","ShortRead"))
library(dplyr)
library(lattice)
library(beeswarm)
library(ggplot2)
#library(ShortRead)


#####################################################################
#                 INTRO
#####################################################################

#####################################################################
#                 MICE DATA
#####################################################################
setwd("C:/Users/Sandra/Documents/GitHub/apuntes-bioinfo/Transcriptomica_Genomica_Epigenomica/practicas/Example_Class1/") # change your path
dir=getwd()
library(downloader) 
library(dplyr)

############# Download data #########################################

# url <- "https://raw.githubusercontent.com/genomicsclass/dagdata/master/inst/extdata/femaleMiceWeights.csv"
# filename <- "femaleMiceWeights.csv" 
# download(url, destfile=paste(dir,"/",filename,sep=""))
# # 
# url <- "https://raw.githubusercontent.com/genomicsclass/dagdata/master/inst/extdata/femaleControlsPopulation.csv"
# filename <- "femaleControlsPopulation.csv" 
# download(url, destfile=paste(dir,"/",filename,sep=""))
# # 
# url <- "https://raw.githubusercontent.com/genomicsclass/dagdata/master/inst/extdata/mice_pheno.csv"
# filename <- "mice_pheno.csv" 
# download(url, destfile=paste(dir,"/",filename,sep=""))

############# EDA #########################################

pheno=read.csv("mice_pheno.csv")
View(pheno)

class(pheno)

which(pheno[,2]=="hf")

length(which(pheno[,2]=="hf"))
length(which(pheno[,2]=="chow"))

hf.data=pheno[which(pheno[,2]=="hf"),]
chow.data=pheno[which(pheno[,2]=="chow"),]

summary(hf.data$Bodyweight)
summary(chow.data$Bodyweight)

#several representations
? barplot

error.bar <- function(x, y, upper, lower=upper, length=0.1,...){
  if(length(x) != length(y) | length(y) !=length(lower) | length(lower) != length(upper))
    stop("vectors must be same length")
  arrows(x,y+upper, x, y-lower, angle=90, code=3, length=length, ...)
}

y.means=c(mean(chow.data[,3],na.rm=T),mean(hf.data[,3],na.rm=T))
y.sd=c(sqrt(var(chow.data[,3],na.rm=T)),sqrt(var(hf.data[,3],na.rm=T)))
y.se=y.sd/sqrt(c(length(chow.data[,3]),length(hf.data[,3])))
X11()
barx=barplot(y.means)
error.bar(barx,y.means, y.se)

X11()
# BOXPLOT
boxplot(pheno[,"Bodyweight"]~pheno[,"Diet"])

X11()
library(lattice)
bwplot(pheno$Bodyweight ~ pheno$Diet|pheno$Sex,
       ylab="Bodyweight", xlab="",
       main="",
       #layout=(c(1,3))
)
X11()
bwplot(pheno$Bodyweight ~ pheno$Sex|pheno$Diet,
       ylab="Bodyweight", xlab="",
       main="",
       #layout=(c(1,3))
)

library(beeswarm)
X11()
beeswarm(Bodyweight ~ Diet, data = pheno, pch = 16,
         pwcol = as.numeric(Sex), xlab = '', 
         ylab = 'Body Weight', 
         labels = c('Chow', 'HF'))
boxplot(Bodyweight ~ Diet, data = pheno, add = T,
        names = c("",""), col="#0000ff22")  
legend('topright', legend = levels(pheno$Sex), title = 'Sex', 
       pch = 16, col = 1:2)

######################################################
# random variables
########################################################
dat <- read.csv("femaleMiceWeights.csv")
head(dat)
View(dat)

# sample of 12 mice under chow diet and 12 mice under HF diet
control <- filter(dat,Diet=="chow") %>% select(Bodyweight) %>% unlist
treatment <- filter(dat,Diet=="hf") %>% select(Bodyweight) %>% unlist

control
treatment

summary(control)
summary(treatment)

X11()
par(mfrow=c(1,2))
hist(control)
hist(treatment)

print( mean(treatment) )
print( mean(control) )
obsdiff <- mean(treatment) - mean(control)
print(obsdiff)

#whole population
population <- read.csv("femaleControlsPopulation.csv")
##use unlist to turn it into a numeric vector
population <- unlist(population) 
mean(population)
#[1] 23.89338

summary(chow.data[,3])
summary(chow.data[which(chow.data$Sex=="F"),3])

#lets sample 12 mice from the control population repeatedly
control <- sample(population,12)
mean(control)

control <- sample(population,12)
mean(control)

control <- sample(population,12)
mean(control)

ctr.mean=numeric()
for (i in 1:1000){
  ctr.mean[i]=mean(sample(population,12))
}

par(mfrow=c(1,2))
myhist=hist(ctr.mean,xlim=c(16,30))
mydensity <- density(ctr.mean)
multiplier <- myhist$counts / myhist$density
mydensity$y <- mydensity$y * multiplier[1]
summary(ctr.mean)

lines(mydensity,col="red")

# what would happen if we take samples of 5 elements
ctr.mean1=numeric()
for (i in 1:1000){
  ctr.mean1[i]=mean(sample(population,5))
}
mean(ctr.mean1)
myhist=hist(ctr.mean1,xlim=c(16,30))
mydensity <- density(ctr.mean1)
multiplier <- myhist$counts / myhist$density
mydensity$y <- mydensity$y * multiplier[1]
lines(mydensity,col="red")
summary(ctr.mean1)

X11()
boxplot(data.frame(N5=ctr.mean1,N12=ctr.mean),notch=T)


######################################################
# null distribution
########################################################
dat <- read.csv("femaleMiceWeights.csv")
head(dat)
View(dat)
control <- filter(dat,Diet=="chow") %>% select(Bodyweight) %>% unlist
treatment <- filter(dat,Diet=="hf") %>% select(Bodyweight) %>% unlist
print(mean(treatment))
print(mean(control))
obs <- mean(treatment) - mean(control)
print(obs)

# if we assume the null hypothesis is that there are no difference between the groups
# lets see what is in general the difference in body weight between samples of 12 animals

#population is the set of 225 female control mice
diff.null=numeric()
ns=10000
for (i in 1:ns){
  control=sample(population,12)
  treatment=sample(population,12)
  diff.null[i]=mean(treatment)-mean(control)
}

summary(diff.null)

max(diff.null)
min(diff.null)
mean(diff.null)
median(diff.null)
sqrt(var(diff.null))

X11()
par(mfrow=c(1,2))
myhist=hist(diff.null)
mydensity <- density(diff.null)
multiplier <- myhist$counts / myhist$density
mydensity$y <- mydensity$y * multiplier[1]

lines(mydensity,col="red")
qqplot(diff.null,rnorm(ns,mean(diff.null),sqrt(var(diff.null))))
abline(a=0,b=1,col="red")


#La mayoría de veces la diferencia es 0, pero algunas sale la cola de normalidad

#Probability of getting a value larger than three under the null
length(which(diff.null>=3)) #3 es la diferencia que teníamos
length(which(diff.null>=3))/ns
#pese a que los grupos sean iguales, hay esos ratones que presentan una diferencia significativa

######################################################
# t-test
########################################################
#imagine pheno contains all the population of mice under HF and chow diets
dat <- read.csv("femaleMiceWeights.csv")
head(dat)
View(dat)

# sample of 12 mice under chow diet and 12 mice under HF diet
control <- filter(dat,Diet=="chow") %>% select(Bodyweight) %>% unlist
treatment <- filter(dat,Diet=="hf") %>% select(Bodyweight) %>% unlist

obs=mean(treatment)-mean(control)

sigma2.x=var(treatment,na.rm=T)
sigma2.y=var(control,na.rm=T)

se=sqrt(sigma2.x/length(treatment)+sigma2.y/length(control))
tstat=obs/se
1-pnorm(tstat)
#0.0199311

t.test(treatment,control)


pop.control <- filter(pheno,Diet=="chow") %>% select(Bodyweight) %>% unlist
pop.treatment <- filter(pheno,Diet=="hf") %>% select(Bodyweight) %>% unlist

pn=tn=numeric()
for (i in 1:1000){
  treatment=sample(pop.treatment,200)
  control=sample(pop.control,200)
  obs=mean(treatment,na.rm=T)-mean(control,na.rm=T)
  se=sqrt(var(treatment,na.rm=T)/length(treatment)+var(control,na.rm=T)/length(control))
  pn[i]=1-pnorm(obs/se)
  tn[i]=t.test(treatment,control)[[3]]
}
plot(pn,tn)
abline(0,1,col="red")

##########################################
# CI
##########################################
dat <- read.csv("mice_pheno.csv")
chowPopulation <- dat[dat$Sex=="F" & dat$Diet=="chow",3]

mu_chow <- mean(chowPopulation)
print(mu_chow)

N <- 30
chow <- sample(chowPopulation,N)
print(mean(chow))

se <- sd(chow)/sqrt(N)
print(se)

pnorm(2) - pnorm(-2)

Q <- qnorm(1- 0.05/2)
interval <- c(mean(chow)-Q*se, mean(chow)+Q*se )
interval
interval[1] < mu_chow & interval[2] > mu_chow

X11()
par(mfrow=c(1,3))
B <- 250

plot(mean(chowPopulation)+c(-7,7),c(1,1),type="n",
     xlab="weight",ylab="interval",ylim=c(1,B))
abline(v=mean(chowPopulation))
for (i in 1:B) {
  chow <- sample(chowPopulation,N)
  se <- sd(chow)/sqrt(N)
  interval <- c(mean(chow)-Q*se, mean(chow)+Q*se)
  covered <- 
    mean(chowPopulation) <= interval[2] & mean(chowPopulation) >= interval[1]
  color <- ifelse(covered,1,2)
  lines(interval, c(i,i),col=color)
}

# Small sample size
plot(mean(chowPopulation)+c(-7,7),c(1,1),type="n",
     xlab="weight",ylab="interval",ylim=c(1,B))
abline(v=mean(chowPopulation))
Q <- qnorm(1- 0.05/2)
N <- 5
for (i in 1:B) {
  chow <- sample(chowPopulation,N)
  se <- sd(chow)/sqrt(N)
  interval <- c(mean(chow)-Q*se, mean(chow)+Q*se)
  covered <- mean(chowPopulation) <= interval[2] & mean(chowPopulation) >= interval[1]
  color <- ifelse(covered,1,2)
  lines(interval, c(i,i),col=color)
}

#for such small N the sample mean is not normal but a students t---> use qt
plot(mean(chowPopulation) + c(-7,7), c(1,1), type="n",
     xlab="weight", ylab="interval", ylim=c(1,B))
abline(v=mean(chowPopulation))
##Q <- qnorm(1- 0.05/2) ##no longer normal so use:
Q <- qt(1- 0.05/2, df=4)
N <- 5
for (i in 1:B) {
  chow <- sample(chowPopulation, N)
  se <- sd(chow)/sqrt(N)
  interval <- c(mean(chow)-Q*se, mean(chow)+Q*se )
  covered <- mean(chowPopulation) <= interval[2] & mean(chowPopulation) >= interval[1]
  color <- ifelse(covered,1,2)
  lines(interval, c(i,i),col=color)
}

qt(1- 0.05/2, df=4)
