#RUBIN-CAUSAL-MODEL SIMULATION
# example alcohol consumption and time-to-death

############ HYPOTHETICAL SCENARIO #############################

ATE<-numeric() #average treatment effect

for (i in 1:10000){
  no.alcohol.true<-rpois(100,7)
  alcohol.true<-rpois(100,5)
  
  ATE[i]<-mean(no.alcohol.true-alcohol.true)
}

hist(ATE)
summary(ATE)

############ RANDOM ASSIGNMENT OF INDIVIDUALS ##################

ace.random<-numeric() #average causal effect

for (i in 1:10000){
  assignment<-rbinom(200,1,0.5)
  n1<-sum(assignment==1)
  n0<-sum(assignment==0)
  no.alcohol<-rpois(n0,7)
  alcohol<-rpois(n1,5)
  ace.random[i]<-mean(no.alcohol)-mean(alcohol)
}

hist(ace.random)
summary(ace.random)
#La media y mediana es muy similar a ATE

############ NON-RANDOM ASSIGNMENT OF INDIVIDUALS ##################

#smokers tend to drink and live shorter than non-smokers

ace.nonrandom<-numeric()

for (i in 1:10000){
  alc.ass<-c(rep(0,100),rep(1,100))
  smk.ass<-rbinom(200,1,0.3)
  tt<-table(alc.ass,smk.ass)
  no.alcohol.no.smk<-rpois(tt[1,1],15)
  no.alcohol.smk<-rpois(tt[1,2],10)
  alcohol.no.smk<-rpois(tt[2,1],10)
  alcohol.smk<-rpois(tt[2,2],5)
  ace.nonrandom[i]<-mean(c(no.alcohol.no.smk,no.alcohol.smk))-mean(c(alcohol.no.smk,alcohol.smk))
}

hist(ace.nonrandom)
summary(ace.nonrandom)
