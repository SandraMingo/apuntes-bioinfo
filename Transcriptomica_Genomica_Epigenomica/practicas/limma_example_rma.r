# Install packages
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("SpikeInSubset")

# Load libraries
library(SpikeInSubset)
library(genefilter)
library(dplyr)
library(limma)

# Define variables
data(rma95)
fac <- factor(rep(1:2,each=3))
rtt <- rowttests(exprs(rma95),fac)

dim(exprs(rma95))
pData(rma95)
#las columnas son los genes spikes que se meten en la muestra. Siempre hay un cambio del doble en las muestras WT 
#que en las muestras KO salvo en un caso donde no hay expresión en una condición

mask <- with(rtt, abs(dm) < .2 & p.value < .01)
spike <- rownames(rma95) %in% colnames(pData(rma95))
cols <- ifelse(mask,"red",ifelse(spike,"dodgerblue","black"))
#pintamos de azul los genes spikes, y de rojo genes que tienen una diferencia de 
#medias entre los dos grupos pequeña (menor de 0.2) pero con un p-valor significativo

# Plot dataset
with(rtt, plot(-dm, -log10(p.value), cex=.8, pch=16,
               xlim=c(-1,1), ylim=c(0,5),
               xlab="difference in means",
               col=cols))
abline(h=2,v=c(-.2,.2), lty=2)

#vemos que los rojos, que no deberían ser significativos, lo son, y de los azules, 
#que deberían ser significativos, no todos lo son.

# variability of red dots
rtt$s <- apply(exprs(rma95), 1, function(row) sqrt(.5 * (var(row[1:3]) + var(row[4:6]))))
with(rtt, plot(s, -log10(p.value), cex=.8, pch=16,
               log="x",xlab="estimate of standard deviation",
               col=cols))

#En la línea del 1 (eje y), los genes son significativos.
#la variabilidad de los genes va a ser pequeña
#los spikes, que son genes que deberían ser muy diferencialmente expresados, 
#son significativos, pero la variabilidad es más normal

# limma
fit <- lmFit(rma95, design=model.matrix(~ fac))
colnames(coef(fit))

fit <- eBayes(fit) #shrinkage
tt <- topTable(fit, coef=2)
tt #tabla que sale también en Galaxy

# compare limma results with previous
limmares <- data.frame(dm=coef(fit)[,"fac2"], p.value=fit$p.value[,"fac2"])
with(limmares, plot(dm, -log10(p.value),cex=.8, pch=16,
                    col=cols,xlab="difference in means",
                    xlim=c(-1,1), ylim=c(0,5)))
abline(h=2,v=c(-.2,.2), lty=2)

#ahora, los puntitos rojos (genes con diferencia de medias pequeña y antes 
#p-valor significativo) ya no son significativos
#los spikes siguen siendo significativos en su mayoría.

# how does limma works
n <- 40 #40 genes cogidos al azar
qs <- seq(from=0,to=.2,length=n)
idx <- sapply(seq_len(n),function(i) which(as.integer(cut(rtt$s^2,qs)) == i)[1])
idx <- idx[!is.na(idx)]

par(mar=c(5,5,2,2))
plot(1,1,xlim=c(0,.21),ylim=c(0,1),type="n",
     xlab="variance estimates",ylab="",yaxt="n")
axis(2,at=c(.1,.9),c("before","after"),las=2)
segments((rtt$s^2)[idx],rep(.1,n),
         fit$s2.post[idx],rep(.9,n))

#estimación de la varianza antes y después de aplicar la estabilización de limma voom
#al aplicar la corrección, se obtiene el shrinkage: los valores grandes se corrigen 
#a valores pequeños, y los pequeños a valores más grandes
#esto se hace teniendo en cuenta la distribución de la varianza en todos los datos.