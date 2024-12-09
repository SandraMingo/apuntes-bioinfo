library(MendelianRandomization)

# “Genetic variants influencing circulating lipid levels and risk of coronary artery
# disease”
# doi: 10.1161/atvbaha.109.201020
#En el paper podemos buscar la tabla con cada SNP, beta y el error estándar
MRInputObject <- mr_input(bx = ldlc, #beta de 28 SNP que se asocian con la cantidad de lípidos y no con la enfermedad coronaria
                          bxse = ldlcse, #estándar error para esos beta
                          by = chdlodds, #beta para la enfermedad coronaria
                          byse = chdloddsse) #estándar error de la enfermedad coronaria


IVWObject <- mr_ivw(MRInputObject,
                    model = "default",
                    robust = FALSE,
                    penalized = FALSE,
                    correl = FALSE,
                    weights = "simple",
                    psi = 0,
                    distribution = "normal",
                    alpha = 0.05)

IVWObject <- mr_ivw(mr_input(bx = ldlc, bxse = ldlcse,
                             by = chdlodds, byse = chdloddsse))

IVWObject
#el p-valor es 0, por lo que es significativo
#hay que mirar el tamaño de los efectos (Estimate), que en este caso es 2,834

#el paquete también admite SNPs correlacionados
# “Using published
# data in Mendelian randomization: a blueprint for efficient identification of causal risk factors”, doi:
#   10.1007/s10654-015-0011-z.
MRInputObject.cor <- mr_input(bx = calcium,
                              bxse = calciumse,
                              by = fastgluc,
                              byse = fastglucse,
                              corr = calc.rho)

IVWObject.correl <- mr_ivw(MRInputObject.cor,
                           model = "default",
                           correl = TRUE,
                           distribution = "normal",
                           alpha = 0.05)
IVWObject.correl <- mr_ivw(mr_input(bx = calcium, bxse = calciumse,
                                    by = fastgluc, byse = fastglucse, corr = calc.rho))
IVWObject.correl
#también es causal (significativo) con un estimador bastante potente