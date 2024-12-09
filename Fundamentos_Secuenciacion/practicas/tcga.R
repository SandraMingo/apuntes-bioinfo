#install.packages("TCGAretriever")
#install.packages("reshape2")
#install.packages("ggplot2")

library(TCGAretriever)
library(reshape2)
library(ggplot2)

# Obtain a list of cancer studies from cBio
# https://www.cbioportal.org/
all_studies <- get_cancer_studies()
head(all_studies)
#Vemos que nos devuelve un dataframe con 469 observaciones y 13 columnas
#Como es demasiada información, vamos a ir filtrando

# Find published TCGA datasets
keep <- grepl("tcga_pub$", all_studies[,'studyId'])
tcga_studies <- all_studies[keep, ]
#Nos devuelve 19 estudios, que son los únicos publicados

# Show results
utils::head(tcga_studies[, c(11, 1, 4)])
View(tcga_studies)

# Define the cancer study id: brca_tcga_pub
#Esto en tcga_studies se encuentra en studyId
my_csid <- "brca_tcga_pub"

# Obtain genetic profiles
#csid viene de cancer study id, y nos devuelve un dataframe
#a la información molecular les llaman perfiles genéticos, lo que está un poco anticuado
brca_pro <- get_genetic_profiles(csid = my_csid)
utils::head(brca_pro[, c(7, 1)])
View(brca_pro)
#Tenemos 12 observaciones con distintos tipos de alteraciones moleculares

# Obtain cases 
#Obtenemos una lista con los 825 casos
brca_cas <- get_case_lists(csid = my_csid)
utils::head(brca_cas[, c(4, 1)])
View(brca_cas)

#Para todas las muestras, vamos a centrar nuestro análisis en 4 genes
# Define a set of genes of interest
#q viene de query
#sacamos los casos con información completa o de mensajero o de mutaciones
q_csid <- 'brca_tcga_pub'
q_genes <- c("TP53", "HMGA1", "E2F1", "EZH2")
q_cases <- "brca_tcga_pub_complete"
rna_prf <- "brca_tcga_pub_mrna"
mut_prf <- "brca_tcga_pub_mutations"

#La información molecular sola no nos sirve de mucho. En el estudio ponía que 
#era case-control, y también es importante la supervivencia. Toda esta información
#se extrae en los datos clínicos
# Download Clinical Data
brca_cli <- get_clinical_data(csid = q_csid, case_list_id = q_cases)
View(brca_cli)
#Puede haber diferentes muestras para un mismo paciente, por lo que hay un 
#identificador para muestra y para paciente.
brca_cli$OS_MONTHS
brca_cli$OS_STATUS
#OS viene de overall survival

#Nos vamos a centrar ahora los cuatro genes para los que queremos ver la información
#sacamos la expresión de los genes
# Download RNA
brca_RNA <- get_molecular_data(case_list_id = q_cases, 
                               gprofile_id = rna_prf, 
                               glist = q_genes)
View(brca_RNA)

#Cambiamos las filas del dataframe por los genes para reestructurar el dataframe
# Set SYMBOLs as rownames
# Note that you may prefer to use the tibble package for this
rownames(brca_RNA) <- brca_RNA$hugoGeneSymbol
brca_RNA <- brca_RNA[, -c(1, 2, 3)]
View(brca_RNA)

# Round numeric vals to 3 decimals
for (i in 1:ncol(brca_RNA)) {
  brca_RNA[, i] <- round(brca_RNA[, i], digits = 3)
}

#Nos descargamos las mutaciones para estos mismos participantes. 
#Se sabe que la expresión de estos genes podría venir condicionada por tener o no mutaciones en TP53
# Download mutations
brca_MUT <- get_mutation_data(case_list_id = q_cases, 
                              gprofile_id = mut_prf, 
                              glist = q_genes)

# Identify Samples carrying a TP53 missense mutation
tp53_mis_keep <- brca_MUT$hugoGeneSymbol == 'TP53' &
  brca_MUT$mutationType == 'Missense_Mutation' &
  !is.na(brca_MUT$sampleId)
sum(tp53_mis_keep)
#esto es para ver si hay muestras de un mismo participante
tp53_mut_samples <- unique(brca_MUT$sampleId[tp53_mis_keep])

# Show results
keep_cols <- c('sampleId', 'hugoGeneSymbol', 'mutationType',  'proteinChange')
utils:::head(brca_MUT[, keep_cols])
View(brca_MUT[ , keep_cols])
#tenemos 180 observaciones porque indican las mutaciones en otros genes que 
#tienen los pacientes que tienen mutaciones en TP53.

############################################################
# Visualize the correlation between EZH2 and E2F1
############################################################

df <- data.frame(sample_id = colnames(brca_RNA), 
                 EZH2 = as.numeric(brca_RNA['EZH2', ]), 
                 E2F1 = as.numeric(brca_RNA['E2F1', ]), 
                 stringsAsFactors = FALSE)

ggplot(df, aes(x = EZH2, y = E2F1)) +
  geom_point(color = 'gray60', size = 0.75) +
  theme_bw() +
  geom_smooth(method = 'lm', color = 'red2', 
              size=0.3, fill = 'gray85') +
  ggtitle('E2F1-EZH2 correlation in BRCA') + 
  theme(plot.title = element_text(hjust = 0.5))

#La expresión está muy correlacionada.
summary(lm(df$EZH2~df$E2F1))
#Numéricamente, la pendiente es de 0,45 y muy significativa
#R^2 es de 0,29.
#Cuidado: en cuanto tengamos muchas observaciones, a nada que la pendiente sea 
#mayor que 0, el resultado es significativo. Por ello, hay que ver el p-valor, 
#la pendiente y el R^2

# Bin samples according to EZH2 expression
EZH2_bins <- make_groups(num_vector = df$EZH2, groups = 5) 
utils::head(EZH2_bins, 12)

# attach bin to df
df$EZH2_bin <- EZH2_bins$rank

# build Boxplot
ggplot(df, aes(x = as.factor(EZH2_bin), y = E2F1)) +
  geom_boxplot(outlier.shape = NA, fill = '#fed976') +
  geom_jitter(width = 0.1, size = 1) +
  theme_bw() +
  xlab('EZH2 Bins') +
  ggtitle('E2F1 Expression vs. Binned EZH2 Expression') +
  theme(plot.title = element_text(face = 'bold', hjust = 0.5))

#A medida que aumenta EZH2, aumenta E2F1

#añadimos la expresión de HMGA1 y TP53
# Coerce to data.frame with numeric features 
mol_df <- data.frame(sample_id = colnames(brca_RNA), 
                     HMGA1 = as.numeric(brca_RNA['HMGA1', ]),
                     TP53 = as.numeric(brca_RNA['TP53', ]),
                     stringsAsFactors = FALSE)

#creamos una columna que sea wild type para los que no tengan mutación en tp53 y mutante los que sí
mol_df$TP53.status = factor(ifelse(mol_df$sample_id %in% tp53_mut_samples, 
                                   '01.wild_type', '02.mutated'))
###########################################################################
# Visualize the correlation between EZH2 and E2F1
###########################################################################

ggplot(mol_df, aes(x = TP53, y = HMGA1)) +
  geom_point(color = 'gray60', size = 0.75) +
  facet_grid(cols = vars(TP53.status)) +
  theme_bw() +
  geom_smooth(mapping = aes(color = TP53.status), 
              method = 'lm', size=0.3, fill = 'gray85') +
  ggtitle('HMGA1-TP53 correlation in BRCA') + 
  theme(plot.title = element_text(hjust = 0.5))
#La asociación entre los dos genes está invertida en cuanto al estado mutado de 
#TP53. Por ello, es muy importante genotipar a los individuos y ver si tienen 
#mutación en TP53, ya que el comportamiento molecular es muy diferente.

