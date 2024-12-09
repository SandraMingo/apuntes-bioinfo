library(gwasrapidd)

#se busca ver si hay asociación/causalidad entre consumo de alcohol e infarto del miocardio
#buscamos SNPs que estén asociados con el consumo del alcohol, pero no con infarto ni con variables confusoras


######## alcohol consumption
my_studies_alc <- get_studies(efo_trait = 'alcohol drinking') 
#este término se ha buscado previamente en la página web poniendo solo "alcohol" y viendo cuál es el que mejor nos convenga

gwasrapidd::n(my_studies_alc) 
#hay 40 estudios asociados con el consumo de alcohol
my_studies_alc@publications$title

#para cada estudio hay muchas variantes (todas las exploradas, no solo las asociadas), 
#por lo que nos limitamos a los primeros 10 estudios
my_associations_alc <- get_associations(study_id = my_studies_alc@studies$study_id[1:10])
gwasrapidd::n(my_associations_alc)

dplyr::filter(my_associations_alc@associations, pvalue < 1e-6) %>% # Filter by p-value
  tidyr::drop_na(pvalue) %>%
  dplyr::select(association_id,beta_number,standard_error)%>%
  tidyr::drop_na(standard_error) -> association_ids_alc

# my_associations_alc_2 <- my_associations_alc[association_ids_alc]
# my_associations2@risk_alleles[c('variant_id', 'risk_allele', 'risk_frequency')]


######## AMI
my_studies_AMI<-get_studies(efo_trait = 'myocardial infarction')
gwasrapidd::n(my_studies_AMI)
my_studies_AMI@publications$title
my_associations_AMI <- get_associations(study_id = my_studies_AMI@studies$study_id)
gwasrapidd::n(my_associations_AMI)

dplyr::filter(my_associations_AMI@associations, pvalue < 1e-6) %>% # Filter by p-value
  tidyr::drop_na(pvalue) %>%
  dplyr::select(association_id,beta_number,standard_error)%>%
  tidyr::drop_na(standard_error) -> association_ids_AMI

#########Falta la parte del MendelianRandomization
