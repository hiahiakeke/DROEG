
data("exampleDepMat")
data("curated_BAGEL_essential")

#COADREAD
coadread_adam <- read.table("/Users/wupeike/Desktop/pan_final_results/Adam_pan/coadread/coadread_adam.csv",sep = ",",header = T,row.names = 1)
coadread_adam <- t(coadread_adam)
coadread_pprofile <- ADAM.panessprofile(depMat = coadread_adam)
coadread_nullmodel<-ADAM.generateNullModel(depMat=coadread_adam,ntrials = 1000)
coadread_EO <- ADAM.empiricalOdds(observedCumSum = coadread_pprofile$CUMsums,
                       simulatedCumSum = coadread_nullmodel$nullCumSUM)
coadread_TPR <- ADAM.truePositiveRate(coadread_adam,curated_BAGEL_essential)
coadread_crossoverpoint <-ADAM.tradeoffEO_TPR(coadread_EO,coadread_TPR$TPR,
                                    test_set_name = 'curated BAGEL essential')
#coadread_crossoverpoint:34

##STAD
stad_adam <- read.table("/Users/wupeike/Desktop/pan_final_results/Adam_pan/stad/stad_adam.csv",sep = ",",header = T,row.names = 1)
stad_adam <- t(stad_adam)
stad_pprofile <- ADAM.panessprofile(depMat = stad_adam)
stad_nullmodel<-ADAM.generateNullModel(depMat=stad_adam,ntrials = 1000)
stad_EO <- ADAM.empiricalOdds(observedCumSum = stad_pprofile$CUMsums,
                              simulatedCumSum = stad_nullmodel$nullCumSUM)
stad_TPR <- ADAM.truePositiveRate(stad_adam,curated_BAGEL_essential)
stad_crossoverpoint <-ADAM.tradeoffEO_TPR(stad_EO,stad_TPR$TPR,
                                          test_set_name = 'curated BAGEL essential')
#stad_crossoverpoint:25

##PAAD
pancreas_adam <- read.table("/Users/wupeike/Desktop/pan_final_results/Adam_pan/Pancreas/pancreas_adam.csv",sep = ",",header = T,row.names = 1)
pancreas_adam <- t(pancreas_adam)
pancreas_pprofile <- ADAM.panessprofile(depMat = pancreas_adam)
pancreas_nullmodel<-ADAM.generateNullModel(depMat=pancreas_adam,ntrials = 1000)
pancreas_EO <- ADAM.empiricalOdds(observedCumSum = pancreas_pprofile$CUMsums,
                              simulatedCumSum = pancreas_nullmodel$nullCumSUM)
pancreas_TPR <- ADAM.truePositiveRate(pancreas_adam,curated_BAGEL_essential)
pancreas_crossoverpoint <-ADAM.tradeoffEO_TPR(pancreas_EO,pancreas_TPR$TPR,
                                          test_set_name = 'curated BAGEL essential')
#pancreas_crossoverpoint:32

##ESCA
esca_adam <- read.table("/Users/wupeike/Desktop/pan_final_results/Adam_pan/ESCA/esca_adam.csv",sep = ",",header = T,row.names = 1)
esca_adam <- t(esca_adam)
esca_pprofile <- ADAM.panessprofile(depMat = esca_adam)
esca_nullmodel<-ADAM.generateNullModel(depMat=esca_adam,ntrials = 1000)
esca_EO <- ADAM.empiricalOdds(observedCumSum = esca_pprofile$CUMsums,
                              simulatedCumSum = esca_nullmodel$nullCumSUM)
esca_TPR <- ADAM.truePositiveRate(esca_adam,curated_BAGEL_essential)
esca_crossoverpoint <-ADAM.tradeoffEO_TPR(esca_EO,esca_TPR$TPR,
                                          test_set_name = 'curated BAGEL essential')
#esca_crossoverpoint:27

##LIHC
lihc_adam <- read.table("/Users/wupeike/Desktop/pan_final_results/Adam_pan/LIHC/lihc_adam.csv",sep = ",",header = T,row.names = 1)
lihc_adam <- t(lihc_adam)
lihc_pprofile <- ADAM.panessprofile(depMat = lihc_adam)
lihc_nullmodel<-ADAM.generateNullModel(depMat=lihc_adam,ntrials = 1000)
lihc_EO <- ADAM.empiricalOdds(observedCumSum = lihc_pprofile$CUMsums,
                              simulatedCumSum = lihc_nullmodel$nullCumSUM)
lihc_TPR <- ADAM.truePositiveRate(lihc_adam,curated_BAGEL_essential)
lihc_crossoverpoint <-ADAM.tradeoffEO_TPR(lihc_EO,lihc_TPR$TPR,
                                          test_set_name = 'curated BAGEL essential')
#lihc_crossoverpoint:18
