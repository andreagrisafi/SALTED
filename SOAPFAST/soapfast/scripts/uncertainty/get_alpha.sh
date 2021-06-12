#!/bin/bash

paste RESIDUAL.*.txt | awk 'BEGIN{nn=mm=0}{n=m=l=0;for (i=1;i<=NF;i+=2){n++;m+=$i};ypred=(m/n)i;l=0;for (i=1;i<=NF;i+=2){l+=($i-ypred)**2};sig2=l/(n-1);print ypred,sig2;nn++;mm+=(ypred-$2)**2 / sig2}END{printf "%.16f\n",(mm/nn)}' | tee alpha.txt
