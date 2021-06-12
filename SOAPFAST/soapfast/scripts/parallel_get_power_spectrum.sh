#!/bin/bash

nrun=-1
ncut=-1
initial=-1
nsubset=-1
submode=''
fname=''
oname=''

# Search through all command-line arguments for ones that we will deal with here
inarg=""
for arg in $(seq 1 $#);do
 if [ "${!arg}" == "-nc" ];then arg1=$((arg+1));ncut=${!arg1}
 elif [ "${!arg}" == "-i" ];then arg1=$((arg+1));initial=${!arg1}
 elif [ "${!arg}" == "-ns" ];then arg1=$((arg+1));nsubset=${!arg1}
 elif [ "${!arg}" == "-sm" ];then arg1=$((arg+1));submode=${!arg1}
 elif [ "${!arg}" == "-nrun" ];then arg1=$((arg+1));nrun=${!arg1}
 elif [ "${!arg}" == "-f" ];then arg1=$((arg+1));fname=${!arg1}
 elif [ "${!arg}" == "-o" ];then arg1=$((arg+1));oname=${!arg1}
 else
  argm=$((arg-1))
  if [ "${!argm}" != "-nc" ] && [ "${!argm}" != "-i" ] && [ "${!argm}" != "-ns" ] && [ "${!argm}" != "-sm" ] && [ "${!arg}" != "-sl" ] && [ "${!argm}" != "-sl" ] && [ "${!argm}" != "-f" ] && [ "${!argm}" != "-nrun" ] && [ "${!argm}" != "-o" ];then inarg=$(echo $inarg" "${!arg});fi
 fi
done

if [ "$fname" == "" ];then echo "ERROR: filename must be specified";exit 1;fi
if [ "$oname" == "" ];then echo "ERROR: output file must be specified";exit 1;fi

# Anything that we have flagged up is to be dealt with either before running the power spectrum routine or afterwards. Beforehand, if we want to take some subset of the XYZ file we do so here.

if [ $nsubset -gt -1 ];then
 if [ "$submode" == "" ] || [ "$submode" == "seq" ];then
  # Get a sequential part of the set
  export nset=$nsubset;export filename=$fname;python -c 'import os;from ase.io import read,write;nset=int(os.environ.get("nset"));fname=os.environ.get("filename");xyz = read(fname,":");write("coords_output.xyz",xyz[:nset])';
 elif [ "$submode" == "random" ];then
  # Get a random part of the set
  export nset=$nsubset;export filename=$fname;python -c 'import os;from ase.io import read,write;import random;nset=int(os.environ.get("nset"));fname=os.environ.get("filename");xyz = read(fname,":");random.shuffle(xyz);write("coords_output.xyz",xyz[:nset])'
 else
  echo "ERROR: submode "$submode" is invalid!"
  exit 1
 fi
else
 cp $fname coords_output.xyz
fi

inarg=$inarg" -f coords_output.xyz"

# Now we split up the power spectrum calculation into multiple calculations.
if [ $nrun -eq -1 ];then nrun=$(nproc --all);fi
nframes=$(cat coords_output.xyz | awk '{if (NF==1){print}}' | wc -l)
numinslice=$(echo $nframes $nrun | awk '{print int($1/$2) + 1}')

# Do the calculations
for i in $(seq 1 $nrun);do
 begin=$(echo $i $numinslice | awk '{print ($1-1)*$2}')
 end=$(echo $i $numinslice $nframes | awk '{endnum = $1*$2;if (endnum>$3){print $3}else{print endnum}}')
 sagpr_get_PS ${inarg} -sl ${begin} ${end} -o PS_output_${i} &
done

wait

# Put output together into one power spectrum
export outfile=$oname
export numrun=$nrun
stack_power_spectra.py

# Remove intermediate output files
rm PS_output_*.npy

# Next, do sparsification if we have asked for it.
if [ $ncut -ne -1 ];then
 feature_fps.py -p ${outfile} -n ${ncut} -i ${initial}
fi

# Rename the sparsified power spectrum
if [ -f ${outfile}_sparse.npy ];then
	mv ${outfile}_sparse.npy ${outfile}.npy
fi
