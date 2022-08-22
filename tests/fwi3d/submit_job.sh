#!/bin/bash

#$ -S /bin/bash
#$ -j y
#$ -cwd
#$ -P img_prob
#$ -N vfwi3d
# -V
# longest time
#$ -l d_rt=10:25:00
#$ -l lustre03=1
#$ -pe cascadelake 96
# -pe cascadelake-misc 288
# -pe haswell 216
#$ -o tdfwi.log
# job array
# -t 1-1:1

module load sge
module load intel
module load pyhpc

export OMP_NUM_THREADS=24

PYTHONPATH=/lustre03/other/EIP/variational/FWI/overthrust/vfwi/ python vfwi3d.py -r 0
