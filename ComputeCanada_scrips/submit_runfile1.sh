#!/bin/bash
#SBATCH --mem-per-cpu=2048M
#SBATCH --time=1:00:00

module load python/3.10
module load scipy-stack
module load java/14.0.2
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index tensorflow

python runfile1.py
