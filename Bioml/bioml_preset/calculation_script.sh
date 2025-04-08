#!/bin/bash
#SBATCH --job-name=bioml_preset
#SBATCH --qos=gp_bscls
#SBATCH --time=48:00:00
#SBATCH --ntasks 1
#SBATCH --account=bsc72
#SBATCH --cpus-per-task 1
#SBATCH --array=1-1
#SBATCH --output=bioml_preset_%a_%A.out
#SBATCH --error=bioml_preset_%a_%A.err

module purge
module load anaconda
module load perl/5.38.2

source activate /gpfs/projects/bsc72/conda_envs/bioml

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
if [[ $SLURM_ARRAY_TASK_ID = 1 ]]; then
python -m BioML.features.generate_pssm -i /gpfs/projects/bsc72/Ruite/bioml/whole_sequence.fasta -Po /gpfs/projects/bsc72/Ruite/Oxipro/machine_learning/POSSUM_Toolkit/ -dbinp /gpfs/projects/bsc72/Ruite/bioml/no_short.fasta --iterations 3 --evalue 0.01 --sensitivity 6.5 --generate_searchdb False -p pssm 
fi

conda deactivate 

