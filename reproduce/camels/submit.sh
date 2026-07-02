#!/bin/bash
# Submit the three-stage CAMELS pipeline with automatic afterok dependencies.
# Run from the degen_detector repo root:
#   bash reproduce/camels/submit.sh

set -e

TRAIN_JOB=$(sbatch --parsable reproduce/camels/train.job)
echo "Submitted train:  job ${TRAIN_JOB}"

SAMPLE_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} reproduce/camels/sample.job)
echo "Submitted sample: job ${SAMPLE_JOB} (depends on ${TRAIN_JOB})"

DETECT_JOB=$(sbatch --parsable --dependency=afterok:${SAMPLE_JOB} reproduce/camels/detect.job)
echo "Submitted detect: job ${DETECT_JOB} (depends on ${SAMPLE_JOB})"

echo ""
echo "Pipeline queued. Monitor with: squeue -u \$USER"
