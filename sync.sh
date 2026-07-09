#!/bin/bash
set -euo pipefail

# Usage:
#   ./sync.sh model1 exp1 29999
#   ./sync.sh model1 '*' 29999
#   ./sync.sh '*' '*' 29999

MODEL=${1:-*}
EXP=${2:-*}
STEP=${3:?Missing checkpoint step}

REMOTE="a800:/data/miaozhuochen/rbot/checkpoints"

echo "Downloading:"
echo "  model = $MODEL"
echo "  exp   = $EXP"
echo "  step  = $STEP"

rsync -avm \
    --include='*/' \
    --include="${MODEL}/${EXP}/config.yaml" \
    --include="${MODEL}/${EXP}/${STEP}/params/***" \
    --include="${MODEL}/${EXP}/${STEP}/assets/***" \
    --exclude='*' \
    "$REMOTE" .
