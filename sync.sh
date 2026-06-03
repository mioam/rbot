#!/bin/bash
rsync -avm \
  --include='*/' \
  --include='*/*/*/params/***' \
  --include='*/*/*/assets/***' \
  --include='*/*/config.yaml' \
  --exclude='*' \
  ${SERVER}/checkpoints/ ./checkpoints
