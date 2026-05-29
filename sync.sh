#!/bin/bash
rsync -avm \
  --include='*/' \
  --include='*/*/*/params/***' \
  --include='*/*/*/assets/***' \
  --exclude='*' \
  ${SERVER}/checkpoints/ ./checkpoints
