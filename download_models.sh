#!/bin/bash
set -e

mkdir -p checkpoints

# CoNSeP checkpoint
echo "Downloading CoNSeP checkpoint..."
gdown "1FtoTDDnuZShZmQujjaFSLVJLD5sAh2_P" -O checkpoints/hovernet_original_consep_notype_tf2pytorch.tar

# PanNuke checkpoint
echo "Downloading PanNuke checkpoint..."
gdown "1SbSArI3KOOWHxRlxnjchO7_MbWzB4lNR" -O checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar

echo "Models downloaded."
