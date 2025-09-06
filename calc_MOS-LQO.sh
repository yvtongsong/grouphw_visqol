#!/bin/bash

VISQOL_BIN=./bazel-bin/visqol
TEST_SET_DIR="/root/TCD_VOIP/TestSet"
OUT_DIR="/root/visqol_csv"

mkdir -p $OUT_DIR

# 遍历每个劣化条件文件夹
for condition in chop clip compspkr echo noise; do
    REF_DIR="$TEST_SET_DIR/$condition/ref"
    DEG_DIR="$TEST_SET_DIR/$condition"

    # 创建每个条件对应输出子文件夹
    mkdir -p "$OUT_DIR/$condition"

    for deg_file in "$DEG_DIR"/C_*.wav; do
        base_name=$(basename "$deg_file")
        ref_file="$REF_DIR/R_${base_name:2}"  # C_02_CHOP_FG.wav -> R_02_CHOP_FG.wav
        out_csv="$OUT_DIR/$condition/${base_name%.wav}.csv"

        $VISQOL_BIN --reference_file "$ref_file" --degraded_file "$deg_file" --results_csv "$out_csv"
    done
done

