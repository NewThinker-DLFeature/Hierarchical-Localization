#!/usr/bin/env bash


# test with cmu_extended/slice2

./display_query_match.py \
    ../outputs/aachen_extended/slice2/pairs-query-netvlad10-backup.txt \
    ../outputs/aachen_extended/slice2/feats-superpoint-n4096-r1024_matches-superglue_pairs-query-netvlad10.h5 \
    ../outputs/aachen_extended/slice2/feats-superpoint-n4096-r1024.h5 \
    ../datasets/cmu_extended/slice2 \
    $@

