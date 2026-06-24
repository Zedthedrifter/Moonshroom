#!/bin/bash

CSV=$1

awk -F, '{for(i=1;i<=NF;i++) a[NR,i]=$i} END{for(i=1;i<=NF;i++) for(j=1;j<=NR;j++) print a[j,i]}' $CSV
