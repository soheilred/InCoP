#!/bin/sh

ID=$(echo $1 | awk '{print $4}')
echo "PID: "$ID
DIR="../output/errors/tbst_${ID}.log"
echo "In Directory:"$DIR
grep "In dir:" $DIR | awk '{print $6}'
grep "In dir:" $DIR | awk '{print $6}' | awk -F "/" '{print $11}'
