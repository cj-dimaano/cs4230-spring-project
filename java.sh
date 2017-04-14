#!/bin/bash
find java/ -name *.java | xargs javac -d bin
java -Xmx4096M -d64 -cp bin ml/NNExperiment1
