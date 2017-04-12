/*******************************************************************************
File: mem.h
Created by: CJ Dimaano
Date created: March 30, 2017

Memory management stuff.

*******************************************************************************/

int init(
    const int layerCount,
    const int layerNodeCount,
    double **x,
    double **y,
    double **w
);
int mallocWeights(const int layerCount, const int layerNodeCount, double **w);
int mallocz(const int layerCount, const int layerNodeCount, double **z);

void cleanup(double **x, double **y, double **w);
void freeWeights(double **w);
void freez(double **z);