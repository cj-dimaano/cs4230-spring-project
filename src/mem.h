/*******************************************************************************
File: mem.h
Created by: CJ Dimaano
Date created: March 30, 2017

Memory management stuff.

*******************************************************************************/

int init(const int, const int, double ****, double ***, double **);
void cleanup(const int, const int, double ****, double ***, double **);
int mallocWeights(const int, const int, double ****);
void freeWeights(const int, const int, double ****);
int mallocz(const int, const int, double ***);
void freez(const int, double ***);
