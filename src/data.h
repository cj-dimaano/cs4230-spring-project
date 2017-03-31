/*******************************************************************************
File: data.h
Created by: CJ Dimaano
Date created: March 30, 2017

Data management stuff.

*******************************************************************************/

#define TRAIN_SET "./data/data.train"
#define TEST_SET "./data/data.test"
#define FEATURE_COUNT 361
#define MAX_EXAMPLES 0x2000


int load(const char * const, double * const * const, double * const);
void fillWeights(const int, const int, double * const * const * const);
void shuffle(const int, double **, double *);
