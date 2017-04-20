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


int load(const char * const, double * const, double * const);
void fillWeights(double * const);
void shuffle(const int, double * const, double * const);
