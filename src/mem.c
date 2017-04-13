/*******************************************************************************
File: mem.c
Created by: CJ Dimaano
Date created: March 30, 2017

Memory management stuff.

*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "mem.h"

/**
 * init
 */
int init(
    double **x,
    double **y,
    double **w
) {
    /*** Allocate space for x and y. ***/
    (*x) = (double *)malloc(MAX_EXAMPLES * FEATURE_COUNT * sizeof(double));
    (*y) = (double *)malloc(MAX_EXAMPLES * sizeof(double));
    (*w) = (double *)calloc(FEATURE_COUNT, sizeof(double));
    if((*x) == NULL || (*y) == NULL || (*w) == NULL) {
        perror("error `init`: not enough memory");
        cleanup(x, y, w);
        return -1;
    }
    return 0;
}

/**
 * freeptr
 */
void freeptr(void **ptr) {
    if((*ptr) != NULL) {
        free((*ptr));
        (*ptr) = NULL;
    }
}

/**
 * cleanup
 */
void cleanup(double **x, double **y, double **w) {
    freeptr((void **)x);
    freeptr((void **)y);
    freeptr((void **)w);
}
