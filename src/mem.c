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
    const int layerCount,
    const int layerNodeCount,
    double **x,
    double **y,
    double **w
) {
    /*** Allocate space for x and y. ***/
    (*x) = (double *)malloc(MAX_EXAMPLES * FEATURE_COUNT * sizeof(double));
    (*y) = (double *)malloc(MAX_EXAMPLES * sizeof(double));
    if((*x) == NULL || (*y) == NULL) {
        perror("error `init`: not enough memory");
        cleanup(x, y, NULL);
        return -1;
    }
    if(mallocWeights(layerCount, layerNodeCount, w) < 0) {
        cleanup(x, y, w);
        return -1;
    }
    return 0;
}

/**
 * mallocWeights
 *
 * @returns
 *   The size of the weights memory space if successfully allocated; otherwise,
 *   -1.
 */
int mallocWeights(const int layerCount, const int layerNodeCount, double **w) {
    int size = FEATURE_COUNT;
    if(layerCount > 0) {
        /*** FEATURE_COUNT * layerNodeCount for the first layer.        ***/
        size = FEATURE_COUNT * layerNodeCount;
        /*** Each hidden layer has `nodeCount` output nodes, and we     ***/
        /*** need `nodeCount + 1` weights per node plus                 ***/
        /*** `layerNodeCount + 1` for the output node.                  ***/
        size += ((layerCount - 1) * layerNodeCount + 1) * (layerNodeCount + 1);
    }
    (*w) = (double *)malloc(size * sizeof(double));
    if((*w) == NULL) {
        perror("error `mallocWeights`: not enough memory");
        return -1;
    }
    return size;
}

/**
 * mallocz
 *   `z` represents the intermediate values of the hidden layers for computing
 *   the prediction. If `layerCount` is 0, then `z` is effectively null.
 */
int mallocz(const int layerCount, const int layerNodeCount, double **z) {
    if(layerCount > 0) {
        (*z) = (double *)malloc(
            layerCount * (layerNodeCount + 1) * sizeof(double)
        );
        if((*z) == NULL) {
            perror("error `mallocz`: not enough memory");
            return -1;
        }
    }
    else
        (*z) = NULL;
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

/**
 * freeWeights
 */
void freeWeights(double **w) {
    freeptr((void **)w);
}

/**
 * freez
 */
void freez(double **z) {
    freeptr((void **)z);
}
