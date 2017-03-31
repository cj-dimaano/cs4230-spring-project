/*******************************************************************************
File: mem.c
Created by: CJ Dimaano
Date created: March 30, 2017

Memory management stuff.

*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "data.h"
#include "mem.h"

/**
 * init
 */
int init(
    const int layerCount,
    const int layerNodeCount,
    double ****w,
    double ***x,
    double **y
) {
    int i;
    /* Allocate y. */
    (*y) = (double *)calloc(MAX_EXAMPLES, sizeof(double));
    if((*y) == NULL) {
        perror("error `init`: not enough memory");
        return -1;
    }
    /* Allocate x. */
    (*x) = (double **)calloc(MAX_EXAMPLES, sizeof(double *));
    if((*x) == NULL) {
        perror("error `init`: not enough memory");
        free((*y));
        return -2;
    }
    for(i = 0; i < MAX_EXAMPLES; i++) {
        (*x)[i] = (double *)calloc(FEATURE_COUNT, sizeof(double));
        if((*x)[i] == NULL)
            break;
    }
    if(i < MAX_EXAMPLES) {
        perror("error `init`: not enough memory");
        while(i > 0) {
            i--;
            free((*x)[i]);
        }
        free((*x));
        free((*y));
        return -3;
    }
    /* Allocate w. */
    i = mallocWeights(layerCount, layerNodeCount, w);
    if(i < 0) {
        for(i = 0; i < MAX_EXAMPLES; i++)
            free((*x)[i]);
        free((*x));
        free((*y));
        return -4;
    }
    return 0;
}

/**
 * cleanup
 */
void cleanup(
    const int layerCount,
    const int layerNodeCount,
    double ****w,
    double ***x,
    double **y
) {
    int i;
    free((*y));
    for(i = 0; i < MAX_EXAMPLES; i++)
        free((*x)[i]);
    free((*x));
    (*y) = NULL;
    (*x) = NULL;
    freeWeights(layerCount, layerNodeCount, w);
}

/**
 * mallocWeights
 */
int mallocWeights(
    const int layerCount,
    const int layerNodeCount,
    double ****w
) {
    int i, j;
    (*w) = (double ***)calloc(layerCount + 1, sizeof(double **));
    if((*w) == NULL) {
        perror("error `mallocWeights`: not enough memory");
        return -1;
    }
    if(layerCount > 0) {
        (*w)[0] = (double **)calloc(layerNodeCount, sizeof(double *));
        if((*w)[0] == NULL) {
            perror("error `mallocWeights`: not enough memory");
            free((*w));
            return -2;
        }
        for(i = 0; i < layerNodeCount; i++) {
            (*w)[0][i] = (double *)calloc(FEATURE_COUNT, sizeof(double));
            if((*w)[0][i] == NULL)
                break;
        }
        if(i < layerNodeCount) {
            perror("error `mallocWeights`: not enough memory");
            while(i > 0) {
                i--;
                free((*w)[0][i]);
            }
            free((*w)[0]);
            free((*w));
            return -3;
        }
        for(i = 1; i < layerCount; i++) {
            (*w)[i] = (double **)calloc(layerNodeCount + 1, sizeof(double *));
            if((*w)[i] == NULL)
                break;
            for(j = 0; j < layerNodeCount; j++) {
                (*w)[i][j] = (double *)calloc(
                    layerNodeCount + 1,
                    sizeof(double)
                );
                if((*w)[i][j] == NULL)
                    break;
            }
            if(j < layerNodeCount) {
                while(j > 0) {
                    j--;
                    free((*w)[i][j]);
                }
                break;
            }
        }
        if(i < layerCount) {
            perror("error `mallocWeights`: not enough memory");
            while(i > 0) {
                i--;
                for(j = 0; j < layerNodeCount; j++)
                    free((*w)[i][j]);
                free((*w)[i]);
            }
            free((*w));
            return -4;
        }
        (*w)[i] = (double **)malloc(sizeof(double *));
        if((*w)[i] == NULL) {
            perror("error `mallocWeights`: not enough memory");
            for(i = 0; i < layerCount; i++) {
                for(j = 0; j < layerNodeCount; j++)
                    free((*w)[i][j]);
                free((*w)[i]);
            }
            free((*w));
            return -5;
        }
        (*w)[i][0] = (double *)calloc(layerNodeCount + 1, sizeof(double));
        if((*w)[i][0] == NULL) {
            perror("error `mallocWeights`: not enough memory");
            for(i = 0; i < layerCount; i++) {
                for(j = 0; j < layerNodeCount; j++)
                    free((*w)[i][j]);
                free((*w)[i]);
            }
            free((*w)[i]);
            free((*w));
            return -6;
        }
    }
    else {
        (*w)[0] = (double **)malloc(sizeof(double *));
        if((*w)[0] == NULL) {
            perror("error `mallocWeights`: not enough memory");
            free((*w));
            return -7;
        }
        (*w)[0][0] = (double *)calloc(FEATURE_COUNT, sizeof(double));
        if((*w)[0][0] == NULL) {
            perror("error `mallocWeights`: not enough memory");
            free((*w)[0]);
            free((*w));
            return -8;
        }
    }
    return 0;
}

/**
 * freeWeights
 */
void freeWeights(
    const int layerCount,
    const int layerNodeCount,
    double ****w
) {
    int i, j;
    for(i = 0; i < layerCount + 1; i++) {
        for(j = 0; j < layerNodeCount; j++)
            free((*w)[i][j]);
        free((*w)[i]);
    }
    free((*w));
    (*w) = NULL;
}

/**
 * mallocz
 */
int mallocz(
    const int layerCount,
    const int layerNodeCount,
    double ***z
) {
    int i;
    (*z) = (double **)calloc(layerCount, sizeof(double *));
    if((*z) == NULL) {
        perror("error `mallocz`: not enough memory");
        return -1;
    }
    for(i = 0; i < layerCount; i++) {
        (*z)[i] = (double *)calloc(layerNodeCount + 1, sizeof(double));
        if((*z)[i] == NULL)
            break;
    }
    if(i < layerCount) {
        perror("error `mallocz`: not enough memory");
        while(i > 0) {
            i--;
            free((*z)[i]);
        }
        free((*z));
        return -2;
    }
    return 0;
}

/**
 * freez
 */
void freez(
    const int layerCount,
    double ***z
) {
    int i;
    for(i = 0; i < layerCount; i++)
        free((*z)[i]);
    free((*z));
    (*z) = NULL;
}
