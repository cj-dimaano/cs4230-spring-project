/*******************************************************************************
File: seq.c
Created by: CJ Dimaano
Date created: March 29, 2017

Sequential code of a fully connected artificial neural network.

Compile with:
```
$ gcc -Wall -lm -o seq seq.c
```

*******************************************************************************/

/** Includes ******************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "data.h"
#include "mem.h"

/** Declarations **************************************************************/

static int train(
    double **,
    double *,
    const int,
    const int,
    const int,
    const int,
    const double,
    double ***
);
static void evaluate(
    const int,
    const int,
    const double ***,
    const double **,
    const double *
);
static double propagate(
    const int,
    const int,
    const double ***,
    const double **,
    const int,
    const int,
    const int
);

/** Main **********************************************************************/

int main(int argc, char **argv) {
    double ***w, **x, *y;
    int ret, layerCount = 0, layerNodeCount = 0, epochs = 1, gamma0 = 1;

    // TODO
    // command line arguments

    /* Allocate memory for examples. */
    ret = init(layerCount, layerNodeCount, &w, &x, &y);
    if(ret < 0)
        return ret;

    /* Load training data. */
    ret = load(TRAIN_SET, x, y);
    if(ret < 0) {
        cleanup(layerCount, layerNodeCount, &w, &x, &y);
        return ret;
    }

    /* Train classifier. */
    ret = train(
        x,
        y,
        ret,
        layerCount,
        layerNodeCount,
        epochs,
        gamma0,
        w
    );
    if(ret < 0) {
        cleanup(layerCount, layerNodeCount, &w, &x, &y);
        return ret;
    }

    /* Load test data. */
    ret = load(TEST_SET, x, y);
    if(ret < 0) {
        cleanup(layerCount, layerNodeCount, &w, &x, &y);
        return ret;
    }

    /* Evaluate classifier accuracy. */
    evaluate(
        layerCount,
        layerNodeCount,
        (const double ***)w,
        (const double **)x,
        (const double *)y
    );

    /* Cleanup memory from examples. */
    cleanup(layerCount, layerNodeCount, &w, &x, &y);

    return 0;
}

/** Static functions **********************************************************/

/**
 * train
 *
 * @summary
 *   Trains the weights of the artificial neural network classifier.
 */
static int train(
    double **x,
    double *y,
    const int count,
    const int layerCount,
    const int layerNodeCount,
    const int epochs,
    const double gamma0,
    double ***w
) {
    int e, i, j, k, l;
    double ***wSwap1, ***wSwap2, ***tmpw, **z, dot, yp, dLy;

    /* Allocate swap weights. */
    i = mallocWeights(layerCount, layerNodeCount, &wSwap2);
    if(i < 0)
        return i;

    /* Allocate z. */
    i = mallocz(layerCount, layerNodeCount, &z);
    if(i < 0)
        return i;

    /* Initialize weights. */
    fillWeights(layerCount, layerNodeCount, (double * const * const * const)w);
    wSwap1 = w;

    /* Train over epochs. */
    for(e = 0; e < epochs; e++) {
        /* Shuffle examples. */
        shuffle(count, x, y);

/******************************************************************************/

        /* Adjust weights for each example. */
        for(i = 0; i < count; i++) {

            /* Compute yp and remember hidden layer features. */
            memcpy(z[0], x[i], FEATURE_COUNT * sizeof(double));
            for(l = 1; l < layerCount + 1; l++) {
                z[l][0] = 1;
                for(j = 1; j < layerNodeCount + 1; j++) {
                    dot = 0;
                    for(k = 0;
                        k < (l == 1 ? FEATURE_COUNT : layerNodeCount + 1);
                        k++)
                        dot += wSwap1[l - 1][j - 1][k] * z[l - 1][k];
                    z[l][j] = 1.0 / (1.0 + exp(-dot));
                }
            }
            yp = 0;
            for(j = 0;
                j < (layerCount > 0 ? layerNodeCount + 1 : FEATURE_COUNT);
                j++)
                yp += wSwap1[layerCount][0][j] * z[layerCount][j];

            /* Save derivitive of square loss. */
            dLy = yp - y[i];

            /* Back propagation. */
            l = layerCount;
            for(j = 0;
                j < (layerCount > 0 ? layerNodeCount + 1 : FEATURE_COUNT);
                j++) {
                wSwap2[l][0][j] = wSwap1[l][0][j] - gamma0 * dLy * z[l][j];
                z[l][j] *= (1.0 - z[l][j]);
            }

            for(l = l - 1; l >= 0; l--) {
                for(j = 0; j < layerNodeCount; j++) {
                    for(k = 0;
                        k < (l == 0 ? FEATURE_COUNT : layerNodeCount + 1);
                        k++)
                        wSwap2[l][j][k] = wSwap1[l][j][k] - gamma0 * dLy
                            * propagate(
                                layerCount,
                                layerNodeCount,
                                (const double ***)wSwap1,
                                (const double **)z,
                                l,
                                j,
                                k
                            );
                    for(k = 0;
                        k < (l == 0 ? FEATURE_COUNT : layerNodeCount + 1);
                        k++)
                        z[l][k] *= (1.0 - z[l][j]);
                }
            }
            tmpw = wSwap1;
            wSwap1 = wSwap2;
            wSwap2 = tmpw;

        }

/******************************************************************************/

    }

    /* Copy trained weights into w. */
    if(w != wSwap1)
        copyWeights(layerCount, layerNodeCount, wSwap1, w);

    /* Cleanup memory. */
    if(w == wSwap1)
        freeWeights(layerCount, layerNodeCount, &wSwap2);
    else
        freeWeights(layerCount, layerNodeCount, &wSwap1);
    freez(layerCount, &z);

    return 0;
}

/**
 * evaluate
 */
static void evaluate(
    const int layerCount,
    const int layerNodeCount,
    const double ***w,
    const double **x,
    const double *y
) {
    // TODO:
}

/**
 * propagate
 */
static double propagate(
    const int layerCount,
    const int layerNodeCount,
    const double ***w,
    const double **z,
    const int i,
    const int j,
    const int k
) {
    double result = 0;
    int j1;

    if(i + 2 < layerCount + 1) {
      for(j1 = 0; j1 < layerNodeCount; j1++)
        result += z[i][k] * w[i + 1][j1][j + 1]
            * propagate(layerCount, layerNodeCount, w, z, i + 1, j1, j + 1);
    }
    else
      result = z[i][k] * w[i + 1][0][j + 1] * z[i + 1][j + 1];

    return result;
}
