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

#include "data.h"
#include "mem.h"

/** Declarations **************************************************************/

static int train(
    const double **,
    const double *,
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
        (const double **)x,
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
    const double **x,
    const double *y,
    const int count,
    const int layerCount,
    const int layerNodeCount,
    const int epochs,
    const double gamma0,
    double ***w
) {
    int e, i;
    double ***wSwap1, ***wSwap2, **z;

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

        /* Adjust weights for each example. */
        for(i = 0; i < count; i++) {

        }
    }

    /* Copy trained weights into w. */

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
