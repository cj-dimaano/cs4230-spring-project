/*******************************************************************************
File: seq.c
Created by: CJ Dimaano
Date created: March 29, 2017

Sequential code of a fully connected artificial neural network.
*******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data.h"
#include "mem.h"

/** Declarations **************************************************************/

static int train(
    double * const x,
    double * const y,
    const int count,
    const int layerCount,
    const int layerNodeCount,
    const int epochs,
    const double gamma0,
    double * const w
);
static void test(
    const double * const x,
    const double * const y,
    const int count,
    const int layerCount,
    const int layerNodeCount,
    const double * const w
);
static double propagate(
    const int layerCount,
    const int layerNodeCount,
    const double * const x_i,
    const double * const w,
    const double * const z,
    const int i,
    const int j,
    const int k
);
static double getPrediction(
    const double * const x_i,
    const int layerCount,
    const int layerNodeCount,
    const double * const w
);
static int parseArgs(const int, char **, int *, int *, int *, double *);
static void printUsage(const char * const);

/** Static data ***************************************************************/

/*** For calculating predictions ***/
static double *v = NULL;
static double *u = NULL;
static int vlen = 0;

/** Main **********************************************************************/

int main(int argc, char **argv) {
    double *w, *x, *y, gamma0 = 0.01;
    int ret, layerCount = 1, layerNodeCount = FEATURE_COUNT / 2, epochs = 100;

    /*** Parse command-line arguments. ***/
    ret = parseArgs(argc, argv, &layerCount, &layerNodeCount, &epochs, &gamma0);
    if(ret < 0) {
        return -1;
    }
    printf("epochs: %d\n", epochs);
    printf("layers: %d\n", layerCount);
    printf("layer nodes: %d\n", layerNodeCount);
    printf("gamma: %f\n", gamma0);

    /*** Initialize v and u. ***/
    vlen = layerNodeCount + 1;
    v = (double *)malloc(vlen * sizeof(double));
    u = (double *)malloc(vlen * sizeof(double));
    if(v == NULL) {
        perror("error `main`: not enough memory");
        return -2;
    }
    if(u == NULL) {
        perror("error `main`: not enough memory");
        free(v);
        return -2;
    }

    /*** Allocate memory for examples. ***/
    if(init(layerCount, layerNodeCount, &x, &y, &w) < 0) {
        free(v);
        free(u);
        return -2;
    }

    /*** Load training data. ***/
    ret = load(TRAIN_SET, x, y);
    if(ret < 0) {
        free(v);
        free(u);
        cleanup(&x, &y, &w);
        return -3;
    }

    /*** Train classifier. ***/
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
        free(v);
        free(u);
        cleanup(&x, &y, &w);
        return -4;
    }

    /*** Load test data. ***/
    ret = load(TEST_SET, x, y);
    if(ret < 0) {
        free(v);
        free(u);
        cleanup(&x, &y, &w);
        return ret;
    }

    /*** Test classifier accuracy. ***/
    test(x, y, ret, layerCount, layerNodeCount, w);

    /*** Cleanup memory from examples. ***/
    free(v);
    free(u);
    cleanup(&x, &y, &w);

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
    double * const x,
    double * const y,
    const int count,
    const int layerCount,
    const int layerNodeCount,
    const int epochs,
    const double gamma0,
    double * const w
) {
    int e, i, j, k, l, wlen;
    double *x_i, *wSwap1, *wSwap2, *wptr1, *wptr2, *z, *zcur, *znxt;
    double dot, yp, dLy;

    /*** Allocate swap weights. ***/
    wlen = mallocWeights(layerCount, layerNodeCount, &wSwap2);
    if(wlen < 0)
        return wlen;

    /*** Allocate z. ***/
    i = mallocz(layerCount, layerNodeCount, &z);
    if(i < 0) {
        freeWeights(&wSwap2);
        return i;
    }

    /*** Initialize weights. ***/
    fillWeights(wlen, w);
    wSwap1 = w;

    if(layerCount > 0) {
        /*** Train over epochs. ***/
        for(e = 0; e < epochs; e++) {

            /*** Shuffle examples. ***/
            shuffle(count, x, y);

/** Sequential 1: Neural Network **********************************************/

            for(i = 0; i < count; i++) {
                x_i = (x + (i * FEATURE_COUNT));
                wptr1 = wSwap1;
                zcur = x_i;
                znxt = z;

                /*** Compute yp and remember hidden layer features. ***/

                /* First layer. */
                znxt[0] = 1;
                for(j = 1; j < layerNodeCount + 1; j++) {
                    dot = 0;
                    for(k = 0; k < FEATURE_COUNT; k++)
                        dot += wptr1[k] * zcur[k];
                    znxt[j] = 1.0 / (1.0 + exp(-dot));
                    wptr1 = (wptr1 + FEATURE_COUNT);
                }
                zcur = znxt;
                znxt = (znxt + layerNodeCount + 1);

                /* Remaining layers. */
                for(j = 1; j < layerCount; j++) {
                    znxt[0] = 1;
                    for(k = 1; k < layerNodeCount + 1; k++) {
                        dot = 0;
                        for(l = 0; l < layerNodeCount + 1; l++)
                            dot += wptr1[l] * zcur[l];
                        znxt[k] = 1.0 / (1.0 + exp(-dot));
                        wptr1 = (wptr1 + layerNodeCount + 1);
                    }
                    zcur = znxt;
                    znxt = (znxt + layerNodeCount + 1);
                }

                yp = 0;
                for(j = 0; j < layerNodeCount + 1; j++)
                    yp += wptr1[j] * zcur[j];
                // zcur = znxt;
                znxt = (zcur - layerNodeCount - 1);
                
                /*** Save derivitive of square loss. ***/
                dLy = yp - y[i];
                
                /*** Update weights using back propagation. ***/

                /* Output layer. */
                l = layerCount;
                wptr2 = (
                    wSwap2 + 
                    layerNodeCount * FEATURE_COUNT + 
                    (layerCount - 1) * layerNodeCount * (layerNodeCount + 1)
                );
                for(j = 0; j < layerNodeCount + 1; j++) {
                    wptr2[j] = wptr1[j] - gamma0 * dLy * zcur[j];
                    zcur[j] *= (1.0 - zcur[j]);
                }

                /* Hidden layers. */
                zcur = znxt;
                znxt = (znxt - layerNodeCount - 1);
                wptr1 = (wptr1 - layerNodeCount * (layerNodeCount + 1));
                wptr2 = (wptr2 - layerNodeCount * (layerNodeCount + 1));
                for(l = l - 1; l > 0; l--) {
                    for(j = 0; j < layerNodeCount; j++) {
                        for(k = 0; k < layerNodeCount + 1; k++)
                            wptr2[j * (layerNodeCount + 1) + k]
                                = wptr1[j * (layerNodeCount + 1) + k]
                                - gamma0 * dLy * propagate(
                                layerCount, layerNodeCount,
                                x_i, wSwap1, z,
                                l, j, k
                            );
                        for(k = 0; k < layerNodeCount + 1; k++)
                            zcur[k] *= (1.0 - zcur[k]);
                    }
                    zcur = znxt;
                    znxt = (znxt - layerNodeCount - 1);
                    wptr1 = (wptr1 - layerNodeCount * (layerNodeCount + 1));
                    wptr2 = (wptr2 - layerNodeCount * (layerNodeCount + 1));
                }

                /* Input layer. */
                for(j = 0; j < layerNodeCount; j++) {
                    for(k = 0; k < FEATURE_COUNT; k++) {
                        wSwap2[j * FEATURE_COUNT + k]
                            = wSwap1[j * FEATURE_COUNT + k]
                            - gamma0 * dLy * propagate(
                                layerCount, layerNodeCount,
                                x_i, wSwap1, z,
                                0, j, k
                            );
                    }
                }
                
                /*** Swap weight buffers. ***/
                wptr1 = wSwap1;
                wSwap1 = wSwap2;
                wSwap2 = wptr1;
            }

/******************************************************************************/

        }
    }

    /*** If there are no hidden layers, then training amounts to the ***/
    /*** perceptron algorithm.                                       ***/
    else {

        /*** Train over epochs. ***/
        for(e = 0; e < epochs; e++) {
            /*** Shuffle examples. ***/
            shuffle(count, x, y);

/** Sequential 2: Simple dot product ******************************************/

            for(i = 0; i < count; i++) {
                x_i = (x + (i * FEATURE_COUNT));

                /*** Compute dot product. ***/
                yp = 0;
                for(j = 0; j < FEATURE_COUNT; j++)
                    yp += wSwap1[j] * x_i[j];

                /*** Save derivitive of square loss. ***/
                dLy = yp - y[i];

                /*** Update weights. ***/
                for(j = 0; j < FEATURE_COUNT; j++)
                    wSwap2[j] = wSwap1[j] - gamma0 * dLy * x_i[j];

                /*** Swap weight buffers. ***/
                wptr1 = wSwap1;
                wSwap1 = wSwap2;
                wSwap2 = wptr1;
            }

/******************************************************************************/

        }
    }

    /*** Cleanup memory. ***/
    if(w == wSwap1)
        freeWeights(&wSwap2);
    else {
        /*** Copy trained weights into w. ***/
        memcpy(w, wSwap1, wlen * sizeof(double));
        freeWeights(&wSwap1);
    }
    freez(&z);

    return 0;
}

/**
 * test
 */
static void test(
    const double * const x,
    const double * const y,
    const int count,
    const int layerCount,
    const int layerNodeCount,
    const double * const w
) {
    int i;
    const double *x_i = x;
    double y_i, y_p, p, r, f1, accuracy;
    /*** True/False Positive/Negative ***/
    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;

    for(i = 0; i < count; i++) {
        y_i = y[i];
        y_p = getPrediction(x_i, layerCount, layerNodeCount, w);
        if(y_i > 0 && y_p > 0)
            tp++;
        else if(y_i > 0 && y_p < 0)
            fn++;
        else if(y_i < 0 && y_p > 0)
            fp++;
        else
            tn++;
        x_i = (x_i + FEATURE_COUNT);
    }

    p = 0;
    r = 0;
    f1 = 0;
    if(tp > 0) {
        p = (double)tp / (double)(tp + fp);
        r = (double)tp / (double)(tp + fn);
        f1 = 2 * p * r / (p + r);
    }
    else {
        if(fp == 0)
            p = 1;
        if(fn == 0)
            r = 1;
    }

    accuracy = (double)(tp + tn) / (double)count;
    printf("accuracy: %f\nf1: %f\n", accuracy, f1);
}

/**
 * propagate
 */
static double propagate(
    const int layerCount,
    const int layerNodeCount,
    const double * const x_i,
    const double * const w,
    const double * const z,
    const int i,
    const int j,
    const int k
) {
    int j1;
    int wl = layerNodeCount * FEATURE_COUNT +
        i * layerNodeCount * (layerNodeCount + 1);
    double result = 0;
    const double * l = (i > 0 ? (z + (i - 1) * (layerNodeCount + 1)) : x_i);

    if(i + 1 < layerCount) {
        for(j1 = 0; j1 < layerNodeCount; j1++)
            result += l[k] * w[wl + j1 * (layerNodeCount + 1) + j + 1]
                * propagate(
                    layerCount, layerNodeCount,
                    x_i, w, z,
                    i + 1, j1, j + 1
                );
    }
    else {
        wl += layerNodeCount * (layerNodeCount + 1);
        result = l[k]
            * w[wl + j + 1]
            * z[(i + 1) * (layerNodeCount + 1) + j + 1];
    }

    return result;
}

/**
 * getPrediction
 */
static double getPrediction(
    const double * const x_i,
    const int layerCount,
    const int layerNodeCount,
    const double * const w
) {
    int i, j, k;
    const double *wptr = w;
    double *zcur, *znxt = v, *ztmp, dot, result = 0;
    if(layerCount > 0) {
        /*** First layer. ***/
        znxt[0] = 1;
        for(i = 1; i < layerNodeCount + 1; i++) {
            dot = 0;
            for(j = 0; j < FEATURE_COUNT; j++)
                dot += wptr[j] * x_i[j];
            // znxt[i] = dot;
            znxt[i] = 1.0 / (1.0 + exp(-dot));
            wptr = (wptr + FEATURE_COUNT);
        }
        zcur = znxt;
        znxt = u;
        /*** Remaining layers. ***/
        for(i = 1; i < layerCount; i++) {
            znxt[0] = 1;
            for(j = 0; j < layerNodeCount; j++) {
                dot = 0;
                for(k = 0; k < layerNodeCount + 1; k++)
                    dot += wptr[k] * zcur[k];
                // znxt[j + 1] = dot;
                znxt[j + 1] = 1.0 / (1.0 + exp(-dot));
                wptr = (wptr + layerNodeCount + 1);
            }
            ztmp = zcur;
            zcur = znxt;
            znxt = ztmp;
        }
        for(i = 0; i < layerNodeCount + 1; i++)
            result += wptr[i] * zcur[i];
    }
    else {
        for(i = 0; i < FEATURE_COUNT; i++)
            result += x_i[i] * w[i];
    }
    return result < 0 ? -1 : 1;
}

/**
 * parseArgs
 */
static int parseArgs(
    const int argc,
    char **argv,
    int *layerCount,
    int *layerNodeCount,
    int *epochs,
    double *gamma0
) {
    int i;
    for(i = 1; i < argc; i++) {
        /*** layerCount ***/
        if(strcmp(argv[i], "-l") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -1;
            }
            (*layerCount) = atoi(argv[i]);
            if((*layerCount) < 0) {
                fprintf(stderr, "error: number of hidden layers must be non-"
                    "negative\n");
                printUsage(argv[0]);
                return -2;
            }
        }
        /*** layerNodeCount ***/
        else if(strcmp(argv[i], "-n") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -3;
            }
            (*layerNodeCount) = atoi(argv[i]);
            if((*layerNodeCount) < 1) {
                fprintf(stderr, "error: number of layer nodes must be greater"
                    " than 0\n");
                printUsage(argv[0]);
                return -4;
            }
        }
        /*** epochs ***/
        else if(strcmp(argv[i], "-e") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -5;
            }
            (*epochs) = atoi(argv[i]);
            if((*epochs) < 1) {
                fprintf(stderr, "error: number of epochs must be greater than"
                    " 0\n");
                printUsage(argv[0]);
                return -6;
            }
        }
        /*** gamma0 ***/
        else if(strcmp(argv[i], "-g") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -7;
            }
            (*gamma0) = atof(argv[i]);
            if((*gamma0) <= 0) {
                fprintf(stderr, "error: gamma0 must be positive\n");
                printUsage(argv[0]);
                return -8;
            }
        }
        /*** unexpected argument ***/
        else {
            fprintf(stderr, "error: unexpected argument\n");
            printUsage(argv[0]);
            return -9;
        }
    }
    return 0;
}

/**
 * printUsage
 */
static void printUsage(const char *prgm) {
    printf("usage:\n");
    printf("\t%s [-e <int>] [-l <int>] [-n <int>] [-g <double>]\n\n", prgm);
    printf("Options:\n");
    printf("\t-e <int>       Specifies the number of epochs over which to"
        " train.\n");
    printf("\t               The default is 1.\n");
    printf("\t-l <int>       Specifies the number of hidden layers.\n");
    printf("\t               The default is 0.\n");
    printf("\t-n <int>       Specifies the number of nodes per hidden"
        " layer.\n");
    printf("\t               The default is 1.\n");
    printf("\t-g <double>    Specifies the gamma0 hyper parameter.\n");
    printf("\t               The default is 0.01.\n\n");
}