/*******************************************************************************
File: seq.c
Created by: CJ Dimaano
Date created: March 29, 2017
*******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <curand.h>
#include <curand_kernel.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

extern "C" {
#include "data.h"
#include "mem.h"
}

/** Declarations **************************************************************/

static int train(
    double * const x,
    double * const y,
    const int count,
    const int epochs,
    const double c,
    const double gamma0,
    const double s,
    double * const w
);
static void test(
    const double * const x,
    const double * const y,
    const int count,
    const double * const w
);
static double getPrediction(
    const double * const x_i,
    const double * const w
);
static int parseArgs(const int, char **, int *, double *, double *, double *);

/** Main **********************************************************************/

int main(int argc, char **argv) {
    double *w, *x, *y, c = 1 << 10, gamma0 = 0.01, s = 10e5;
    int ret, epochs = 200;

    /*** Parse command-line arguments. ***/
    ret = parseArgs(argc, argv, &epochs, &c, &gamma0, &s);
    if(ret < 0) {
        return -1;
    }
    printf("epochs: %d\n", epochs);
    printf("C: %f\n", c);
    printf("gamma0: %f\n", gamma0);
    printf("s: %f\n", s);

    /*** Allocate memory. ***/
    if(init(&x, &y, &w) < 0) {
        return -2;
    }

    /*** Load training data. ***/
    ret = load(TRAIN_SET, x, y);
    if(ret < 0) {
        cleanup(&x, &y, &w);
        return -3;
    }

    /*** Train classifier. ***/
    ret = train(x, y, ret, epochs, c, gamma0, s, w);
    if(ret < 0) {
        cleanup(&x, &y, &w);
        return -4;
    }

    /*** Load test data. ***/
    ret = load(TEST_SET, x, y);
    if(ret < 0) {
        cleanup(&x, &y, &w);
        return ret;
    }

    /*** Test classifier accuracy. ***/
    test(x, y, ret, w);

    /*** Cleanup memory from examples. ***/
    cleanup(&x, &y, &w);

    return 0;
}

/** Static functions **********************************************************/

extern __global__ void trainCompute(double *p_w, double *p_x, double gamma0, double a, double b, double c, double e, int i, double t) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int j = tx + 32 * bx;

  if (j < FEATURE_COUNT) {
    p_w[j] = p_w[j] - (gamma0 / (1 + gamma0 * t / c)) * (a * p_x[i * FEATURE_COUNT + j] + b * p_w[j]);
  }

  __syncthreads();
}

void generateMoves(int count, int * moves) {
  int i, j, tmp;

  for (i = 0; i < count; i++) {
    moves[i] = i;
  }

  srand(time(NULL));
  for (i = 0; i < count; i++) {
    j = rand() % count;
    tmp = moves[i];
    moves[i] = moves[j];
    moves[j] = tmp;
  }
}

extern __global__ void performMoves(int count, int * moves, double * p_x, double * p_x_new, double * p_y, double *p_y_new) {
  int tid = threadIdx.x + 32 * blockIdx.x;

  if (tid < count) {
    memcpy((p_x_new + moves[tid] * FEATURE_COUNT), (p_x + tid * FEATURE_COUNT), FEATURE_COUNT * sizeof(double));
    p_y_new[moves[tid]] = p_y[tid];
  }
  __syncthreads();
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

int compare(double *a, double *b, int size, double threshold) {
  int i;
  for (i=0; i<size; i++) {
    if (abs(a[i]-b[i]) > threshold) return 0;
  }
  return 1;
}


/**
 * train
 *
 * @summary
 *   Trains the weights of the classifier.
 */
static int train(
    double * const x,
    double * const y,
    const int count,
    const int epochs,
    const double c,
    const double gamma0,
    const double s,
    double * const w
) {
    int epoch, i, j;
    double a, b = 2 / (s * s), t = 1, dot, e;

    // Allocate buffers on the GPU.
    int x_size = MAX_EXAMPLES * FEATURE_COUNT * sizeof(double);
    int y_size = MAX_EXAMPLES * sizeof(double);
    int w_size = FEATURE_COUNT * sizeof(double);

    double * p_x;
    cudaMalloc((void **)&p_x, x_size);

    double * p_x_new;
    cudaMalloc((void **)&p_x_new, x_size);

    double * p_x_temp;

    int * moves = (int *)malloc(count * sizeof(int));

    int * p_moves;
    cudaMalloc((void **)&p_moves, count * sizeof(int));

    double *p_y;
    cudaMalloc((void **)&p_y, y_size);

    double *p_y_new;
    cudaMalloc((void **)&p_y_new, y_size);

    double *p_y_temp;

    double * p_w;
    cudaMalloc((void **)&p_w, w_size);

    // Copy host buffers into device buffers.
    cudaMemcpy(p_x, x, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(p_w, w, w_size, cudaMemcpyHostToDevice);

    // Perform the computation.
    dim3 shuffleGrid((count+31)/32, 1);
    dim3 shuffleBlock(32, 1);

    dim3 dimGrid((FEATURE_COUNT+31)/32, 1);
    dim3 dimBlock(32, 1);

    for(epoch = 0; epoch < epochs; epoch++) {

        generateMoves(count, moves);

        cudaMemcpy(p_moves, moves, count * sizeof(int), cudaMemcpyHostToDevice);
        performMoves<<<shuffleGrid,shuffleBlock>>>(count, p_moves, p_x, p_x_new, p_y, p_y_new);

        gpuErrchk( cudaDeviceSynchronize() );

        cudaMemcpy(y, p_y_new, y_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(x, p_x_new, x_size, cudaMemcpyDeviceToHost);

        gpuErrchk( cudaDeviceSynchronize() );

        p_x_temp = p_x;
        p_x = p_x_new;
        p_x_new = p_x_temp;

        p_y_temp = p_y;
        p_y = p_y_new;
        p_y_new = p_y_temp;

        for(i = 0; i < count; i++) {
            dot = 0;
            for(j = 0; j < FEATURE_COUNT; j++)
                dot += x[i * FEATURE_COUNT + j] * w[j];
            e = exp(-y[i] * dot);
            a = -y[i] * e / (1 + e);

            trainCompute<<<dimGrid,dimBlock>>>(p_w, p_x, gamma0, a, b, c, e, i, t);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );

            cudaMemcpy(w, p_w, w_size, cudaMemcpyDeviceToHost);

            t += 1.0;
        }
    }

    // Copy the result off of the GPU.
    cudaMemcpy(w, p_w, w_size, cudaMemcpyDeviceToHost);

    // Free the created buffers.
    cudaFree(p_x);
    cudaFree(p_x_new);
    cudaFree(p_y);
    cudaFree(moves);
    cudaFree(p_w);

    return 0;
}

/**
 * test
 */
static void test(
    const double * const x,
    const double * const y,
    const int count,
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
        y_p = getPrediction(x_i, w);
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
    printf("tp, fp, tn, fn: %d, %d, %d, %d\n", tp, fp, tn, fn);
    printf("accuracy: %f\nf1: %f\n", accuracy, f1);
}

/**
 * getPrediction
 */
static double getPrediction(
    const double * const x_i,
    const double * const w
) {
    int i;
    double dot = 0;
    for(i = 0; i < FEATURE_COUNT; i++)
        dot += x_i[i] * w[i];
    return dot > 0 ? 1 : -1;
}

/**
 * printUsage
 */
static void printUsage(const char *prgm) {
    printf("usage:\n");
    printf("\t%s [-e <int>] [-C <double>] [-s <double>] [-g <double>]\n\n", prgm);
    printf("Options:\n");
    printf("\t-e <int>       Specifies the number of epochs over which to"
        " train.\n");
    printf("\t-C <double>    Specifies the C hyper parameter.\n");
    printf("\t-g <double>    Specifies the gamma0 hyper parameter.\n");
    printf("\t-s <double>    Specifies the s hyper parameter.\n");
}

/**
 * parseArgs
 */
static int parseArgs(
    const int argc,
    char **argv,
    int *epochs,
    double *c,
    double *gamma0,
    double *s
) {
    int i;
    for(i = 1; i < argc; i++) {
        /*** epochs ***/
        if(strcmp(argv[i], "-e") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -1;
            }
            (*epochs) = atoi(argv[i]);
            if((*epochs) < 1) {
                fprintf(stderr, "error: number of epochs must be greater than"
                    " 0\n");
                printUsage(argv[0]);
                return -2;
            }
        }
        /*** C ***/
        else if(strcmp(argv[i], "-C") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -3;
            }
            (*c) = atof(argv[i]);
            if((*c) <= 0) {
                fprintf(stderr, "error: C must be positive\n");
                printUsage(argv[0]);
                return -4;
            }
        }
        /*** gamma0 ***/
        else if(strcmp(argv[i], "-g") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -5;
            }
            (*gamma0) = atof(argv[i]);
            if((*gamma0) <= 0) {
                fprintf(stderr, "error: gamma0 must be positive\n");
                printUsage(argv[0]);
                return -6;
            }
        }
        /*** s ***/
        else if(strcmp(argv[i], "-s") == 0) {
            i++;
            if(i == argc) {
                fprintf(stderr, "error: unexpected end of argument list\n");
                printUsage(argv[0]);
                return -7;
            }
            (*s) = atof(argv[i]);
            if((*s) <= 0) {
                fprintf(stderr, "error: s must be positive\n");
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
