/*******************************************************************************
File: data.c
Created by: CJ Dimaano
Date created: March 30, 2017

Data management stuff.

Compile with:
```
$ gcc -Wall -lm -c -o data.o data.c
```

*******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "data.h"


/**
 * load
 *
 * @summary
 *   Loads a set of examples from the given file.
 *
 * @description
 *   Examples are separated by line. The first token in the line is the label;
 *   subsequent tokens are features in the format `<index>:<value>`, where
 *   `<index>` is in the range [1..FEATURE_COUNT]. Feature values are converted
 *   to a number using the formula `1 - exp(-<value>)`.
 */
int load(
    const char * const filePath,
    double * const * const x,
    double * const y
) {
    int i, val, count = 0;
    char line[0x400], *token;
    FILE *file;

    /* Open the file. */
    file = fopen(filePath, "r");
    if(file == NULL) {
        perror("error `load`: opening file");
        return -1;
    }

    /* Read each line. */
    while(fgets(line, 0x400, file) != NULL) {
        if(line[0] == 0) {
            perror("error `load`: reading line");
            fclose(file);
            return -2;
        }

        /* Get the label from the line. */
        token = strtok(line, " ");
        if(token == NULL) {
            fprintf(stderr, "error `load`: parsing label\n");
            fclose(file);
            return -3;
        }
        y[count] = atoi(token);
        if(y[count] == 0.0)
            y[count] = -1;

        /* Reset the example features. */
        x[count][0] = 1;
        for(i = 1; i < FEATURE_COUNT; i++)
            x[count][i] = 0;

        /* Parse example features from the line. */
        while((token = strtok(NULL, " ")) != NULL) {
            if(sscanf(token, "%d:%d", &i, &val) < 2) {
                fprintf(
                    stderr,
                    "error `load`: unable to parse token: %s\n",
                    token
                );
                fclose(file);
                return -4;
            }
            x[count][i] = 1.0 - exp((double)(-1 * val));
        }

        /* Empty the string and update example count. */
        line[0] = 0;
        count++;
    }

    /* Close the file and return the number of examples. */
    fclose(file);
    return count;
}

/**
 * fillWeights
 */
void fillWeights(
    const int layerCount,
    const int layerNodeCount,
    double * const * const * const w
) {
    int i, j, k;
    srand(time(NULL));
    if(layerCount > 0) {
        for(i = 0; i < layerNodeCount; i++) {
            for(j = 0; j < FEATURE_COUNT; j++)
                w[0][i][j] = ((double)rand() / (double)RAND_MAX) * 2 - 1;
        }
        for(i = 1; i < layerCount; i++) {
            for(j = 0; j < layerNodeCount; j++)
                for(k = 0; k < layerNodeCount + 1; k++)
                    w[i][j][k] = ((double)rand() / (double)RAND_MAX) * 2 - 1;
        }
        for(i = 0; i < FEATURE_COUNT; i++)
            w[layerCount][0][i] = ((double)rand() / (double)RAND_MAX) * 2 - 1;
    }
    else {
        for(i = 0; i < FEATURE_COUNT; i++)
            w[0][0][i] = ((double)rand() / (double)RAND_MAX) * 2 - 1;
    }
}

/**
 * shuffle
 */
void shuffle(const int count, double **x, double *y) {
    int i, j;
    double *tmpx, tmpy;
    srand(time(NULL));
    for(i = 0; i < count; i++) {
        j = rand() % count;
        tmpx = x[i];
        x[i] = x[j];
        x[j] = tmpx;
        tmpy = y[i];
        y[i] = y[j];
        y[j] = tmpy;
    }
}
