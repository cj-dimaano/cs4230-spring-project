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
    double * const x,
    double * const y
) {
    int i, val, count = 0;
    char line[0x400], *token;
    FILE *file;

    /*** Open the file. ***/
    file = fopen(filePath, "r");
    if(file == NULL) {
        perror("error `load`: opening file");
        return -1;
    }

    /*** Read each line. ***/
    while(fgets(line, 0x400, file) != NULL) {
        if(line[0] == 0) {
            perror("error `load`: reading line");
            fclose(file);
            return -2;
        }

        /*** Get the label from the line. ***/
        token = strtok(line, " ");
        if(token == NULL) {
            fprintf(stderr, "error `load`: parsing label\n");
            fclose(file);
            return -3;
        }
        y[count] = (atoi(token) == 0 ? -1 : 1);

        /*** Reset the example features. ***/
        x[count * FEATURE_COUNT] = 1;
        for(i = 1; i < FEATURE_COUNT; i++)
            x[count * FEATURE_COUNT + i] = 0;

        /*** Parse example features from the line. ***/
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
            x[count * FEATURE_COUNT + i] = 1.0 - exp((double)(-val));
        }

        /*** Empty the string and update example count. ***/
        line[0] = 0;
        count++;
    }

    /*** Close the file and return the number of examples. ***/
    fclose(file);
    return count;
}

/**
 * fillWeights
 */
void fillWeights(
    const int len,
    double * const w
) {
    int i;
    srand(time(NULL));
    for(i = 0; i < len; i++)
        w[i] = ((double)rand() / (double)RAND_MAX) * 2 - 1;
}

/**
 * shuffle
 */
void shuffle(const int count, double * const x, double * const y) {
    int i, j, k;
    double tmpx, tmpy;
    srand(time(NULL));
    for(i = 0; i < count; i++) {
        j = rand() % count;
        for(k = 0; k < FEATURE_COUNT; k++) {
            tmpx = x[i * FEATURE_COUNT + k];
            x[i * FEATURE_COUNT + k] = x[j * FEATURE_COUNT + k];
            x[j * FEATURE_COUNT + k] = tmpx;
        }
        tmpy = y[i];
        y[i] = y[j];
        y[j] = tmpy;
    }
}
