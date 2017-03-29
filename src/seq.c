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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAIN_SET "./data/data.train"
#define TEST_SET "./data/data.test"
#define FEATURE_COUNT 361
#define MAX_EXAMPLES 0x2000

static int load(const char *filePath, double **x, double *y);

int main(int argc, char **argv) {
    double **x, *y;
    int count, i;

    /* Allocate memory for examples. (Assume there is enough memory.) */
    x = (double **)calloc(MAX_EXAMPLES, sizeof(double *));
    y = (double *)calloc(MAX_EXAMPLES, sizeof(double));
    for(i = 0; i < MAX_EXAMPLES; i++)
        x[i] = (double *)calloc(FEATURE_COUNT, sizeof(double));

    /* Load training data. */
    count = load(TRAIN_SET, x, y);
    if(count < 0)
        return count;

    /* Train classifier. */

    /* Load test data. */

    /* Evaluate classifier accuracy. */

    /* Free memory from examples. */
    free(y);
    for(i = 0; i < MAX_EXAMPLES; i++)
        free(x[i]);
    free(x);

    return 0;
}

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
static int load(const char *filePath, double **x, double *y) {
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

        /* Reset the example features. */
        x[count][0] = 1;
        for(i = 1; i < FEATURE_COUNT; i++)
            x[count][i] = 0;

        /* Parse example features from the line. */
        while((token = strtok(NULL, " ")) != NULL) {
            if(sscanf(token, "%d:%d", &i, &val) < 2) {
                fprintf(stderr, "error `load`: unable to parse token: %s\n", token);
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