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

#define TRAIN_SET "./data/data.train"
#define TEST_SET "./data/data.test"
#define FEATURE_COUNT 361
#define MAX_EXAMPLES 0x2000

static int load(char **filePath, double **x, double *y);

int main(int argc, char **argv) {
    double x[MAX_EXAMPLES][FEATURE_COUNT], y[MAX_EXAMPLES];
    int count;

    count = load(TRAIN_SET, x, y);
    if(count < 0)
        return count;

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
static int load(const char **filePath, double **x, double *y) {
    int i, count = 0, index, val;
    char token[40], line[0x400];
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
        if(sscanf(line, "%d", &val) < 1) {
            perror("error `load`: parsing label");
            fclose(file);
            return -3;
        }
        y[count] = val;
        printf("y = %d\n", val);

        /* Reset the example features. */
        x[count][0] = 1;
        for(i = 1; i < FEATURE_COUNT; i++)
            x[count][i] = 0;

        /* Parse example features from the line. */
        while(sscanf(line, "%s", token) == 1) {
            i = 0;
            while(i < 40 && token[i] != ':' && token[i] != 0)
                i++;
            if(i == 40 || token[i] == 0) {
                perror("error `load`: unexpcted token");
                fclose(file);
                return -4;
            }
            index = atoi(token);
            val = atoi(token + i + 1);
            x[count][index] = 1.0 - exp((double)(-1 * val));
            printf("x[%d][%d] = %d\n", count, index, val);
        }

        /* Empty the string and update example count. */
        line[0] = 0;
        count++;
    }

    /* Close the file and return the number of examples. */
    fclose(file);
    return count;
}