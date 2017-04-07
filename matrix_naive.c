#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

struct naive_priv {
    int **values;
};

#define PRIV(x) \
    ((struct naive_priv *) ((x)->priv))

static void assign(Matrix *thiz, int row, int col, int **data)
{
    int i, j;

    thiz->row = row;
    thiz->col = col;

    /* allocate for thiz */
    if (!(thiz->priv = malloc(thiz->row * thiz->col * sizeof(int))))
        return;
    if (!(PRIV(thiz)->values = (int **)malloc(row * sizeof(int *))))
        return;
    for (i = 0; i < thiz->row; i++)
        if (!(PRIV(thiz)->values[i] = (int *)malloc(thiz->col * sizeof(int))))
            return;

    for (i = 0; i < thiz->row; i++)
        for (j = 0; j < thiz->col; j++)
            PRIV(thiz)->values[i][j] = data[i][j];
}

static bool equal(const Matrix *l, const Matrix *r)
{
    if ((l->row != r->row) || (l->col != r->col))
        return false;

    for (int i = 0; i < l->row; i++)
        for (int j = 0; j < l->col; j++)
            if (PRIV(l)->values[i][j] != PRIV(r)->values[i][j])
                return false;
    return true;
}

static bool mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    int i, j, k;

    if (l->col != r->row)
        return false;

    dst->row = l->row;
    dst->col = r->col;

    /* allocate for dst */
    if (!(dst->priv = malloc(dst->row * dst->col * sizeof(int))))
        return false;
    if (!(PRIV(dst)->values = (int **)malloc(dst->row * sizeof(int *))))
        return false;
    for (i = 0; i < dst->row; i++)
        if(!(PRIV(dst)->values[i] = (int *)malloc(dst->col * sizeof(int))))
            return false;

    for (i = 0; i < l->row; i++)
        for (j = 0; j < r->col; j++)
            for (k = 0; k < l->col; k++)
                PRIV(dst)->values[i][j] += PRIV(l)->values[i][k] *
                                           PRIV(r)->values[k][j];
    return true;
}

static char *info(void)
{
    return "naive";
}

MatrixAlgo NaiveMatrixProvider = {
    .assign = assign,
    .equal = equal,
    .mul = mul,
    .info = info,
};
