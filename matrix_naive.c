#include "matrix.h"
#include <stdlib.h>
#include <string.h>

struct naive_priv {
    float values[4][4];
};

#define PRIV(x) \
    ((struct naive_priv *) ((x)->priv))

static void assign(Matrix *thiz, Mat4x4 data)
{
    /* FIXME: don't hardcode row & col */
    thiz->row = thiz->col = 4;

    thiz->priv = malloc(4 * 4 * sizeof(float));
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            PRIV(thiz)->values[i][j] = data.values[i][j];
}

static const float epsilon = 1 / 10000.0;

static bool equal(const Matrix *l, const Matrix *r)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (PRIV(l)->values[i][j] + epsilon < PRIV(r)->values[i][j] ||
                    PRIV(r)->values[i][j] + epsilon < PRIV(l)->values[i][j])
                return false;
    return true;
}

static bool mul(Matrix *dst, const Matrix *l, const Matrix *r)
{
    if(!(dst->priv = malloc(4 * 4 * sizeof(float))))
        return false;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++)
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
