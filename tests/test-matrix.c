#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SseMatrixProvider,
};

int main()
{
    int i;
    bool not_equal = false;

    for (i = 0; i < 2; i++) {
        MatrixAlgo *algo = matrix_providers[i];

        Matrix dst, m, n, fixed;
        algo->assign(&m, (Mat4x4) {
            .values = {
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
            },
        });

        algo->assign(&n, (Mat4x4) {
            .values = {
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
                { 1, 2, 3, 4, },
                { 5, 6, 7, 8, },
            },
        });

        algo->mul(&dst, &m, &n);

        algo->assign(&fixed, (Mat4x4) {
            .values = {
                { 34,  44,  54,  64, },
                { 82, 108, 134, 160, },
                { 34,  44,  54,  64, },
                { 82, 108, 134, 160, },
            },
        });

        if (algo->equal(&dst, &fixed))
            printf("%s result equal!!\n", algo->info());
        else {
            printf("%s result not equal!!\n", algo->info());
            not_equal = true;
        }
        free(m.priv);
        free(n.priv);
        free(dst.priv);
    }
    if (not_equal)
        return -1;
    else
        return 0;
}
