#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SRC1_ROWS 40
#define SRC1_COLS 20
#define SRC2_ROWS 20
#define SRC2_COLS 40

MatrixAlgo *matrix_providers[] = {
    &NaiveMatrixProvider,
    &SseMatrixProvider,
};

int **get_space(int row, int col)
{
    int **src;
    if (!(src = (int **)malloc(row * sizeof(int *))))
        return NULL;
    for(int i = 0; i < row; i++)
        if (!(src[i] = (int *)malloc(col * sizeof(int)))) {
            free(src);
            return NULL;
        }
    return src;
}

void rand_data(int row, int col, int **src)
{
    srand(time(NULL));

    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            src[i][j] = rand() % 10;
}

int main()
{
    int i, j, k;
    bool not_equal = false;
    int **src1, **src2, **mul_ans;

    /*allocate*/
    if (!(src1 = get_space(SRC1_ROWS, SRC1_COLS)))
        return -1;
    if (!(src2 = get_space(SRC2_ROWS, SRC2_COLS)))
        return -1;
    if (!(mul_ans = get_space(SRC1_ROWS, SRC2_COLS)))
        return -1;

    /*get rand data*/
    rand_data(SRC1_ROWS, SRC1_COLS, src1);
    rand_data(SRC2_ROWS, SRC2_COLS, src2);

    /*calculate the answer*/
    for (i = 0; i < SRC1_ROWS; i ++)
        for (j = 0; j < SRC2_COLS; j++)
            mul_ans[i][j] = 0;
    if (SRC1_COLS == SRC2_ROWS)
        for (i = 0; i < SRC1_ROWS; i ++)
            for (j = 0; j < SRC2_COLS; j++)
                for (k = 0; k < SRC1_COLS; k++)
                    mul_ans[i][j] += src1[i][k] * src2[k][j];
    else {
        printf("ERROR:src1 rows must equal to src2 cols\n");
        free(src1);
        free(src2);
        free(mul_ans);
        return -1;
    }

    for (i = 0; i < 2; i++) {
        MatrixAlgo *algo = matrix_providers[i];

        Matrix dst, m, n, fixed;

        algo->assign(&m, SRC1_ROWS, SRC1_COLS, src1);
        algo->assign(&n, SRC2_ROWS, SRC2_COLS, src2);

        if (!algo->mul(&dst, &m, &n))
            printf("ERROR:mul err\n");

        algo->assign(&fixed, SRC1_ROWS, SRC2_COLS, mul_ans);

        if (algo->equal(&dst, &fixed))
            printf("%s result equal!!\n", algo->info());
        else {
            printf("%s result not equal!!\n", algo->info());
            not_equal = true;
        }

        free(m.priv);
        free(n.priv);
        free(dst.priv);
        free(fixed.priv);
    }

    free(src1);
    free(src2);
    free(mul_ans);

    if (not_equal)
        return -1;
    else
        return 0;
}
