#ifndef MATRIX_H_
#define MATRIX_H_

#include <stdbool.h>

/* predefined shortcut */
#define DECLARE_MATRIX(col, row) \
    typedef struct { float values[col][row]; } Mat ## col ## x ## row
DECLARE_MATRIX(3, 3);
DECLARE_MATRIX(4, 4);

typedef struct {
    int row, col;
    void *priv;
} Matrix;

typedef struct {
    void (*assign)(Matrix *thiz, Mat4x4);
    bool (*equal)(const Matrix *l, const Matrix *r);
    bool (*mul)(Matrix *dst, const Matrix *l, const Matrix *r);
    char *(*info)(void);
} MatrixAlgo;

/* Available matrix providers */
extern MatrixAlgo NaiveMatrixProvider;
extern MatrixAlgo SseMatrixProvider;

#endif /* MATRIX_H_ */
