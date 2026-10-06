Add `warp.optim.linear.FSAI`, an adaptive factorized sparse approximate inverse preconditioner for
symmetric positive-definite `warp.sparse.BsrMatrix` inputs with `float16`, `float32` or `float64`
entries. It is also available as `warp.optim.linear.preconditioner(A, "fsai")`, and
can be refit after the matrix values change with `FSAI.update()`.
