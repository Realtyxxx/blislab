/*
 * --------------------------------------------------------------------------
 * BLISLAB
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *
 *
 * */

#include "bl_dgemm.h"

void bl_dgemm(int m, int n, int k, double *A, int lda, double *B, int ldb,
              double *C,  // must be aligned
              int ldc     // ldc must also be aligned
) {
  int i, j, p;

  // Early return if possible
  if (m == 0 || n == 0 || k == 0) {
    printf("bl_dgemm(): early return\n");
    return;
  }

  for (j = 0; j < n; j++) {  // Start 2-nd loop
    for (p = 0; p < k; p++) {
      // Start 1-st loop
      double *c_pointer = &C(0, j);
      for (i = 0; i < m - 4; i += 4) {  // Start 0-th loop

        // C[ j * ldc + i ] += A[ p * lda + i ] * B[ j * ldb + p ];
        *c_pointer += A(i, p) * B(p, j);            // Each operand is a MACRO defined in bl_dgemm() function.
        *(c_pointer + 1) += A(i + 1, p) * B(p, j);  // Each operand is a MACRO defined in bl_dgemm() function.
        *(c_pointer + 2) += A(i + 2, p) * B(p, j);  // Each operand is a MACRO defined in bl_dgemm() function.
        *(c_pointer + 3) += A(i + 3, p) * B(p, j);  // Each operand is a MACRO defined in bl_dgemm() function.
        c_pointer += 4;
      }  // End   0-th loop
      while (i < m) {
        *c_pointer += A(i, p) * B(p, j);
        i++;
        c_pointer++;
      }
    }  // End   1-st loop
  }    // End   2-nd loop
}
