#include "mat_op.h"
#include <blas.h>


namespace {

template<bool TA, bool TB, bool isOverWrite>
void CeqAxB_tmpl(const matw &A, const matw &B, matw &C)
{


}

}

void CeqAxB(const matw &A, const matw &B, matw &C, bool isOverWrite /*= false*/)
{
  ptrdiff_t M = A.H; // assert (M == C.H)
  ptrdiff_t K = A.W; // assert (K == B.H)
  ptrdiff_t N = B.W; // assert (N == C.W)

  float alpha = 1.0;
  float beta = isOverWrite? 0.0 : 1.0;

  sgemm(
    "n", "n",
    &M, &N, &K,
    &alpha,
    (float*)A.beg, &M,
    (float*)B.beg, &K,
    &beta,
    (float*)C.beg, &M);

  return;
}

void CeqATxB(const matw &A, const matw &B, matw &C, bool isOverWrite /*= false*/)
{

  return;
}

void CeqAxBT(const matw &A, const matw &B, matw &C, bool isOverWrite /*= false*/)
{

}
