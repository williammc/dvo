#pragma once
#include <Eigen/Core>
#include <dvo/core/math_sse.h>
#include <dvo/util/stopwatch.h>

#include <iostream>

typedef Eigen::Matrix<float, Eigen::Dynamic, 6> JacobianWorkspace;
typedef Eigen::Matrix<float, 6, 6> Hessian;

static const int Dimension = 2;

void do_reduction(const JacobianWorkspace &j,
                  const Eigen::Matrix<float, Dimension, Dimension> &alpha,
                  Hessian &A) {
  A.setZero();

  for (int idx = 0; idx < j.rows(); idx += Dimension) {
    if (Dimension == 1)
      A += j.block<Dimension, 6>(idx, 0).transpose() * alpha *
           j.block<Dimension, 6>(idx, 0);
    else
      A += j.block<Dimension, 6>(idx, 0).transpose() * alpha *
           j.block<Dimension, 6>(idx, 0);
  }
}

void do_optimized_reduction(
    const JacobianWorkspace &j,
    const Eigen::Matrix<float, Dimension, Dimension> &alpha, Hessian &A) {
  dvo::core::OptimizedSelfAdjointMatrix6x6f A_opt;
  A_opt.setZero();

  Eigen::Matrix2f my_alpha;
  my_alpha.block<Dimension, Dimension>(0, 0) = alpha;

  for (int idx = 0; idx < j.rows(); idx += Dimension) {
    if (Dimension == 1)
      A_opt.rankUpdate(j.block<1, 6>(idx, 0).transpose(), my_alpha(0, 0));
    else
      A_opt.rankUpdate(j.block<2, 6>(idx, 0), my_alpha);
  }

  A_opt.toEigen(A);
}

int main(int argc, char **argv) {
  Hessian A1, A2;

  Eigen::Matrix<float, Dimension, Dimension> alpha;
  alpha.setRandom();

  if (Dimension > 1)
    alpha(0, 1) = alpha(1, 0); // make symmetric

  JacobianWorkspace J;
  J.resize(640 * 480 * Dimension, Eigen::NoChange);
  J.setRandom();

  dvo::util::stopwatch w1("unoptimized", int(1e3 * 0.5)),
      w2("optimized", int(1e3 * 0.5));

  for (int idx = 0; idx < 1e2; idx++)
    do_reduction(J, alpha, A1);

  for (int idx = 0; idx < 1e3; idx++) {
    w1.start();
    do_reduction(J, alpha, A1);
    w1.stopAndPrint();
  }

  for (int idx = 0; idx < 1e2; idx++)
    do_optimized_reduction(J, alpha, A2);

  for (int idx = 0; idx < 1e3; idx++) {
    w2.start();
    do_optimized_reduction(J, alpha, A2);
    w2.stopAndPrint();
  }
}
