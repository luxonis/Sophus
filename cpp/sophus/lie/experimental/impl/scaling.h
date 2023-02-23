// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

/// @file
/// Cartesian - Euclidean vector space as Lie group

#pragma once
#include "sophus/lie/experimental/lie_group_concept.h"

namespace sophus {
namespace lie {

template <class TScalar>
class Scaling2Impl {
 public:
  using Scalar = TScalar;
  static int const kDof = 2;
  static int const kNumParams = 2;
  static int const kPointDim = 2;
  static int const kAmbientDim = 2;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, 2>(1.0, 1.0);
  }

  static auto areParamsValid(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar>;

    if (!(scale_factors.array() >= kThr).all()) {
      return FARM_UNEXPECTED(
          "scale factors ({}, {}) not positive.\n",
          "thr: {}",
          scale_factors[0],
          scale_factors[1],
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& log_scale_factors)
      -> Eigen::Vector<Scalar, kNumParams> {
    using std::exp;
    return Eigen::Vector<Scalar, 2>(
        exp(log_scale_factors[0]), exp(log_scale_factors[1]));
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Vector<Scalar, kDof> {
    using std::log;
    return Eigen::Vector<Scalar, 2>(
        log(scale_factors[0]), log(scale_factors[1]));
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& scale_factors)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    for (int i = 0; i < kDof; ++i) {
      mat.diagonal()[i] = scale_factors[i];
    }
    return mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return mat.diagonal();
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  // group operations

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, 2>(
        1.0 / scale_factors[0], 1.0 / scale_factors[1]);
  }

  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, 2>(
        lhs_params[0] * rhs_params[0], lhs_params[1] * rhs_params[1]);
  }

  // Point actions

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return Eigen::Vector<Scalar, 2>(
        point[0] * scale_factors[0], point[1] * scale_factors[1]);
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }
  // Matrices

  static auto compactMatrix(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {scale_factors[0], 0.0}, {0.0, scale_factors[1]}};
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return compactMatrix(scale_factors);
  }

  // subgroup concepts

  static auto matV(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto matVInverse(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return matrix(-point);
  }

  // for tests

  static auto exampleTangents() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    return std::vector<Eigen::Vector<Scalar, kDof>>({
        Eigen::Vector<Scalar, kDof>({std::exp(1.0), std::exp(1.0)}),
        Eigen::Vector<Scalar, kDof>({1.1, 1.1}),
        Eigen::Vector<Scalar, kDof>({2.0, 1.1}),
        Eigen::Vector<Scalar, kDof>({2.0, std::exp(1.0)}),
    });
  }

  static auto exampleParams()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return std::vector<Eigen::Vector<Scalar, kDof>>(
        {Eigen::Vector<Scalar, kDof>({1.0, 1.0}),
         Eigen::Vector<Scalar, kDof>({1.0, 2.0}),
         Eigen::Vector<Scalar, kDof>({1.5, 1.0}),
         Eigen::Vector<Scalar, kDof>({5.0, 1.237})});
  }
};
}  // namespace lie
}  // namespace sophus
