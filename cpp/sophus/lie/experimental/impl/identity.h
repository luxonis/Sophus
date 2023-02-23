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

template <class TScalar, int kDim>
class IdentityImpl {
 public:
  using Scalar = TScalar;
  static int const kDof = 0;
  static int const kNumParams = 0;
  static int const kPointDim = kDim;
  static int const kAmbientDim = kDim;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, kDim>::Zero();
  }

  static auto areParamsValid(
      Eigen::Vector<Scalar, kNumParams> const& scale_factors)
      -> sophus::Expected<Success> {
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, kNumParams> {
    return tangent;
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kDof> {
    return params;
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    return mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>();
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  // group operations

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return params;
  }

  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return lhs_params;
  }

  // Point actions

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return point;
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }
  // Matrices

  static auto compactMatrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, kPointDim, kAmbientDim>::Identity();
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return compactMatrix(params);
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
    return Eigen::Matrix<Scalar, kPointDim, kDof>::Zero();
  }

  // for tests

  static auto exampleTangents() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    return std::vector<Eigen::Vector<Scalar, kDof>>();
  }

  static auto exampleParams()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return std::vector<Eigen::Vector<Scalar, kNumParams>>();
  }
};

template <class TScalar>
using Identity2Impl = IdentityImpl<TScalar, 2>;

template <class TScalar>
using Identity3Impl = IdentityImpl<TScalar, 3>;

}  // namespace lie
}  // namespace sophus
