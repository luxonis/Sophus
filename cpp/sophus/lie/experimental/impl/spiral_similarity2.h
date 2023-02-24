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
#include "sophus/lie/experimental/impl/complex.h"
#include "sophus/lie/experimental/lie_group_concept.h"

namespace sophus {
namespace lie {

template <class TScalar>
class SpiralSimilarity2Impl {
 public:
  using Scalar = TScalar;
  using Complex = ComplexNumberImpl<TScalar>;

  static int const kDof = 2;
  static int const kNumParams = 2;
  static int const kPointDim = 2;
  static int const kAmbientDim = 2;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, 2>(1.0, 0.0);
  }

  static auto areParamsValid(
      Eigen::Vector<Scalar, kNumParams> const& non_zero_complex)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar> * kEpsilon<Scalar>;
    const Scalar squared_norm = non_zero_complex.squaredNorm();
    using std::abs;
    if (!(squared_norm < kThr || squared_norm > 1.0 / kThr)) {
      return FARM_UNEXPECTED(
          "complex number ({}, {}) is too large or too small.\n"
          "Squared norm: {}, thr: {}",
          non_zero_complex[0],
          non_zero_complex[1],
          squared_norm,
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& angle_logscale)
      -> Eigen::Vector<Scalar, kNumParams> {
    using std::exp;
    using std::max;
    using std::min;

    Scalar const sigma = angle_logscale[1];
    Scalar s = exp(sigma);
    // Ensuring proper scale
    s = max(s, kEpsilonPlus<Scalar>);
    s = min(s, Scalar(1.) / kEpsilonPlus<Scalar>);
    Eigen::Vector2<Scalar> z =
        Rotation2Impl<Scalar>::exp(angle_logscale.template head<1>())
            .unitComplex();
    z *= s;
    return z;
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& complex)
      -> Eigen::Vector<Scalar, kDof> {
    using std::log;
    Eigen::Vector<Scalar, kDof> theta_sigma;
    theta_sigma[0] = Rotation2Impl<Scalar>::log(complex)[0];
    theta_sigma[1] = log(complex.norm());
    return theta_sigma;
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& angle_logscale)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {angle_logscale[1], -angle_logscale[0]},
        {angle_logscale[0], angle_logscale[1]}};
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>{mat(1, 0), mat(0,0)};
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, 1, 1>::Identity();
  }

  // group operations

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& unit_complex)
      -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, kNumParams>(
        unit_complex.x(), -unit_complex.y());
  }

  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    auto result = Complex::multiplication(lhs_params, rhs_params);
    Scalar const squared_scale = result.squaredNorm();

    if (squared_scale < kEpsilon<ResultT> * kEpsilon<ResultT>) {
      /// Saturation to ensure class invariant.
      result_complex.normalize();
      result_complex *= kEpsilonPlus<ResultT>;
    }
    if (squared_scale > Scalar(1.) / (kEpsilon<ResultT> * kEpsilon<ResultT>)) {
      /// Saturation to ensure class invariant.
      result_complex.normalize();
      result_complex /= kEpsilonPlus<ResultT>;
    }
    return RxSo2Product<TOtherDerived>(result_complex);
  }

  // Point actions
  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& unit_complex,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return Complex::multiplication(unit_complex, point);
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  // matrices

  static auto compactMatrix(
      Eigen::Vector<Scalar, kNumParams> const& unit_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, 2, 2>{
        {unit_complex.x(), -unit_complex.y()},
        {unit_complex.y(), unit_complex.x()}};
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& unit_complex)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return compactMatrix(unit_complex);
  }

  // Sub-group concepts
  static auto matV(Eigen::Vector<Scalar, kDof> const&)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto matVInverse(Eigen::Vector<Scalar, kDof> const&)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return Eigen::Matrix<Scalar, kPointDim, kPointDim>::Identity();
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const&,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, 2, 1>(point[1], -point[0]);
  }

  // for tests

  static auto exampleTangents() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    return std::vector<Eigen::Vector<Scalar, kDof>>({
        Eigen::Vector<Scalar, kDof>{0.0},
        Eigen::Vector<Scalar, kDof>{0.00001},
        Eigen::Vector<Scalar, kDof>{1.0},
        Eigen::Vector<Scalar, kDof>{-1.0},
        Eigen::Vector<Scalar, kDof>{5.0},
        Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar>},
        Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar> + 0.00001},
    });
  }

  static auto exampleParams()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return std::vector<Eigen::Vector<Scalar, kNumParams>>({
        Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{0.0}),
        Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{1.0}),
        Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{0.5 * kPi<Scalar>}),
        Rotation2Impl::exp(Eigen::Vector<Scalar, kDof>{kPi<Scalar>}),
    });
  }
};

}  // namespace lie
}  // namespace sophus
