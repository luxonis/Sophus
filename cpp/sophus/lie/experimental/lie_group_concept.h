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
#include "sophus/common/concept_utils.h"
#include "sophus/common/types.h"
#include "sophus/linalg/homogeneous.h"
#include "sophus/linalg/skew_symmetric.h"

namespace sophus {
namespace lie {

template <class T>
concept LieGroupImplConcept =
    (T::kPointDim == 2 || T::kPointDim == 3) &&  // 2d or 3d points
    (T::kPointDim == T::kAmbientDim  // inhomogeneous point representation
     ||
     T::kPointDim + 1 == T::kAmbientDim)  // or homogeneous point representation
    &&
    requires(
        T g,
        Eigen::Vector<typename T::Scalar, T::kDof> tangent,
        Eigen::Vector<typename T::Scalar, T::kPointDim> point,
        Eigen::Vector<typename T::Scalar, T::kNumParams> params,
        Eigen::Matrix<typename T::Scalar, T::kAmbientDim, T::kAmbientDim>
            matrix,
        Eigen::Matrix<typename T::Scalar, T::kDof, T::kDof> adjoint) {
      // constructors and factories
      {
        T::identityParams()
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

      { T::areParamsValid(params) } -> ConvertibleTo<sophus::Expected<Success>>;

      // Manifold / Lie Group concepts

      {
        T::exp(tangent)
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

      {
        T::log(params)
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kDof>>;

      {
        T::hat(tangent)
        } -> ConvertibleTo<
            Eigen::Matrix<typename T::Scalar, T::kAmbientDim, T::kAmbientDim>>;

      {
        T::vee(matrix)
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kDof>>;

      {
        T::adj(params)
        } -> ConvertibleTo<Eigen::Matrix<typename T::Scalar, T::kDof, T::kDof>>;

      // group operations
      {
        T::multiplication(params, params)
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

      {
        T::inverse(params)
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kNumParams>>;

      // Point actions

      {
        T::action(params, point)
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kPointDim>>;

      {
        T::toAmbient(point)
        } -> ConvertibleTo<Eigen::Vector<typename T::Scalar, T::kAmbientDim>>;

      // Matrices

      {
        T::compactMatrix(params)
        } -> ConvertibleTo<
            Eigen::Matrix<typename T::Scalar, T::kPointDim, T::kAmbientDim>>;

      {
        T::matrix(params)
        } -> ConvertibleTo<
            Eigen::Matrix<typename T::Scalar, T::kAmbientDim, T::kAmbientDim>>;

      // for tests
      {
        T::exampleTangents()
        } -> ConvertibleTo<
            std::vector<Eigen::Vector<typename T::Scalar, T::kDof>>>;

      {
        T::exampleParams()
        } -> ConvertibleTo<
            std::vector<Eigen::Vector<typename T::Scalar, T::kNumParams>>>;
    };

template <class T>
concept LeftJacobianImplConcept =
    LieGroupImplConcept<T> &&
    requires(
        T g,
        Eigen::Vector<typename T::Scalar, T::kNumParams> params,
        Eigen::Vector<typename T::Scalar, T::kPointDim> point) {
      {
        T::leftJacobian(params)
        } -> ConvertibleTo<
            Eigen::Matrix<typename T::Scalar, T::kDof, T::kDof>>;

      {
        T::leftJacobianInverse(params)
        } -> ConvertibleTo<
            Eigen::Matrix<typename T::Scalar, T::kDof, T::kDof>>;

      {
        T::topRightAdj(params, point)
        } -> ConvertibleTo<
            Eigen::Matrix<typename T::Scalar, T::kPointDim, T::kDof>>;
    };

// TODO: harmonize with the implementation in geometry.

template <class TScalar>
class Rotation2Impl {
 public:
  using Scalar = TScalar;
  static int const kDof = 1;
  static int const kNumParams = 2;
  static int const kPointDim = 2;
  static int const kAmbientDim = 2;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return Eigen::Vector<Scalar, 2>(1.0, 0.0);
  }

  static auto areParamsValid(
      Eigen::Vector<Scalar, kNumParams> const& unit_complex)
      -> sophus::Expected<Success> {
    static const Scalar kThr = kEpsilon<Scalar>;
    const Scalar squared_norm = unit_complex.squaredNorm();
    using std::abs;
    if (!(abs(squared_norm - 1.0) <= kThr)) {
      return FARM_UNEXPECTED(
          "complex number ({}, {}) is not unit length.\n"
          "Squared norm: {}, thr: {}",
          unit_complex[0],
          unit_complex[1],
          squared_norm,
          kThr);
    }
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& angle)
      -> Eigen::Vector<Scalar, kNumParams> {
    using std::cos;
    using std::sin;
    return Eigen::Vector<Scalar, 2>(cos(angle[0]), sin(angle[0]));
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& unit_complex)
      -> Eigen::Vector<Scalar, kDof> {
    using std::atan2;
    return Eigen::Vector<Scalar, 1>{atan2(unit_complex.y(), unit_complex.x())};
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& angle)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return Eigen::Matrix<Scalar, 2, 2>{{0, -angle[0]}, {angle[0], 0}};
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return Eigen::Matrix<Scalar, kDof, 1>{mat(1, 0)};
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, 1, 1>::Identity();
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const&,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, 2, 1>(point[1], -point[0]);
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
    Scalar const lhs_real = lhs_params.x();
    Scalar const lhs_imag = lhs_params.y();
    Scalar const rhs_real = rhs_params.x();
    Scalar const rhs_imag = rhs_params.y();

    // complex multiplication
    Scalar const result_real = lhs_real * rhs_real - lhs_imag * rhs_imag;
    Scalar const result_imag = lhs_real * rhs_imag + lhs_imag * rhs_real;

    Scalar const squared_norm =
        result_real * result_real + result_imag * result_imag;
    // We can assume that the squared-norm is close to 1 since we deal with a
    // unit complex number. Due to numerical precision issues, there might
    // be a small drift after pose concatenation. Hence, we need to renormalizes
    // the complex number here.
    // Since squared-norm is close to 1, we do not need to calculate the costly
    // square-root, but can use an approximation around 1 (see
    // http://stackoverflow.com/a/12934750 for details).
    if (squared_norm != 1.0) {
      Scalar const scale = 2.0 / (1.0 + squared_norm);
      return Eigen::Vector<Scalar, kNumParams>(
          result_real * scale, result_imag * scale);
    }
    return Eigen::Vector<Scalar, kNumParams>(result_real, result_imag);
  }

  // Point actions
  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& unit_complex,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    Eigen::Vector<Scalar, kPointDim> out;
    Scalar const lhs_real = unit_complex.x();
    Scalar const lhs_imag = unit_complex.y();
    Scalar const rhs_real = point.x();
    Scalar const rhs_imag = point.y();

    // complex multiplication
    out[0] = lhs_real * rhs_real - lhs_imag * rhs_imag;
    out[1] = lhs_real * rhs_imag + lhs_imag * rhs_real;

    return out;
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

  // left Jacobian

  static auto leftJacobian(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  static auto leftJacobianInverse(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
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

  // left Jacobian

  static auto leftJacobian(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  static auto leftJacobianInverse(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return Eigen::Matrix<Scalar, kDof, kDof>::Identity();
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const&,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    return Eigen::Matrix<Scalar, 2, 2>::Identity();
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

template <
    class TScalar,
    template <class>
    class TLeftGroup,
    template <class>
    class TRightGroup>
  requires LieGroupImplConcept<TLeftGroup<TScalar>> &&
           LieGroupImplConcept<TRightGroup<TScalar>>
class DirectProduct {
 public:
  using Scalar = TScalar;
  using LeftGroup = TLeftGroup<Scalar>;
  using RightGroup = TRightGroup<Scalar>;

  static int const kDof = LeftGroup::kDof + RightGroup::kDof;
  static int const kNumParams = LeftGroup::kNumParams + RightGroup::kNumParams;
  static int const kPointDim = LeftGroup::kPointDim;
  static_assert(kPointDim == RightGroup::kPointDim);
  static int const kAmbientDim = LeftGroup::kAmbientDim;
  static_assert(kAmbientDim == RightGroup::kAmbientDim);
  static_assert(kPointDim == kAmbientDim);

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return params(LeftGroup::identityParams(), RightGroup::identityParams());
  }

  static auto areParamsValid(Eigen::Vector<Scalar, kNumParams> const& params)
      -> sophus::Expected<Success> {
    farm_ng::Error error;

    auto are_left_params_valid = LeftGroup::areParamsValid(leftParams(params));
    auto are_right_params_valid =
        RightGroup::areParamsValid(rightParams(params));

    if (!are_left_params_valid) {
      error.details.insert(
          error.details.end(),
          are_left_params_valid.error().details.begin(),
          are_left_params_valid.error().details.end());
    }
    if (!are_right_params_valid) {
      error.details.insert(
          error.details.end(),
          are_right_params_valid.error().details.begin(),
          are_right_params_valid.error().details.end());
    }

    if (!error.details.empty()) {
      return tl::unexpected(error);
    }
    return sophus::Expected<Success>{};
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> tangent)
      -> Eigen::Vector<Scalar, kNumParams> {
    return params(
        LeftGroup::exp(leftTangent(tangent)),
        RightGroup::exp(rightTangent(tangent)));
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kDof> {
    return tangent(
        LeftGroup::log(leftParams(params)),
        RightGroup::log(rightParams(params)));
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& scale_factors)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return LeftGroup::hat(leftTangent(scale_factors)) +
           RightGroup::hat(rightTangent(scale_factors));
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return tangent(LeftGroup::vee(mat), RightGroup::vee(mat));
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat_adj;
    mat_adj.setZero();
    mat_adj.template topLeftCorner<LeftGroup::kDof, LeftGroup::kDof>() =
        LeftGroup::adj(leftParams(params));
    mat_adj.template bottomRightCorner<RightGroup::kDof, RightGroup::kDof>() =
        RightGroup::adj(rightParams(params));
    return mat_adj;
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const&,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat_adj;
    mat_adj.setZero();
    mat_adj.template topLeftCorner<kPointDim, LeftGroup::kDof>() =
        LeftGroup::topRightAdj(leftParams(params), point);
    mat_adj.template bottomRightCorner<kPointDim, RightGroup::kDof>() =
        RightGroup::topRightAdj(rightParams(params), point);
    return mat_adj;
  }

  // group operations
  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    return params(
        LeftGroup::multiplication(
            leftParams(lhs_params), leftParams(rhs_params)),
        RightGroup::multiplication(
            rightParams(lhs_params), rightParams(rhs_params)));
  }

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, LeftGroup::kNumParams> left =
        LeftGroup::inverse(leftParams(params));
    Eigen::Vector<Scalar, LeftGroup::kNumParams> right =
        RightGroup::inverse(rightParams(params));
    return DirectProduct::params(left, right);
  }

  // Point actions

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    Eigen::Vector<Scalar, kPointDim> hop1 =
        RightGroup::action(rightParams(params), point);
    Eigen::Vector<Scalar, kPointDim> hop2 =
        LeftGroup::action(leftParams(params), hop1);
    return hop2;
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return point;
  }

  // Matrices

  static auto compactMatrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return LeftGroup::compactMatrix(leftParams(params)) *
           RightGroup::compactMatrix(rightParams(params));
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return compactMatrix(params);
  }

  // left jacobian

  static auto leftJacobian(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return LeftGroup::leftJacobian(leftParams(params)) *
           RightGroup::leftJacobian(rightParams(params));
  }

  static auto leftJacobianInverse(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return LeftGroup::leftJacobianInverse(leftParams(params)) *
           RightGroup::leftJacobianInverse(rightParams(params));
  }

  // for tests

  static auto exampleTangents() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    std::vector<Eigen::Vector<Scalar, kDof>> examples;
    for (auto const& left_tangent : LeftGroup::exampleTangents()) {
      for (auto const& right_tangents : RightGroup::exampleTangents()) {
        examples.push_back(tangent(left_tangent, right_tangents));
      }
    }
    return examples;
  }

  static auto exampleParams()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    std::vector<Eigen::Vector<Scalar, kNumParams>> examples;
    for (auto const& left_params : LeftGroup::exampleParams()) {
      for (auto const& right_params : RightGroup::exampleParams()) {
        examples.push_back(params(left_params, right_params));
      }
    }
    return examples;
  }

 private:
  static auto leftParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, LeftGroup::kNumParams> {
    return params.template head<LeftGroup::kNumParams>();
  }

  static auto rightParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, RightGroup::kNumParams> {
    return params.template tail<RightGroup::kNumParams>();
  }

  static auto params(
      Eigen::Vector<Scalar, LeftGroup::kNumParams> const& params1,
      Eigen::Vector<Scalar, RightGroup::kNumParams> const& params2)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, kNumParams> params;
    params.template head<LeftGroup::kNumParams>() = params1;
    params.template tail<RightGroup::kNumParams>() = params2;
    return params;
  }

  static auto leftTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, LeftGroup::kDof> {
    return tangent.template head<LeftGroup::kDof>();
  }

  static auto rightTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, RightGroup::kDof> {
    return tangent.template tail<RightGroup::kDof>();
  }

  static auto tangent(
      Eigen::Vector<Scalar, LeftGroup::kDof> const& tangent1,
      Eigen::Vector<Scalar, RightGroup::kDof> const& tangent2)
      -> Eigen::Vector<Scalar, kDof> {
    Eigen::Vector<Scalar, kDof> tangent;
    tangent.template head<LeftGroup::kDof>() = tangent1;
    tangent.template tail<RightGroup::kDof>() = tangent2;
    return tangent;
  }
};

template <class TScalar, int kTranslationDim, template <class> class TLeftGroup>
  requires LeftJacobianImplConcept<TLeftGroup<TScalar>>
class SemiDirectProductWithTranslation {
 public:
  using Scalar = TScalar;
  using LeftGroup = TLeftGroup<Scalar>;

  // The is also the dimension of the translation.
  static int const kPointDim = kTranslationDim;
  static_assert(kPointDim == LeftGroup::kPointDim);
  static_assert(kPointDim == LeftGroup::kAmbientDim);

  static int const kDof = LeftGroup::kDof + kPointDim;
  static int const kNumParams = LeftGroup::kNumParams + kPointDim;
  static int const kAmbientDim = kPointDim + 1;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return params(
        LeftGroup::identityParams(), Eigen::Vector<Scalar, kPointDim>::Zero());
  }

  static auto areParamsValid(Eigen::Vector<Scalar, kNumParams> const& params)
      -> sophus::Expected<Success> {
    return LeftGroup::areParamsValid(leftParams(params));
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> tangent)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, LeftGroup::kNumParams> left_params =
        LeftGroup::exp(leftTangent(tangent));
    return params(
        left_params,
        (LeftGroup::leftJacobian(left_params) * translationTangent(tangent)).eval());
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kDof> {
    Eigen::Vector<Scalar, LeftGroup::kNumParams> left_params =
        leftParams(params);
    return tangent(
        LeftGroup::log(left_params),
        LeftGroup::leftJacobianInverse(left_params) * translation(params));
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> hat_mat;
    hat_mat.setZero();
    hat_mat.template topLeftCorner<kPointDim, kPointDim>() =
        LeftGroup::hat(leftTangent(tangent));
    hat_mat.template topRightCorner<kPointDim, 1>() =
        translationTangent(tangent);
    return hat_mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return tangent(
        LeftGroup::vee(
            mat.template topLeftCorner<kPointDim, kPointDim>().eval()),
        mat.template topRightCorner<kPointDim, 1>().eval());
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat_adjoint;

    Eigen::Vector<Scalar, LeftGroup::kNumParams> left_params =
        leftParams(params);

    mat_adjoint.template topLeftCorner<kPointDim, kPointDim>() =
        LeftGroup::matrix(left_params);
    mat_adjoint.template topRightCorner<kPointDim, LeftGroup::kDof>() =
        LeftGroup::topRightAdj(left_params, translation(params));

    mat_adjoint.template bottomLeftCorner<LeftGroup::kDof, LeftGroup::kDof>()
        .setZero();
    mat_adjoint.template bottomRightCorner<LeftGroup::kDof, LeftGroup::kDof>() =
        LeftGroup::adj(left_params);

    return mat_adjoint;
  }

  // group operations
  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, LeftGroup::kNumParams> left_params =
        LeftGroup::multiplication(
            leftParams(lhs_params), leftParams(rhs_params));
    return SemiDirectProductWithTranslation::params(
        left_params, LeftGroup::action(left_params, translation(params)));
  }

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, LeftGroup::kNumParams> left_params =
        LeftGroup::inverse(leftParams(params));
    return SemiDirectProductWithTranslation::params(
        left_params,
        (-LeftGroup::action(left_params, translation(params))).eval());
  }

  // Point actions

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return LeftGroup::action(leftParams(params), point) + translation(params);
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kAmbientDim> {
    return unproj(point);
  }

  // Matrices

  static auto compactMatrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kPointDim, kAmbientDim> mat;
    mat.template topLeftCorner<kPointDim, kPointDim>() =
        LeftGroup::compactMatrix(leftParams(params));
    mat.template topRightCorner<kPointDim, 1>() = translation(params);
    return mat;
  }

  static auto matrix(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> mat;
    mat.setZero();
    mat.template topLeftCorner<kPointDim, kAmbientDim>() =
        compactMatrix(params);
    mat(kPointDim, kPointDim) = 1.0;
    return mat;
  }

  // for tests

  static auto exampleTangents() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    std::vector<Eigen::Vector<Scalar, kDof>> examples;
    for (auto const& left_tangent : LeftGroup::exampleTangents()) {
      for (auto const& translation_tangents : exampleTranslations()) {
        examples.push_back(tangent(left_tangent, translation_tangents));
      }
    }
    return examples;
  }

  static auto exampleParams()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    std::vector<Eigen::Vector<Scalar, kNumParams>> examples;
    for (auto const& left_params : LeftGroup::exampleParams()) {
      for (auto const& right_params : exampleTranslations()) {
        examples.push_back(params(left_params, right_params));
      }
    }
    return examples;
  }

 private:
  static auto leftParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, LeftGroup::kNumParams> {
    return params.template head<LeftGroup::kNumParams>();
  }

  static auto translation(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kPointDim> {
    return params.template tail<kPointDim>();
  }

  static auto params(
      Eigen::Vector<Scalar, LeftGroup::kNumParams> const& params1,
      Eigen::Vector<Scalar, kPointDim> const& translation)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, kNumParams> params;
    params.template head<LeftGroup::kNumParams>() = params1;
    params.template tail<kPointDim>() = translation;
    return params;
  }

  static auto leftTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, LeftGroup::kDof> {
    return tangent.template head<LeftGroup::kDof>();
  }

  static auto translationTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, kPointDim> {
    return tangent.template tail<kPointDim>();
  }

  static auto tangent(
      Eigen::Vector<Scalar, LeftGroup::kDof> const& tangent1,
      Eigen::Vector<Scalar, kPointDim> const& translation)
      -> Eigen::Vector<Scalar, kDof> {
    Eigen::Vector<Scalar, kDof> tangent;
    tangent.template head<LeftGroup::kDof>() = tangent1;
    tangent.template tail<kPointDim>() = translation;
    return tangent;
  }

  static auto exampleTranslations()
      -> std::vector<Eigen::Vector<Scalar, kPointDim>> {
    std::vector<Eigen::Vector<Scalar, kPointDim>> examples;

    if constexpr (kPointDim == 2) {
      examples.push_back(Eigen::Vector<Scalar, 2>::Zero());
      examples.push_back(Eigen::Vector<Scalar, 2>::Ones());

    } else {
      if constexpr (kPointDim == 3) {
        examples.push_back(Eigen::Vector<Scalar, 3>::Zero());
        examples.push_back(Eigen::Vector<Scalar, 3>::Ones());
      }
    }
    return examples;
  }
};

static_assert(LeftJacobianImplConcept<Rotation2Impl<double>>);
static_assert(
    LeftJacobianImplConcept<DirectProduct<float, Scaling2Impl, Rotation2Impl>>);
static_assert(LieGroupImplConcept<
              SemiDirectProductWithTranslation<float, 2, Rotation2Impl>>);

template <LieGroupImplConcept TImpl>
class Group {
 public:
  using Scalar = typename TImpl::Scalar;
  static int constexpr kDof = TImpl::kDof;
  static int constexpr kNumParams = TImpl::kNumParams;
  static int constexpr kPointDim = TImpl::kPointDim;
  static int constexpr kAmbientDim = TImpl::kAmbientDim;

  // constructors and factories

  Group() : params_(TImpl::identityParams()) {}

  Group(Group const&) = default;
  Group& operator=(Group const&) = default;

  static auto fromParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Group {
    Group g(UninitTag{});
    g.setParams(params);
    return g;
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& tangent) -> Group {
    return Group(TImpl::exp(tangent));
  }

  auto log() const -> Eigen::Vector<Scalar, kDof> {
    return TImpl::log(this->params_);
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return TImpl::hat(tangent);
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Vector<Scalar, kDof> {
    return TImpl::vee(mat);
  }

  auto adj() const -> Eigen::Matrix<Scalar, kDof, kDof> {
    return TImpl::adj(this->params_);
  }

  // group operations

  auto operator*(Group const& rhs) const -> Group {
    return Group(TImpl::multiplication(this->params_, rhs.params_));
  }

  auto inverse() const -> Group {
    return Group::fromParams(TImpl::inverse(this->params_));
  }

  // Point actions

  auto operator*(Eigen::Vector<Scalar, kPointDim> const& point) const
      -> Eigen::Vector<Scalar, kPointDim> {
    return TImpl::action(this->params_, point);
  }

  static auto toAmbient(Eigen::Vector<Scalar, kPointDim> const& point) {
    return TImpl::toAmbient(point);
  }

  // Matrices

  auto compactMatrix() const -> Eigen::Matrix<Scalar, kPointDim, kAmbientDim> {
    return TImpl::compactMatrix(this->params_);
  }

  auto matrix() const -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    return TImpl::matrix(this->params_);
  }

  // for tests

  static auto exampleTangents() -> std::vector<Eigen::Vector<Scalar, kDof>> {
    return TImpl::exampleTangents();
  }

  static auto exampleParams()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    return TImpl::exampleParams();
  }

  // getters and setters

  Eigen::Vector<Scalar, kNumParams> const& params() const {
    return this->params_;
  }

  void setParams(Eigen::Vector<Scalar, kNumParams> const& params) {
    // Hack to get unexpected error message on failure.
    auto maybe_valid = TImpl::areParamsValid(params);
    SOPHUS_UNWRAP(maybe_valid);
    this->params_ = params;
  }

 protected:
  explicit Group(UninitTag /*unused*/) {}

  Group(Eigen::Vector<Scalar, kNumParams> const& params) : params_(params) {}

  Eigen::Vector<Scalar, kNumParams> params_;
};

template <LeftJacobianImplConcept TImpl>
class GroupWithLeftJacobian : public Group<TImpl> {
 public:
  using Scalar = typename TImpl::Scalar;
  static int constexpr kDof = TImpl::kDof;
  static int constexpr kNumParams = TImpl::kNumParams;
  static int constexpr kPointDim = TImpl::kPointDim;
  static int constexpr kAmbientDim = TImpl::kAmbientDim;

  // constructors and factories

  GroupWithLeftJacobian() : Group<TImpl>() {}
  GroupWithLeftJacobian(GroupWithLeftJacobian const&) = default;
  GroupWithLeftJacobian& operator=(GroupWithLeftJacobian const&) = default;

  GroupWithLeftJacobian(Group<TImpl>&& base) : Group<TImpl>(base) {}

  static auto fromParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> GroupWithLeftJacobian {
    return GroupWithLeftJacobian(Group<TImpl>::fromParams(params));
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> const& tangent)
      -> GroupWithLeftJacobian {
    return GroupWithLeftJacobian(Group<TImpl>::exp(tangent));
  }

  // group operations

  auto operator*(GroupWithLeftJacobian const& rhs) const
      -> GroupWithLeftJacobian {
    return GroupWithLeftJacobian(this->Group<TImpl>::operator*(rhs));
  }

  auto inverse() const -> GroupWithLeftJacobian {
    return GroupWithLeftJacobian(this->Group<TImpl>::inverse());
  }

  // Point actions

  // needed so the operator* for group multiplication does not hide the point
  // action multiplication from the base.
  using Group<TImpl>::operator*;

  // left Jacobian

  auto leftJacobian() const -> Eigen::Matrix<Scalar, kDof, kDof> {
    return TImpl::leftJacobian(this->params_);
  }

  auto leftJacobianInverse(Eigen::Vector<Scalar, kNumParams> const&) const
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    return TImpl::leftJacobianInverse(this->params_);
  }
};

}  // namespace lie

template <class Scalar>
using Rotation2 /*aka SO(2) */ =
    lie::GroupWithLeftJacobian<lie::Rotation2Impl<Scalar>>;

template <class Scalar>
using Scaling2 = lie::GroupWithLeftJacobian<lie::Scaling2Impl<Scalar>>;

template <class Scalar>
using ScalingRotation2 = lie::GroupWithLeftJacobian<
    lie::DirectProduct<Scalar, lie::Scaling2Impl, lie::Rotation2Impl>>;

template <class Scalar>
using Isometry2 /*aka SE(2) */ = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 2, lie::Rotation2Impl>>;

template <class Scalar>
using ScalingTranslation2 = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 2, lie::Scaling2Impl>>;

// using SpiralSimilarity; // UniformScaling && Rotation
// using Similarity; // SpiralSimilarity && Translation
// using Dilation; // UniformScaling && Translation

}  // namespace sophus
