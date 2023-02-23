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
    FARM_WARN("hop:\n{}\n", mat_adj);

    mat_adj.template bottomRightCorner<RightGroup::kDof, RightGroup::kDof>() =
        RightGroup::adj(rightParams(params));
    FARM_WARN("hop:\n{}\n", mat_adj);

    return mat_adj;
  }

  static auto topRightAdj(
      Eigen::Vector<Scalar, kNumParams> const&,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Matrix<Scalar, kPointDim, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat_adj;
    auto adj_from_left = LeftGroup::topRightAdj(leftParams(params), point);
    auto adj_from_right = RightGroup::topRightAdj(rightParams(params), point);

    FARM_WARN(
        "adj_from_left:\n{}\n"
        "adj_from_right:\n{}\n",
        adj_from_left,
        adj_from_right);

    mat_adj.setZero();
    mat_adj.template topLeftCorner<kPointDim, LeftGroup::kDof>() =
        adj_from_left;
    FARM_WARN("hop:\n{}\n", mat_adj);
    mat_adj.template bottomRightCorner<kPointDim, RightGroup::kDof>() =
        adj_from_right;
    FARM_WARN("hop:\n{}\n", mat_adj);
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

  // subgroup concepts

  static auto matV(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return LeftGroup::matV(leftParams(params)) *
           RightGroup::matV(rightParams(params));
  }

  static auto matVInverse(Eigen::Vector<Scalar, kNumParams> const&)
      -> Eigen::Matrix<Scalar, kPointDim, kPointDim> {
    return LeftGroup::matVInverse(leftParams(params)) *
           RightGroup::matVInverse(rightParams(params));
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

}  // namespace lie
}  // namespace sophus
