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
#include "sophus/linalg/homogeneous.h"

namespace sophus {
namespace lie {

template <class TScalar, int kTranslationDim, template <class> class TSubGroup>
requires LieSubgroupImplConcept<TSubGroup<TScalar>>
class SemiDirectProductWithTranslation {
 public:
  using Scalar = TScalar;
  using SubGroup = TSubGroup<Scalar>;

  // The is also the dimension of the translation.
  static int const kPointDim = kTranslationDim;
  static_assert(kPointDim == SubGroup::kPointDim);
  static_assert(kPointDim == SubGroup::kAmbientDim);

  static int const kDof = SubGroup::kDof + kPointDim;
  static int const kNumParams = SubGroup::kNumParams + kPointDim;
  static int const kAmbientDim = kPointDim + 1;

  // constructors and factories

  static auto identityParams() -> Eigen::Vector<Scalar, kNumParams> {
    return params(
        SubGroup::identityParams(), Eigen::Vector<Scalar, kPointDim>::Zero());
  }

  static auto areParamsValid(Eigen::Vector<Scalar, kNumParams> const& params)
      -> sophus::Expected<Success> {
    return SubGroup::areParamsValid(subGroupParams(params));
  }

  // Manifold / Lie Group concepts

  static auto exp(Eigen::Vector<Scalar, kDof> tangent)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        SubGroup::exp(subgroupTangent(tangent));
    return params(
        subgroup_params,
        (SubGroup::matV(subgroup_params) * translationTangent(tangent)).eval());
  }

  static auto log(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kDof> {
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        subGroupParams(params);
    return tangent(
        SubGroup::matVInverse(subgroup_params) * translation(params),
        SubGroup::log(subgroup_params));
  }

  static auto hat(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> {
    Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> hat_mat;
    hat_mat.setZero();
    hat_mat.template topLeftCorner<kPointDim, kPointDim>() =
        SubGroup::hat(subgroupTangent(tangent));
    hat_mat.template topRightCorner<kPointDim, 1>() =
        translationTangent(tangent);
    return hat_mat;
  }

  static auto vee(Eigen::Matrix<Scalar, kAmbientDim, kAmbientDim> const& mat)
      -> Eigen::Matrix<Scalar, kDof, 1> {
    return tangent(
        mat.template topRightCorner<kPointDim, 1>().eval(),
        SubGroup::vee(
            mat.template topLeftCorner<kPointDim, kPointDim>().eval()));
  }

  static auto adj(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Matrix<Scalar, kDof, kDof> {
    Eigen::Matrix<Scalar, kDof, kDof> mat_adjoint;

    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        subGroupParams(params);

    mat_adjoint.template topLeftCorner<kPointDim, kPointDim>() =
        SubGroup::matrix(subgroup_params);
    mat_adjoint.template topRightCorner<kPointDim, SubGroup::kDof>() =
        SubGroup::topRightAdj(subgroup_params, translation(params));

    mat_adjoint.template bottomLeftCorner<SubGroup::kDof, kPointDim>()
        .setZero();
    mat_adjoint.template bottomRightCorner<SubGroup::kDof, SubGroup::kDof>() =
        SubGroup::adj(subgroup_params);

    return mat_adjoint;
  }

  // group operations
  static auto multiplication(
      Eigen::Vector<Scalar, kNumParams> const& lhs_params,
      Eigen::Vector<Scalar, kNumParams> const& rhs_params)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        SubGroup::multiplication(
            subGroupParams(lhs_params), subGroupParams(rhs_params));
    return SemiDirectProductWithTranslation::params(
        subgroup_params,
        SubGroup::action(subgroup_params, translation(params)));
  }

  static auto inverse(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, SubGroup::kNumParams> subgroup_params =
        SubGroup::inverse(subGroupParams(params));
    return SemiDirectProductWithTranslation::params(
        subgroup_params,
        (-SubGroup::action(subgroup_params, translation(params))).eval());
  }

  // Point actions

  static auto action(
      Eigen::Vector<Scalar, kNumParams> const& params,
      Eigen::Vector<Scalar, kPointDim> const& point)
      -> Eigen::Vector<Scalar, kPointDim> {
    return SubGroup::action(subGroupParams(params), point) +
           translation(params);
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
        SubGroup::compactMatrix(subGroupParams(params));
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
    for (auto const& group_tangent : SubGroup::exampleTangents()) {
      for (auto const& translation_tangents : exampleTranslations()) {
        examples.push_back(tangent(translation_tangents, group_tangent));
      }
    }
    return examples;
  }

  static auto exampleParams()
      -> std::vector<Eigen::Vector<Scalar, kNumParams>> {
    std::vector<Eigen::Vector<Scalar, kNumParams>> examples;
    for (auto const& subgroup_params : SubGroup::exampleParams()) {
      for (auto const& right_params : exampleTranslations()) {
        examples.push_back(params(subgroup_params, right_params));
      }
    }
    return examples;
  }

 private:
  static auto subGroupParams(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, SubGroup::kNumParams> {
    return params.template head<SubGroup::kNumParams>();
  }

  static auto translation(Eigen::Vector<Scalar, kNumParams> const& params)
      -> Eigen::Vector<Scalar, kPointDim> {
    return params.template tail<kPointDim>();
  }

  static auto params(
      Eigen::Vector<Scalar, SubGroup::kNumParams> const& sub_group_params,
      Eigen::Vector<Scalar, kPointDim> const& translation)
      -> Eigen::Vector<Scalar, kNumParams> {
    Eigen::Vector<Scalar, kNumParams> params;
    params.template head<SubGroup::kNumParams>() = sub_group_params;
    params.template tail<kPointDim>() = translation;
    return params;
  }

  static auto subgroupTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, SubGroup::kDof> {
    return tangent.template tail<SubGroup::kDof>();
  }

  static auto translationTangent(Eigen::Vector<Scalar, kDof> const& tangent)
      -> Eigen::Vector<Scalar, kPointDim> {
    return tangent.template head<kPointDim>();
  }

  static auto tangent(
      Eigen::Vector<Scalar, kPointDim> const& translation,
      Eigen::Vector<Scalar, SubGroup::kDof> const& subgroup_tangent)
      -> Eigen::Vector<Scalar, kDof> {
    Eigen::Vector<Scalar, kDof> tangent;
    tangent.template head<kPointDim>() = translation;
    tangent.template tail<SubGroup::kDof>() = subgroup_tangent;
    return tangent;
  }

  static auto exampleTranslations()
      -> std::vector<Eigen::Vector<Scalar, kPointDim>> {
    std::vector<Eigen::Vector<Scalar, kPointDim>> examples;

    if constexpr (kPointDim == 2) {
      examples.push_back(Eigen::Vector<Scalar, 2>::Zero());
      examples.push_back(Eigen::Vector<Scalar, 2>(0.2, 1.0));

    } else {
      if constexpr (kPointDim == 3) {
        examples.push_back(Eigen::Vector<Scalar, 3>::Zero());
        examples.push_back(Eigen::Vector<Scalar, 3>(0.2, 1.0, 0.0));
      }
    }
    return examples;
  }
};

}  // namespace lie
}  // namespace sophus
