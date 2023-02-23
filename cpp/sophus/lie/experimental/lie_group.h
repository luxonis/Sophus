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

#include "sophus/lie/experimental/impl/identity.h"
#include "sophus/lie/experimental/impl/rotation2.h"
#include "sophus/lie/experimental/impl/rotation3.h"
#include "sophus/lie/experimental/impl/scaling.h"
#include "sophus/lie/experimental/impl/semi_direct_product.h"
#include "sophus/lie/experimental/lie_group_concept.h"

namespace sophus {
namespace lie {

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

}  // namespace lie

template <class Scalar>
using Identity2 = lie::Group<lie::Identity2Impl<Scalar>>;
template <class Scalar>
using Identity3 = lie::Group<lie::Identity3Impl<Scalar>>;

template <class Scalar>
using Rotation2 /*aka SO(2) */ = lie::Group<lie::Rotation2Impl<Scalar>>;
template <class Scalar>
using Rotation3 /*aka SO(3) */ = lie::Group<lie::Rotation3Impl<Scalar>>;

template <class Scalar>
using Scaling2 = lie::Group<lie::ScalingImpl<Scalar, 2>>;
template <class Scalar>
using Scaling3 = lie::Group<lie::ScalingImpl<Scalar, 3>>;

template <class Scalar>
using Isometry2 /*aka SE(2) */ = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 2, lie::Rotation2Impl>>;
template <class Scalar>
using Isometry3 /*aka SE(3) */ = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 3, lie::Rotation3Impl>>;

template <class Scalar>
using Translation2 = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 2, lie::Identity2Impl>>;
template <class Scalar>
using Translation3 = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 3, lie::Identity3Impl>>;

template <class Scalar>
using ScalingTranslation2 = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 2, lie::Scaling2Impl>>;
template <class Scalar>
using ScalingTranslation3 = lie::Group<
    lie::SemiDirectProductWithTranslation<Scalar, 3, lie::Scaling3Impl>>;

// using SpiralSimilarity; // UniformScaling && Rotation
// using Similarity; // SpiralSimilarity && Translation
// using Dilation; // UniformScaling && Translation

}  // namespace sophus
