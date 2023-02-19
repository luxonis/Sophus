// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/experimental/lie_group_concept.h"

#include <gtest/gtest.h>

using namespace sophus;

template <class G>
void expLogTest(std::string group_name) {
  for (auto const& params : G::exampleParams()) {
    auto g = G::fromParams(params);
    auto matrix_before = g.matrix();
    auto matrix_after = G::exp(g.log()).matrix();

    FARM_ASSERT_NEAR(
        matrix_before, matrix_after, 0.001, "\expLogTest: {}\n", group_name);
  }
}

template <class G>
void mapPointTest(
    std::string group_name,
    std::vector<Eigen::Vector<typename G::Scalar, G::kPointDim>> const&
        point_vec) {
  for (size_t point_id = 0; point_id < point_vec.size(); ++point_id) {
    auto point_in = point_vec[point_id];
    for (size_t params_id = 0; params_id < G::exampleParams().size();
         ++params_id) {
      auto params = G::exampleParams()[params_id];
      auto g = G::fromParams(params);
      Eigen::Vector<typename G::Scalar, G::kPointDim> out_point_from_matrix =
          g.matrix() * G::toAmbient(point_in);
      Eigen::Vector<typename G::Scalar, G::kPointDim> out_point_from_action =
          g * point_in;

      FARM_ASSERT_NEAR(
          out_point_from_matrix,
          out_point_from_action,
          0.001,
          "\nmapPointTest: {}\n"
          "point # {} ({})\n"
          "params # {} ({}); matrix:\n"
          "{}",
          group_name,
          point_id,
          point_in.transpose(),
          params_id,
          params.transpose(),
          g.matrix());
    }
  }
}

template <class G>
void tests(
    std::string group_name,
    std::vector<Eigen::Vector<typename G::Scalar, G::kPointDim>> const&
        point_vec) {
  expLogTest<G>(group_name);
  mapPointTest<G>(group_name, point_vec);
}

template <class Scalar>
void testAllGroups2() {
  std::vector<Eigen::Vector<Scalar, 2>> point_vec;
  point_vec.push_back(Eigen::Vector<Scalar, 2>(Scalar(1), Scalar(2)));
  point_vec.push_back(Eigen::Vector<Scalar, 2>(Scalar(1), Scalar(-3)));

  tests<sophus::Rotation2<Scalar>>("Rotation(2)", point_vec);
  tests<sophus::Scaling2<Scalar>>("Scaling(2)", point_vec);
  tests<sophus::ScalingRotation2<Scalar>>("ScalingRotation(2)", point_vec);
  tests<sophus::Isometry2<Scalar>>("Isometry(2)", point_vec);
  tests<sophus::ScalingTranslation2<Scalar>>("ScalingTranslation", point_vec);
}

TEST(lie_group_concept, unit) { testAllGroups2<double>(); }
