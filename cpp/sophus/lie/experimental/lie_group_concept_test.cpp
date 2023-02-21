// Copyright (c) 2011, Hauke Strasdat
// Copyright (c) 2012, Steven Lovegrove
// Copyright (c) 2021, farm-ng, inc.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "sophus/lie/experimental/lie_group_concept.h"

#include <gtest/gtest.h>
#include <sophus/calculus/num_diff.h>

using namespace sophus;

// adjointTest
// template <class G>
// bool adjointTest() {
//   bool passed = true;
//   for (size_t i = 0; i < group_vec_.size(); ++i) {
//     Transformation t = group_vec_[i].matrix();
//     Adjoint ad = group_vec_[i].adj();
//     for (size_t j = 0; j < tangent_vec_.size(); ++j) {
//       Tangent x = tangent_vec_[j];

//       Transformation mat_i;
//       mat_i.setIdentity();
//       Tangent ad1 = ad * x;
//       Tangent ad2 = LieGroup::vee(
//           t * LieGroup::hat(x) * group_vec_[i].inverse().matrix());
//       SOPHUS_TEST_APPROX(
//           passed,
//           ad1,
//           ad2,
//           Scalar(10) * small_eps,
//           "Adjoint case %, %",
//           mat_i,
//           j);
//     }
//   }
//   return passed;
// }

// // implemented for So3 and Se3
// template <class G>
// void leftJacobianTest(std::string group_name) {
//   using Scalar = typename G::Scalar;
//   Scalar const small_eps_sqrt = kEpsilonSqrt<Scalar>;

//   for (size_t tangent_id = 0; tangent_id < G::exampleTangents().size();
//        ++tangent_id) {
//     auto x = G::exampleTangents()[tangent_id];
//     G const inv_exp_x = G::exp(x).inverse();

//     // Explicit implement the derivative in the Lie Group in first principles
//     // as a vector field: D_x f(x) = D_h log(f(x + h) . f(x)^{-1})
//     Eigen::Matrix<Scalar, G::kDof, G::kDof> const j_num =
//         vectorFieldNumDiff<Scalar, G::kDof, G::kDof>(
//             [&inv_exp_x](auto const& x_plus_delta) {
//               return (G::exp(x_plus_delta) * inv_exp_x).log();
//             },
//             x);

//     // Analytical left Jacobian
//     Eigen::Matrix<Scalar, G::kDof, G::kDof> const j = G::leftJacobian(x);
//     FARM_ASSERT_NEAR(
//         j,
//         j_num,
//         Scalar(100) * small_eps_sqrt,
//         "leftJacobianTest #1: {}",
//         group_name);

//     Eigen::Matrix<Scalar, G::kDof, G::kDof> j_inv =
//     G::leftJacobianInverse(x);

//     FARM_ASSERT_NEAR(
//         j,
//         j_inv.inverse().eval(),
//         Scalar(100) * small_eps_sqrt,
//         "leftJacobianTest #2: {}",
//         group_name);
//   }
// }

//  bool moreJacobiansTest() {
//     bool passed = true;
//     for (auto const& point : point_vec_) {
//       Eigen::Matrix<Scalar, kPointDim, kDoF> j =
//           LieGroup::dxExpXTimesPointAt0(point);
//       Tangent t;
//       setToZero(t);
//       Eigen::Matrix<Scalar, kPointDim, kDoF> const j_num =
//           vectorFieldNumDiff<Scalar, kPointDim, kDoF>(
//               [point](Tangent const& x) { return LieGroup::exp(x) * point; },
//               t);

//       SOPHUS_TEST_APPROX(
//           passed, j, j_num, small_eps_sqrt, "Dx_exp_x_times_point_at_0");
//     }
//     return passed;
//   }

//   bool contructorAndAssignmentTest() {
//     bool passed = true;
//     for (LieGroup foo_transform_bar : group_vec_) {
//       LieGroup foo_t2_bar = foo_transform_bar;
//       SOPHUS_TEST_APPROX(
//           passed,
//           foo_transform_bar.matrix(),
//           foo_t2_bar.matrix(),
//           small_eps,
//           "Copy constructor: %\nvs\n %",
//           transpose(foo_transform_bar.matrix()),
//           transpose(foo_t2_bar.matrix()));
//       LieGroup foo_t3_bar;
//       foo_t3_bar = foo_transform_bar;
//       SOPHUS_TEST_APPROX(
//           passed,
//           foo_transform_bar.matrix(),
//           foo_t3_bar.matrix(),
//           small_eps,
//           "Copy assignment: %\nvs\n %",
//           transpose(foo_transform_bar.matrix()),
//           transpose(foo_t3_bar.matrix()));

//       LieGroup foo_t4_bar(foo_transform_bar.matrix());
//       SOPHUS_TEST_APPROX(
//           passed,
//           foo_transform_bar.matrix(),
//           foo_t4_bar.matrix(),
//           small_eps,
//           "Constructor from homogeneous matrix: %\nvs\n %",
//           transpose(foo_transform_bar.matrix()),
//           transpose(foo_t4_bar.matrix()));

//       Eigen::Map<LieGroup> foo_tmap_bar(foo_transform_bar.data());
//       LieGroup foo_t5_bar = foo_tmap_bar;
//       SOPHUS_TEST_APPROX(
//           passed,
//           foo_transform_bar.matrix(),
//           foo_t5_bar.matrix(),
//           small_eps,
//           "Assignment from Eigen::Map type: %\nvs\n %",
//           transpose(foo_transform_bar.matrix()),
//           transpose(foo_t5_bar.matrix()));

//       Eigen::Map<LieGroup const> foo_tcmap_bar(foo_transform_bar.data());
//       LieGroup foo_t6_bar;
//       foo_t6_bar = foo_tcmap_bar;
//       SOPHUS_TEST_APPROX(
//           passed,
//           foo_transform_bar.matrix(),
//           foo_t5_bar.matrix(),
//           small_eps,
//           "Assignment from Eigen::Map type: %\nvs\n %",
//           transpose(foo_transform_bar.matrix()),
//           transpose(foo_t5_bar.matrix()));

//       LieGroup i;
//       Eigen::Map<LieGroup> foo_tmap2_bar(i.data());
//       foo_tmap2_bar = foo_transform_bar;
//       SOPHUS_TEST_APPROX(
//           passed,
//           foo_tmap2_bar.matrix(),
//           foo_transform_bar.matrix(),
//           small_eps,
//           "Assignment to Eigen::Map type: %\nvs\n %",
//           transpose(foo_tmap2_bar.matrix()),
//           transpose(foo_transform_bar.matrix()));
//     }
//     return passed;
//   }

//   bool derivativeTest() {
//     bool passed = true;

//     LieGroup g;
//     for (int i = 0; i < kDoF; ++i) {
//       Transformation gi = g.dxiExpmatXAt0(i);
//       Transformation gi2 = curveNumDiff(
//           [i](Scalar xi) -> Transformation {
//             Tangent x;
//             setToZero(x);
//             setElementAt(x, xi, i);
//             return LieGroup::exp(x).matrix();
//           },
//           Scalar(0));
//       SOPHUS_TEST_APPROX(
//           passed, gi, gi2, small_eps_sqrt, "Dxi_exp_x_matrix_at_ case %", i);
//     }

//     return passed;
//   }

//   template <class TG = LieGroup>
//   bool additionalDerivativeTest() {
//     bool passed = true;
//     for (size_t j = 0; j < tangent_vec_.size(); ++j) {
//       Tangent a = tangent_vec_[j];
//       Eigen::Matrix<Scalar, kNumParameters, kDoF> d = LieGroup::dxExpX(a);
//       Eigen::Matrix<Scalar, kNumParameters, kDoF> jac_num =
//           vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
//               [](Tangent const& x) -> Eigen::Vector<Scalar, kNumParameters> {
//                 return LieGroup::exp(x).params();
//               },
//               a);

//       SOPHUS_TEST_APPROX(
//           passed, d, jac_num, 3 * small_eps_sqrt, "dxExpX case: %", j);
//     }

//     Tangent o;
//     setToZero(o);
//     Eigen::Matrix<Scalar, kNumParameters, kDoF> j = LieGroup::dxExpXAt0();
//     Eigen::Matrix<Scalar, kNumParameters, kDoF> j_num =
//         vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
//             [](Tangent const& x) -> Eigen::Vector<Scalar, kNumParameters> {
//               return LieGroup::exp(x).params();
//             },
//             o);
//     SOPHUS_TEST_APPROX(passed, j, j_num, small_eps_sqrt, "Dx_exp_x_at_0");

//     for (size_t i = 0; i < group_vec_.size(); ++i) {
//       LieGroup t = group_vec_[i];

//       Eigen::Matrix<Scalar, kNumParameters, kDoF> j = t.dxThisMulExpXAt0();
//       Eigen::Matrix<Scalar, kNumParameters, kDoF> j_num =
//           vectorFieldNumDiff<Scalar, kNumParameters, kDoF>(
//               [t](Tangent const& x) -> Eigen::Vector<Scalar, kNumParameters>
//               {
//                 return (t * LieGroup::exp(x)).params();
//               },
//               o);

//       SOPHUS_TEST_APPROX(
//           passed,
//           j,
//           j_num,
//           small_eps_sqrt,
//           "Dx_this_mul_exp_x_at_0 case: %",
//           i);
//     }

//     for (size_t i = 0; i < group_vec_.size(); ++i) {
//       LieGroup t = group_vec_[i];

//       Eigen::Matrix<Scalar, kDoF, kDoF> j =
//           t.dxLogThisInvTimesXAtThis() * t.dxThisMulExpXAt0();
//       Eigen::Matrix<Scalar, kDoF, kDoF> j_exp =
//           Eigen::Matrix<Scalar, kDoF, kDoF>::Identity();

//       SOPHUS_TEST_APPROX(
//           passed,
//           j,
//           j_exp,
//           small_eps_sqrt,
//           "Dy_log_this_inv_by_at_x case: %",
//           i);
//     }
//     return passed;
//   }

// template <class G>
// void productTest(std::string group_name) {
//   bool passed = true;

//   for (size_t params_id = 1; params_id < G::exampleParams().size();
//        ++params_id) {
//     auto params1 = G::exampleParams()[params_id - 1];
//     auto params2 = G::exampleParams()[params_id];
//     auto g1 = G::fromParams(params);
//     auto g2 = G::fromParams(params);

//     G product = g1 * g2;
//     g1 *= g2;
//     FARM_ASSERT_NEAR(
//         g1.matrix(), product.matrix(), small_eps, "Product case: %", i);
//   }
//   return passed;
// }

template <class G>
void expLogTest(std::string group_name) {
  for (size_t params_id = 0; params_id < G::exampleParams().size();
       ++params_id) {
    auto params = G::exampleParams()[params_id];
    auto g = G::fromParams(params);
    auto matrix_before = g.compactMatrix();
    auto matrix_after = G::exp(g.log()).compactMatrix();

    FARM_ASSERT_NEAR(
        matrix_before,
        matrix_after,
        0.001,
        "\expLogTest: {}\n"
        "params #{}",
        group_name,
        params_id);
  }
}

// bool expMapTest() {
//   bool passed = true;
//   for (size_t i = 0; i < tangent_vec_.size(); ++i) {
//     Tangent omega = tangent_vec_[i];
//     Transformation exp_x = LieGroup::exp(omega).matrix();
//     Transformation expmap_hat_x = (LieGroup::hat(omega)).exp();
//     SOPHUS_TEST_APPROX(
//         passed,
//         exp_x,
//         expmap_hat_x,
//         Scalar(10) * small_eps,
//         "expmap(hat(x)) - exp(x) case: %",
//         i);
//   }
//   return passed;
// }

template <class G>
void groupActionTest(
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
          g.compactMatrix() * G::toAmbient(point_in);
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
          g.compactMatrix());
    }
  }
}

// bool lineActionTest() {
//     bool passed = point_vec_.size() > 1;

//     for (size_t i = 0; i < group_vec_.size(); ++i) {
//       for (size_t j = 0; j + 1 < point_vec_.size(); ++j) {
//         Point const& p1 = point_vec_[j];
//         Point const& p2 = point_vec_[j + 1];
//         Line l = Line::Through(p1, p2);
//         Point p1_t = group_vec_[i] * p1;
//         Point p2_t = group_vec_[i] * p2;
//         Line l_t = group_vec_[i] * l;

//         SOPHUS_TEST_APPROX(
//             passed,
//             l_t.squaredDistance(p1_t),
//             static_cast<Scalar>(0),
//             small_eps,
//             "Transform line case (1st point) : %",
//             i);
//         SOPHUS_TEST_APPROX(
//             passed,
//             l_t.squaredDistance(p2_t),
//             static_cast<Scalar>(0),
//             small_eps,
//             "Transform line case (2nd point) : %",
//             i);
//         SOPHUS_TEST_APPROX(
//             passed,
//             l_t.direction().squaredNorm(),
//             l.direction().squaredNorm(),
//             small_eps,
//             "Transform line case (direction) : %",
//             i);
//       }
//     }
//     return passed;
//   }

//   bool planeActionTest() {
//     int const point_dim = Point::RowsAtCompileTime;
//     bool passed = point_vec_.size() >= point_dim;
//     for (size_t i = 0; i < group_vec_.size(); ++i) {
//       for (size_t j = 0; j + point_dim - 1 < point_vec_.size(); ++j) {
//         Point points[point_dim];

//         Point points_t[point_dim];
//         for (int k = 0; k < point_dim; ++k) {
//           points[k] = point_vec_[j + k];
//           points_t[k] = group_vec_[i] * points[k];
//         }

//         Hyperplane const plane = through(points);

//         Hyperplane const plane_t = group_vec_[i] * plane;

//         for (int k = 0; k < point_dim; ++k) {
//           SOPHUS_TEST_APPROX(
//               passed,
//               plane_t.signedDistance(points_t[k]),
//               static_cast<Scalar>(0.),
//               small_eps,
//               "Transform plane case (point #%): %",
//               k,
//               i);
//         }
//         SOPHUS_TEST_APPROX(
//             passed,
//             plane_t.normal().squaredNorm(),
//             plane.normal().squaredNorm(),
//             small_eps,
//             "Transform plane case (normal): %",
//             i);
//       }
//     }
//     return passed;
//   }

//   bool lieBracketTest() {
//     bool passed = true;
//     for (size_t i = 0; i < tangent_vec_.size(); ++i) {
//       for (size_t j = 0; j < tangent_vec_.size(); ++j) {
//         Tangent tangent1 =
//             LieGroup::lieBracket(tangent_vec_[i], tangent_vec_[j]);
//         Transformation hati = LieGroup::hat(tangent_vec_[i]);
//         Transformation hatj = LieGroup::hat(tangent_vec_[j]);

//         Tangent tangent2 = LieGroup::vee(hati * hatj - hatj * hati);
//         SOPHUS_TEST_APPROX(
//             passed, tangent1, tangent2, small_eps, "Lie Bracket case: %", i);
//       }
//     }
//     return passed;
//   }

//   bool veeHatTest() {
//     bool passed = true;
//     for (size_t i = 0; i < tangent_vec_.size(); ++i) {
//       SOPHUS_TEST_APPROX(
//           passed,
//           Tangent(tangent_vec_[i]),
//           LieGroup::vee(LieGroup::hat(tangent_vec_[i])),
//           small_eps,
//           "Hat-vee case: %",
//           i);
//     }
//     return passed;
//   }

//   bool newDeleteSmokeTest() {
//     bool passed = true;
//     LieGroup* raw_ptr = nullptr;
//     raw_ptr = new LieGroup();
//     SOPHUS_TEST_NEQ(passed, reinterpret_cast<std::uintptr_t>(raw_ptr), 0,
//     ""); delete raw_ptr; return passed;
//   }

//   bool interpolateAndMeanTest() {
//     bool passed = true;
//     using std::sqrt;
//     Scalar const eps = kEpsilon<Scalar>;
//     Scalar const sqrt_eps = sqrt(eps);
//     // TODO: Improve accuracy of ``interpolate`` (and hence ``exp`` and
//     ``log``)
//     //       so that we can use more accurate bounds in these tests, i.e.
//     //       ``eps`` instead of ``sqrt_eps``.

//     for (LieGroup const& foo_transform_bar : group_vec_) {
//       for (LieGroup const& foo_transform_daz : group_vec_) {
//         // Test boundary conditions ``alpha=0`` and ``alpha=1``.
//         LieGroup foo_t_quiz =
//             interpolate(foo_transform_bar, foo_transform_daz, Scalar(0));
//         SOPHUS_TEST_APPROX(
//             passed,
//             foo_t_quiz.matrix(),
//             foo_transform_bar.matrix(),
//             sqrt_eps,
//             "");
//         foo_t_quiz =
//             interpolate(foo_transform_bar, foo_transform_daz, Scalar(1));
//         SOPHUS_TEST_APPROX(
//             passed,
//             foo_t_quiz.matrix(),
//             foo_transform_daz.matrix(),
//             sqrt_eps,
//             "");
//       }
//     }
//     for (Scalar alpha :
//          {Scalar(0.1), Scalar(0.5), Scalar(0.75), Scalar(0.99)}) {
//       for (LieGroup const& foo_transform_bar : group_vec_) {
//         for (LieGroup const& foo_transform_daz : group_vec_) {
//           LieGroup foo_t_quiz =
//               interpolate(foo_transform_bar, foo_transform_daz, alpha);
//           // test left-invariance:
//           //
//           // dash_T_foo * interp(foo_transform_bar, foo_transform_daz)
//           // == interp(dash_T_foo * foo_transform_bar, dash_T_foo *
//           // foo_transform_daz)

//           if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
//                   foo_transform_bar.inverse() * foo_transform_daz)) {
//             // skip check since there is a shortest path ambiguity
//             continue;
//           }
//           for (LieGroup const& dash_t_foo : group_vec_) {
//             LieGroup dash_t_quiz = interpolate(
//                 dash_t_foo * foo_transform_bar,
//                 dash_t_foo * foo_transform_daz,
//                 alpha);
//             SOPHUS_TEST_APPROX(
//                 passed,
//                 dash_t_quiz.matrix(),
//                 (dash_t_foo * foo_t_quiz).matrix(),
//                 sqrt_eps,
//                 "");
//           }
//           // test inverse-invariance:
//           //
//           // interp(foo_transform_bar, foo_transform_daz).inverse()
//           // == interp(foo_transform_bar.inverse(), dash_T_foo.inverse())
//           LieGroup quiz_t_foo = interpolate(
//               foo_transform_bar.inverse(), foo_transform_daz.inverse(),
//               alpha);
//           SOPHUS_TEST_APPROX(
//               passed,
//               quiz_t_foo.inverse().matrix(),
//               foo_t_quiz.matrix(),
//               sqrt_eps,
//               "");
//         }
//       }

//       for (LieGroup const& bar_transform_foo : group_vec_) {
//         for (LieGroup const& baz_transform_foo : group_vec_) {
//           LieGroup quiz_t_foo =
//               interpolate(bar_transform_foo, baz_transform_foo, alpha);
//           // test right-invariance:
//           //
//           // interp(bar_transform_foo, bar_transform_foo) * foo_T_dash
//           // == interp(bar_transform_foo * foo_T_dash, bar_transform_foo *
//           // foo_T_dash)

//           if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
//                   bar_transform_foo * baz_transform_foo.inverse())) {
//             // skip check since there is a shortest path ambiguity
//             continue;
//           }
//           for (LieGroup const& foo_t_dash : group_vec_) {
//             LieGroup quiz_t_dash = interpolate(
//                 bar_transform_foo * foo_t_dash,
//                 baz_transform_foo * foo_t_dash,
//                 alpha);
//             SOPHUS_TEST_APPROX(
//                 passed,
//                 quiz_t_dash.matrix(),
//                 (quiz_t_foo * foo_t_dash).matrix(),
//                 sqrt_eps,
//                 "");
//           }
//         }
//       }
//     }

//     for (LieGroup const& foo_transform_bar : group_vec_) {
//       for (LieGroup const& foo_transform_daz : group_vec_) {
//         if (interp_details::Traits<LieGroup>::hasShortestPathAmbiguity(
//                 foo_transform_bar.inverse() * foo_transform_daz)) {
//           // skip check since there is a shortest path ambiguity
//           continue;
//         }

//         // test average({A, B}) == interp(A, B):
//         LieGroup foo_t_quiz =
//             interpolate(foo_transform_bar, foo_transform_daz, 0.5);
//         std::optional<LieGroup> foo_t_iaverage = iterativeMean(
//             std::array<LieGroup, 2>({{foo_transform_bar,
//             foo_transform_daz}}), 20);
//         std::optional<LieGroup> foo_t_average = average(
//             std::array<LieGroup, 2>({{foo_transform_bar,
//             foo_transform_daz}}));
//         SOPHUS_TEST(
//             passed,
//             bool(foo_t_average),
//             "log(foo_transform_bar): %\nlog(foo_transform_daz): %",
//             transpose(foo_transform_bar.log()),
//             transpose(foo_transform_daz.log()),
//             "");
//         if (foo_t_average) {
//           SOPHUS_TEST_APPROX(
//               passed,
//               foo_t_quiz.matrix(),
//               foo_t_average->matrix(),
//               sqrt_eps,
//               "log(foo_transform_bar): %\nlog(foo_transform_daz): %\n"
//               "log(interp): %\nlog(average): %",
//               transpose(foo_transform_bar.log()),
//               transpose(foo_transform_daz.log()),
//               transpose(foo_t_quiz.log()),
//               transpose(foo_t_average->log()),
//               "");
//         }
//         SOPHUS_TEST(
//             passed,
//             bool(foo_t_iaverage),
//             "log(foo_transform_bar): %\nlog(foo_transform_daz): %\n"
//             "log(interp): %\nlog(iaverage): %",
//             transpose(foo_transform_bar.log()),
//             transpose(foo_transform_daz.log()),
//             transpose(foo_t_quiz.log()),
//             transpose(foo_t_iaverage->log()),
//             "");
//         if (foo_t_iaverage) {
//           SOPHUS_TEST_APPROX(
//               passed,
//               foo_t_quiz.matrix(),
//               foo_t_iaverage->matrix(),
//               sqrt_eps,
//               "log(foo_transform_bar): %\nlog(foo_transform_daz): %",
//               transpose(foo_transform_bar.log()),
//               transpose(foo_transform_daz.log()),
//               "");
//         }
//       }
//     }

//     return passed;
//   }

//   bool testRandomSmoke() {
//     bool passed = true;
//     std::default_random_engine engine;
//     for (int i = 0; i < 100; ++i) {
//       LieGroup g = LieGroup::sampleUniform(engine);
//       SOPHUS_TEST_EQUAL(passed, g.params(), g.params(), "");
//     }
//     return passed;
//   }

//   template <class TS = Scalar>
//   std::enable_if_t<std::is_same<TS, float>::value, bool> testSpline() {
//     // skip tests for Scalar == float
//     return true;
//   }

//   template <class TS = Scalar>
//   std::enable_if_t<!std::is_same<TS, float>::value, bool> testSpline() {
//     // run tests for Scalar != float
//     bool passed = true;

//     for (LieGroup const& t_world_foo : group_vec_) {
//       for (LieGroup const& t_world_bar : group_vec_) {
//         std::vector<LieGroup> control_poses;
//         control_poses.push_back(interpolate(t_world_foo, t_world_bar, 0.0));

//         for (double p = 0.2; p < 1.0; p += 0.2) {
//           LieGroup t_world_inter = interpolate(t_world_foo, t_world_bar, p);
//           control_poses.push_back(t_world_inter);
//         }

//         BasisSplineImpl<LieGroup> spline(control_poses, 1.0);

//         LieGroup t = spline.parentFromSpline(0.0, 1.0);
//         LieGroup t2 = spline.parentFromSpline(1.0, 0.0);

//         SOPHUS_TEST_APPROX(
//             passed,
//             t.matrix(),
//             t2.matrix(),
//             10 * small_eps_sqrt,
//             "parent_T_spline");

//         Transformation dt_parent_t_spline = spline.dtParentFromSpline(0.0,
//         0.5); Transformation dt_parent_t_spline2 = curveNumDiff(
//             [&](double u_bar) -> Transformation {
//               return spline.parentFromSpline(0.0, u_bar).matrix();
//             },
//             0.5);
//         SOPHUS_TEST_APPROX(
//             passed,
//             dt_parent_t_spline,
//             dt_parent_t_spline2,
//             100 * small_eps_sqrt,
//             "Dt_parent_T_spline");

//         Transformation dt2_parent_t_spline =
//             spline.dt2ParentFromSpline(0.0, 0.5);
//         Transformation dt2_parent_t_spline2 = curveNumDiff(
//             [&](double u_bar) -> Transformation {
//               return spline.dtParentFromSpline(0.0, u_bar).matrix();
//             },
//             0.5);
//         SOPHUS_TEST_APPROX(
//             passed,
//             dt2_parent_t_spline,
//             dt2_parent_t_spline2,
//             20 * small_eps_sqrt,
//             "Dt2_parent_T_spline");

//         for (double frac : {0.01, 0.25, 0.5, 0.9, 0.99}) {
//           double t0 = 1.0;
//           double delta_t = 0.1;
//           BasisSpline<LieGroup> spline(control_poses, t0, delta_t);
//           double t = t0 + frac * delta_t;

//           Transformation dt_parent_t_spline = spline.dtParentFromSpline(t);
//           Transformation dt_parent_t_spline2 = curveNumDiff(
//               [&](double t_bar) -> Transformation {
//                 return spline.parentFromSpline(t_bar).matrix();
//               },
//               t);
//           SOPHUS_TEST_APPROX(
//               passed,
//               dt_parent_t_spline,
//               dt_parent_t_spline2,
//               80 * small_eps_sqrt,
//               "Dt_parent_T_spline");

//           Transformation dt2_parent_t_spline = spline.dt2ParentFromSpline(t);
//           Transformation dt2_parent_t_spline2 = curveNumDiff(
//               [&](double t_bar) -> Transformation {
//                 return spline.dtParentFromSpline(t_bar).matrix();
//               },
//               t);
//           SOPHUS_TEST_APPROX(
//               passed,
//               dt2_parent_t_spline,
//               dt2_parent_t_spline2,
//               20 * small_eps_sqrt,
//               "Dt2_parent_T_spline");
//         }
//       }
//     }
//     return passed;
//   }

template <class G>
void tests(
    std::string group_name,
    std::vector<Eigen::Vector<typename G::Scalar, G::kPointDim>> const&
        point_vec) {
  expLogTest<G>(group_name);
  groupActionTest<G>(group_name, point_vec);
  // leftJacobianTest<G>(group_name);
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
