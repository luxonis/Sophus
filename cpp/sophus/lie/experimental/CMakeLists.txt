#[[
sophus_lie_group_experimental



]]

farm_ng_add_library(sophus_lie_group_experimental
  NAMESPACE Sophus
  INCLUDE_DIR ../../
  HEADERS
  lie_group_concept
)
target_link_libraries(sophus_lie_group_experimental INTERFACE sophus_linalg)


if(${BUILD_SOPHUS_TESTS})
  foreach(test_basename
    lie_group_concept
  )
    farm_ng_add_test(${test_basename}
                          PARENT_LIBRARY sophus_lie_group_experimental
                          LINK_LIBRARIES sophus_lie_group_experimental
                          LABELS small)
  endforeach()
endif()
