# Executables for tests / development / debugging

add_executable(cyl_extraction_main
  cyl_extract_main.cpp
  ../src/lib/LidarCylExtractor.cpp
)
target_include_directories(cyl_extraction_main
  PUBLIC
  ../include
)
target_link_libraries(cyl_extraction_main
  ${rosbag_LIBRARIES}
  beam::calibration
  PCL::PCL
)

file(COPY test_bags
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

file(COPY template_pointclouds
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

file(COPY calibration
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
