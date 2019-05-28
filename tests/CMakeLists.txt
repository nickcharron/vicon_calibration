# Executables for tests / development / debugging

file(COPY test_bags
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

file(COPY template_pointclouds
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

file(COPY calibration
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)
