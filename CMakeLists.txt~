cmake_minimum_required(VERSION 2.8.3)
project(epuck_filter)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)


catkin_package(
   INCLUDE_DIRS include
   LIBRARIES epuck_filter
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS} /usr/include/eigen3/)

## Declare a C++ executable
 add_executable(epuck_filter
	 src/epuck_filter_node.cpp
	 src/KalmanFilter.cpp
         )

target_link_libraries(epuck_filter ${catkin_LIBRARIES}) 

add_dependencies(epuck_filter ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


 install(TARGETS epuck_filter
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )


