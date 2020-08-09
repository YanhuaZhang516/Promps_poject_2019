cmake_minimum_required(VERSION 2.8.3)
project(belief_tracker)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy 
  tf 
  std_msgs
  
)

catkin_package(
  LIBRARIES belief_tracker
  CATKIN_DEPENDS tf rospy  std_msgs
)

catkin_python_setup()

