cmake_minimum_required(VERSION 2.8.3)
project(promp_robot_control_msgs)

#set(CMAKE_BUILD_TYPE Debug)
set(ROS_BUILD_TYPE RelWithDebInfo)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs 
  nav_msgs
)

#######################################
## Declare ROS messages and services ##
#######################################
# Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg
  FILES
  Start.msg
  PrompStatus.msg
  TemporalScalingStatus.msg
  PredictedTrajectories.msg
)

## Generate services in the 'srv' folder
add_service_files(DIRECTORY srv
  FILES
  Goals.srv
  Belief.srv
  ChangeState.srv
  PrompControlUpdate.srv
  StartExperiment.srv
  OnlinePhaseAdaptationUpdate.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  nav_msgs
)

###################################################
## Declare things to be passed to other projects ##
###################################################
catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs nav_msgs
)

