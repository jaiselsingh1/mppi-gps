cmake_minimum_required(VERSION 3.20)
project(acrobot_mjpc_reference LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
include(FetchContent)
FetchContent_Declare(abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG fb3621f4f897824c0dbe0615fa94543df6192f30)
FetchContent_MakeAvailable(abseil)
if(NOT EXISTS "${MJPC_SOURCE}/mjpc/tasks/acrobot/acrobot.cc")
  message(FATAL_ERROR "MJPC_SOURCE must name the frozen official source export")
endif()
if(NOT EXISTS "${MUJOCO_SDK}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "MUJOCO_SDK must name the existing MuJoCo 3.6.0 wheel directory")
endif()
find_package(Threads REQUIRED)
add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES
  IMPORTED_LOCATION "${MUJOCO_SDK}/libmujoco.so.3.6.0"
  INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_SDK}/include")
set(MJPC_FILES task.cc norm.cc trajectory.cc utilities.cc threadpool.cc
  states/state.cc tasks/acrobot/acrobot.cc planners/planner.cc
  planners/model_derivatives.cc planners/cost_derivatives.cc
  planners/sampling/planner.cc planners/sampling/policy.cc
  planners/ilqg/planner.cc planners/ilqg/policy.cc
  planners/ilqg/backward_pass.cc spline/spline.cc)
list(TRANSFORM MJPC_FILES PREPEND "${MJPC_SOURCE}/mjpc/")
add_library(reference_core STATIC ${MJPC_FILES})
target_include_directories(reference_core PUBLIC "${MJPC_SOURCE}")
target_compile_options(reference_core PRIVATE "-include" "${CMAKE_CURRENT_SOURCE_DIR}/compat.h")
# Calls are redirected; downloaded upstream source bytes stay intact.
target_compile_definitions(reference_core PRIVATE
  SOURCE_DIR="${MJPC_SOURCE}"
  mj_step=reference_counted_mj_step
  mjd_transitionFD=reference_rk4_transitionFD)
target_link_libraries(reference_core PUBLIC mujoco Threads::Threads
  absl::flat_hash_map absl::strings absl::check absl::log
  absl::random_random absl::span absl::base)
add_executable(acrobot_mjpc_reference main.cc)
target_link_libraries(acrobot_mjpc_reference PRIVATE reference_core)
set_target_properties(acrobot_mjpc_reference PROPERTIES BUILD_RPATH "${MUJOCO_SDK}")
