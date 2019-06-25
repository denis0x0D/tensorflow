# Description:
#   Vulkan-platform specific StreamExecutor support code.

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//tensorflow:tensorflow.bzl", "tf_copts")
load("@local_config_vulkan//vulkan:build_defs.bzl", "if_vulkan_is_configured")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

package(
    default_visibility = [":friends"],
)

# Filegroup used to collect source files for the dependency check.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "vulkan_platform_id",
    srcs = ["vulkan_platform_id.cc"],
    hdrs = ["vulkan_platform_id.h"],
    deps = ["//tensorflow/stream_executor:platform"],
)
