project(pythorch)
aux_source_directory(. DIR_LIB_SRCS)
include_directories(.)
add_library(pythorch ${DIR_LIB_SRCS})