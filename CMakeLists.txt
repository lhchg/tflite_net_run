
populate_source_vars("${TFLITE_SOURCE_DIR}/examples/tflite_net_run"
  TFLITE_NET_RUN_SRCS
  FILTER "_src\\.cc$"
)
file (GLOB SRC_FILES ${TFLITE_SOURCE_DIR}/examples/tflite_net_run/src/*.cc)
list(APPEND TFLITE_NET_RUN_SRCS
  ${SRC_FILES}
)
list(APPEND TFLITE_NET_RUN_SRCS
  ${TF_SOURCE_DIR}/core/util/stats_calculator.cc
  ${TFLITE_SOURCE_DIR}/profiling/memory_info.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summarizer.cc
  ${TFLITE_SOURCE_DIR}/profiling/profile_summary_formatter.cc
  ${TFLITE_SOURCE_DIR}/profiling/time.cc
  ${TFLITE_SOURCE_DIR}/tools/command_line_flags.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/default_execution_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/delegates/delegate_provider.cc
  ${TFLITE_SOURCE_DIR}/tools/evaluation/utils.cc
  ${TFLITE_SOURCE_DIR}/tools/tool_params.cc
)

if(TFLITE_ENABLE_XNNPACK)
  list(APPEND TFLITE_NET_RUN_SRCS
    #${TFLITE_SOURCE_DIR}/tools/delegates/xnnpack_delegate_provider.cc
    #${TFLITE_SOURCE_DIR}/core/acceleration/configuration/c/xnnpack_plugin.cc
  )
else()
  set(TFLITE_TFLITE_NET_RUN_CC_OPTIONS "-DTFLITE_WITHOUT_XNNPACK")
endif()  # TFLITE_ENABLE_XNNPACK

if(CMAKE_SYSTEM_NAME MATCHES "Android")
  if(_TFLITE_ENABLE_NNAPI)
    list(APPEND TFLITE_NET_RUN_SRCS
      ${TFLITE_SOURCE_DIR}/tools/delegates/nnapi_delegate_provider.cc
    )
  endif()  # _TFLITE_ENABLE_NNAPI
endif()  # Android

if(TFLITE_ENABLE_GPU)
  #file(GLOB GPU_DELEGATE_FILES ${TFLITE_SOURCE_DIR}/delegates/gpu/*.cc)
  #list(FILTER GPU_DELEGATE_FILES EXCLUDE REGEX ".*test.*")
  #file(GLOB ASYNC_FILES ${TFLITE_SOURCE_DIR}/async/*.cc)
  list(APPEND TFLITE_NET_RUN_SRCS
    ${TFLITE_SOURCE_DIR}/tools/delegates/gpu_delegate_provider.cc
    #${GPU_DELEGATE_FILES}
    
  )
endif()  # TFLITE_ENABLE_GPU

add_executable(tflite_net_run
  ${TFLITE_NET_RUN_SRCS}
)

target_link_libraries(tflite_net_run
  tensorflow-lite
)
