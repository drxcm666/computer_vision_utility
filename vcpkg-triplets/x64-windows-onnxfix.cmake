set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

# onnxruntime in vcpkg expects onnx with static schema registration disabled.
list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS "-DONNX_DISABLE_STATIC_REGISTRATION=ON")
