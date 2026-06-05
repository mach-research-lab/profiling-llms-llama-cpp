# ── utils (shared utility) ──────────────────────────────────────────
set(TARGET utils)
add_library(utils STATIC utils.cpp)
target_compile_features(utils PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

# ── llama-papi ──────────────────────────────────────────────────────────────
set(TARGET llama-papi)
add_executable(${TARGET} llama-papi.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} 
    ${CMAKE_SOURCE_DIR}/papi-install/lib/libpapi.a sqlite3 utils)

# ── kv-measure ───────────────────────────────────────────────────────────────
set(TARGET kv-measure)
add_executable(${TARGET} kv-measure.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} utils)

# ── llama-energy ─────────────────────────────────────────────────────────────
set(TARGET llama-energy)
add_executable(${TARGET} llama-energy.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} utils)

# ── llama-measurement-top-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-top-view)
add_executable(${TARGET} llama-measurement-top-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} 
    ${CMAKE_SOURCE_DIR}/papi-install/lib/libpapi.a sqlite3 utils)

# ── llama-measurement-phase-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-phase-view)
add_executable(${TARGET} llama-measurement-phase-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} 
    ${CMAKE_SOURCE_DIR}/papi-install/lib/libpapi.a sqlite3 utils)

# ── llama-measurement-decoder-block-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-decoder-block-view)
add_executable(${TARGET} llama-measurement-decoder-block-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} 
    ${CMAKE_SOURCE_DIR}/papi-install/lib/libpapi.a sqlite3 utils)