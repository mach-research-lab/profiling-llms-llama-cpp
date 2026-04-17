# ── utils (shared utility) ──────────────────────────────────────────
add_library(utils STATIC utils.cpp)
target_compile_features(utils PRIVATE cxx_std_17)

# ── llama-papi ──────────────────────────────────────────────────────────────
find_package(SQLite3 REQUIRED)
set(TARGET llama-papi)
add_executable(${TARGET} llama-papi.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE /usr/local/include)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so SQLite::SQLite3 utils)

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
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so utils)

# ── llama-measurement-phase-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-phase-view)
add_executable(${TARGET} llama-measurement-phase-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so utils)

# ── llama-measurement-decoder-block-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-decoder-block-view)
add_executable(${TARGET} llama-measurement-decoder-block-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so utils)
