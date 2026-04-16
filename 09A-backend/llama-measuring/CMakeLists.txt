# ── energy_measure (shared utility) ──────────────────────────────────────────
add_library(energy-measure STATIC energy-measure.cpp)
target_compile_features(energy-measure PRIVATE cxx_std_17)

# ── llama-papi ──────────────────────────────────────────────────────────────
find_package(PostgreSQL REQUIRED)
set(TARGET llama-papi)
add_executable(${TARGET} llama-papi.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE /usr/local/include ${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so ${PostgreSQL_LIBRARIES})

# ── kv-measure ───────────────────────────────────────────────────────────────
set(TARGET kv-measure)
add_executable(${TARGET} kv-measure.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

# ── llama-energy ─────────────────────────────────────────────────────────────
set(TARGET llama-energy)
add_executable(${TARGET} llama-energy.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})

# ── llama-measurement-top-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-top-view)
add_executable(${TARGET} llama-measurement-top-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE /usr/local/include ${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so ${PostgreSQL_LIBRARIES} energy-measure)

# ── llama-measurement-phase-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-phase-view)
add_executable(${TARGET} llama-measurement-phase-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE /usr/local/include ${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so energy-measure)

# ── llama-measurement-decoder-block-view ─────────────────────────────────────────────────────────────
set(TARGET llama-measurement-decoder-block-view)
add_executable(${TARGET} llama-measurement-decoder-block-view.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE /usr/local/include ${PostgreSQL_INCLUDE_DIRS})
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} papi /usr/local/lib/libpapi.so)