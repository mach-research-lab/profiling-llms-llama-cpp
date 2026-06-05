cmake_minimum_required(VERSION 3.14)
project("09A" C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build llama.cpp as a dependency
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)

add_subdirectory(llama.cpp)

# Our tools
add_subdirectory(09A-backend/llama-measuring)

# Build SQLite
add_library(sqlite3 STATIC sqlite-local/sqlite3.c)
target_include_directories(sqlite3 PUBLIC sqlite-local)

# Build PAPI
set(PAPI_INSTALL_DIR ${CMAKE_SOURCE_DIR}/papi-install)
if(NOT EXISTS ${PAPI_INSTALL_DIR}/lib/libpapi.a)
    message(STATUS "Building PAPI from source...")
    execute_process(
        COMMAND ./configure --prefix=${PAPI_INSTALL_DIR}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/papi/src
    )
    execute_process(
        COMMAND make -j
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/papi/src
    )
    execute_process(
        COMMAND make install
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/papi/src
    )
endif()

add_library(papi STATIC IMPORTED GLOBAL)
set_target_properties(papi PROPERTIES
    IMPORTED_LOCATION ${PAPI_INSTALL_DIR}/lib/libpapi.a
    INTERFACE_INCLUDE_DIRECTORIES ${PAPI_INSTALL_DIR}/include
)

# Build Stream
set(STREAM_BINARY ${CMAKE_SOURCE_DIR}/stream/stream_c.exe)
if(NOT EXISTS ${STREAM_BINARY})
    message(STATUS "Building Stream benchmark...")
    execute_process(
        COMMAND make
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/stream
    )
endif()

find_program(NPM_EXECUTABLE npm)
if(NPM_EXECUTABLE)
    message(STATUS "Building React frontend...")
    execute_process(
        COMMAND ${NPM_EXECUTABLE} install
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/09A-frontend/web-interface
    )
    execute_process(
        COMMAND ${NPM_EXECUTABLE} run build
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/09A-frontend/web-interface
    )
else()
    message(WARNING "npm not found — frontend will not be built. Install Node.js to enable the GUI.")
endif()