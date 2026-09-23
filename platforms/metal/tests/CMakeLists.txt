#
# Testing
#

ENABLE_TESTING()

# Automatically create tests using files named "Test*.cpp"
FILE(GLOB TEST_PROGS "*Test*.cpp")
FOREACH(TEST_PROG ${TEST_PROGS})
    GET_FILENAME_COMPONENT(TEST_ROOT ${TEST_PROG} NAME_WE)

    # Link with shared library
    ADD_EXECUTABLE(${TEST_ROOT} ${TEST_PROG})
    TARGET_LINK_LIBRARIES(${TEST_ROOT} ${SHARED_TARGET})
    SET_TARGET_PROPERTIES(${TEST_ROOT} PROPERTIES LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}" CXX_STANDARD 17)

    ADD_TEST(${TEST_ROOT}Single ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} single)
    ADD_TEST(${TEST_ROOT}Mixed ${EXECUTABLE_OUTPUT_PATH}/${TEST_ROOT} mixed)

ENDFOREACH(TEST_PROG ${TEST_PROGS})

# QTBIntegrator calls cos on mixed values, which Metal has no df64 version of, so in mixed precision it
# must fail with the error for that instead of running in single precision.
SET_TESTS_PROPERTIES(TestMetalQTBIntegratorMixed PROPERTIES PASS_REGULAR_EXPRESSION
    "exception: Error compiling program: cos is called on a mixed precision value, which Metal can only evaluate in single precision")
