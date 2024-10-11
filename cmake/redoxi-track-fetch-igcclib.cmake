include(redoxi-track-fetch-common)

function(fetch_igcclib)
    set(multiValueArgs COMPONENTS)
    cmake_parse_arguments(FETCH_IGCCLIB "" "" "${multiValueArgs}" ${ARGN})

    set(components ${FETCH_IGCCLIB_COMPONENTS})
    if(NOT components)
        set(WITH_ALL_COMPONENTS ON)    # by default fetch all components
    else()
        set(WITH_ALL_COMPONENTS OFF)    # fetch only specified components
    endif()
    
    # for each comp in componets, define a variable called WITH_<comp> and set it to ON
    foreach(comp ${components})
        string(TOUPPER ${comp} comp_upper)
        set(WITH_${comp_upper} ON)
    endforeach()

    # fetch igcclib from github
    # url : https://github.com/igamenovoer/igcclib.git
    FetchContent_Declare(
        igcclib
        GIT_REPOSITORY https://github.com/igamenovoer/igcclib.git
        GIT_TAG        master
        GIT_SHALLOW    1 # only fetch the latest commit
    )

    # make it available
    FetchContent_MakeAvailable(igcclib)

    # set igcclib_DIR to the directory where igcclib was fetched
    set(igcclib_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
endfunction()