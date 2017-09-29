set( SRCS Rover.cpp BarAgent.cpp)
add_library( Agents SHARED ${SRCS} )
target_link_libraries(Agents Domains Learning Utilities)
