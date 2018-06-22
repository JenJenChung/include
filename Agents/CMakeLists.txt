set( SRCS Rover.cpp BarAgent.cpp Intersection.cpp)
add_library( Agents STATIC ${SRCS} )
target_link_libraries(Agents Domains Learning Utilities POMDPs)
