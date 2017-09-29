set( SRCS SingleRover.cpp MAPElitesRover.cpp Target.cpp MultiRover.cpp Bar.cpp MultiNightBar.cpp)
add_library( Domains SHARED ${SRCS} )
target_link_libraries(Domains Learning Utilities)
