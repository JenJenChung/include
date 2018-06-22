set( SRCS SingleRover.cpp MAPElitesRover.cpp Target.cpp MultiRover.cpp MultiNightBar.cpp Bar.cpp Warehouse.cpp AGV.cpp)
add_library( Domains STATIC ${SRCS} )
target_link_libraries(Domains Learning Utilities Agents)
