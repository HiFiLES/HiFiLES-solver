include_directories(.)
# Create programs.
add_executable(ptest ptest.c io.c adaptgraph.c)
add_executable(mtest mtest.c io.c)
add_executable(parmetis_prog parmetis.c io.c adaptgraph.c)
set_target_properties(parmetis_prog PROPERTIES OUTPUT_NAME parmetis)
add_executable(pometis pometis.c io.c)
# Link to libparmetis.
foreach(prog ptest mtest parmetis_prog pometis)
  target_link_libraries(${prog} parmetis)
endforeach(prog)

install(TARGETS ptest mtest parmetis_prog pometis
  RUNTIME DESTINATION bin)
