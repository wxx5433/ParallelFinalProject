default:
	g++ -fopenmp -g -O2 -o cpu_bc main.cpp graph.cpp cpu_bc.cpp
	#g++ -fopenmp -g -O0 -o cpu_bc main.cpp graph.cpp cpu_bc.cpp

run: ./cpu_bc
	#LD_LIBRARY_PATH=./lib/:$(LD_LIBRARY_PATH) ./bfs ${GRAPH} $(N_THREADS)

clean:
	rm -rf bfs *.o *~ *.*~
