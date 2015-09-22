LIBRARY := libfast-dnn.so
JAVA_HOME := /usr/lib/jvm/java-8-oracle
PROJECT_HOME := /home/afsina/projects/fast-dnn
OBJFILES := dnn.o jni_dnn.o float_dnn.o
INCLUDES := -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux
CXXFLAGS := $(INCLUDES) -g -O2 -march=native -mtune=native -std=c++11 -funroll-loops -msse4 -fPIC

$(LIBRARY): $(OBJFILES)
	$(CXX) -shared -o $@ $^
	cp $@ $(PROJECT_HOME)/src/java/resources