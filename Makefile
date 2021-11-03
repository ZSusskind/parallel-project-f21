CXX := /usr/bin/g++
CXX_FLAGS := -std=c++17 -Wall -Wextra
LIB_FLAGS := -fopenmp

OBJS := $(subst .cpp,,$(wildcard *.cpp))

$(OBJS): %: %.cpp
	$(CXX) $(CXX_FLAGS) $< $(LIB_FLAGS) -o $@

.PHONY: all
all: $(OBJS)
