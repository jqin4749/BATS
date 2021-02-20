#!/bin/bash 

export AOCL_BOARD_PACKAGE_ROOT=/home/jack/softwares/intelFPGA_pro/18.1/hld/board/s5_ref
make CXX=g++
aoc -march=emulator  dev/myGEMM6.cl -o ../bin/emulator/myGEMM6 -I include
cp ../bin/host ../bin/emulator/
