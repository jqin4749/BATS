#!/bin/bash 

export AOCL_BOARD_PACKAGE_ROOT=/home/jack/softwares/intelFPGA_pro/18.1/hld/board/s5_ref
make CXX=g++
aoc -march=emulator  dev/coder.cl -o ../bin/emulator/coder -I include
cp ../bin/host ../bin/emulator/
cp ../bin/emulator/coder.aocx /home/jack/Desktop/bats-code/bats-lib/coder.aocx
