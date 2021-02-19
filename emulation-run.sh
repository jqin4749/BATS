#!/bin/bash 
export AOCL_BOARD_PACKAGE_ROOT=/home/jack/softwares/intelFPGA_pro/18.1/hld/board/s5_ref
cd ../bin/emulator 
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./host
cd -
