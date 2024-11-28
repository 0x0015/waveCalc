#!/bin/bash

if command -v hipcc 2>&1 >/dev/null
then
	echo -DAGPU_BACKEND_HIP
	exit 0
fi

if command -v nvcc 2>&1 >/dev/null
then
	echo -DAGPU_BACKEND_CUDA --x cu --extended-lambda -ccbin='gcc-13' 
	exit 0
fi

echo " "
exit 1
