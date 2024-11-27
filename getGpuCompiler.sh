#!/bin/bash

if command -v hipcc 2>&1 >/dev/null
then
	echo hipcc
	exit 0
fi

if command -v cu++ 2>&1 >/dev/null
then
	echo cu++
	exit 0
fi

echo null
exit 1
