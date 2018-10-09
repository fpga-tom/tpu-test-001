#!/bin/bash
ulimit -t 1
#out=`python bf.py -f /tmp/program.bf`
out=`clisp /tmp/program.bf`
if [[ $out == 'Hello World!' ]]; then
	exit 0
else
	exit 255
fi
