#!/bin/bash
ulimit -t 1
#out=`python bf.py -f /tmp/program.bf`
#out=`clisp /tmp/program.bf`
#out=`./uni -b < /tmp/program.bf`
xxd -r /tmp/program.bf /tmp/program.bin
chmod u+x /tmp/program.bin
out='/tmp/program.bin'
if [[ $out == 'Hello World!' ]]; then
	exit 0
else
	exit 255
fi
