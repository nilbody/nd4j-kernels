include ../../vars.mk

SUBDIRS := \
add \
div \
equals \
greaterthan \
greaterthanorequal \
lessthan \
max \
min \
mul \
notequals \
rdiv \
rsub \
set \
setvalorless \
sub


all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_scalar.cu  -o $(shell pwd)/../../output/$@.ptx

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)

