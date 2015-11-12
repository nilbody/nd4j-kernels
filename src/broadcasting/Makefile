include ../../vars.mk
CC:=nvcc
SUBDIRS := \
add \
copy \
div \
mul \
rdiv \
rsub \
sub


all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@.cu  $(LINK) -o $(shell pwd)/../../output/$@_broadcast.$(EXTENSION)

.PHONY: all $(SUBDIRS)

