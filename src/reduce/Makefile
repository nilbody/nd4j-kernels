include ../../vars.mk
CC:=nvcc
SUBDIRS := \
bias \
iamax \
max \
mean \
min \
norm1 \
norm2 \
normmax \
prod \
std \
sum \
var


all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_strided.cu  -o $(shell pwd)/../../output/$@_strided.ptx

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)

