include ../../vars.mk
SUBDIRS := \
add \
div \
mul \
rdiv \
rsub \
softmax \
sub \
eps \
copy \
eq \
gt \
lt

all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_strided.cu  -o -o $(shell pwd)/../../output/$@.ptx

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)

