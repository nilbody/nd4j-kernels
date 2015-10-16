include ../../vars.mk

SUBDIRS := \
cosinesimilarity \
euclidean \
manhattan


all: $(SUBDIRS)
$(SUBDIRS):
	cd $@ && $(BASE_COMPILE)  $(shell pwd)/$@/$@_strided.cu   -o $(shell pwd)/../../output/$@.ptx

.PHONY: all $(SUBDIRS)
.PHONY: all $(SUBDIRS)
