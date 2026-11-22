TARGET := storybananamind-wii
OBJS := wii_story.o story_data.o
DATA := story-wii-weights.bin story-wii-tokenizer.bin
MODEL_STAMP := .model-data-stamp
CONVERTED_STAMP := .converted-stamp

DEVKITPRO ?= /opt/devkitpro
DEVKITPPC ?= $(DEVKITPRO)/devkitPPC
DEPSDIR := .
include $(DEVKITPPC)/wii_rules

INCLUDE := -I$(LIBOGC_INC)
CFLAGS := -g -O3 -Wall -Wextra $(MACHDEP) $(INCLUDE)
CXXFLAGS := $(CFLAGS) -std=gnu++11
LD := $(CXX)
LDFLAGS := $(MACHDEP) -Wl,-Map,$(TARGET).map
LIBPATHS := -L$(LIBOGC_LIB)
LIBS := -lwiikeyboard -lwiiuse -lbte -logc -lm

all: $(TARGET).dol wii-ran-ai-hbc.zip

$(TARGET).elf: $(OBJS)

$(MODEL_STAMP): download_model.py
	python3 download_model.py
	touch $@

config.json model.safetensors tokenizer.json: $(MODEL_STAMP)
	@test -f $@

$(CONVERTED_STAMP): config.json model.safetensors tokenizer.json convert_story_raw.py
	python3 convert_story_raw.py
	touch $@

$(DATA): $(CONVERTED_STAMP)
	@test -f $@

story_data.o: $(DATA)

wii-ran-ai-hbc.zip: $(TARGET).dol meta.xml icon.png package_hbc.py
	python3 package_hbc.py

clean:
	rm -f $(OBJS) *.d $(TARGET).elf $(TARGET).dol $(TARGET).map \
		wii-ran-ai-hbc.zip $(MODEL_STAMP) $(CONVERTED_STAMP)

.PHONY: all clean
