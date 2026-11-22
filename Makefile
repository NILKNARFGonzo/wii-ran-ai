TARGET := storybananamind-wii
OBJS := wii_story.o story_data.o
DATA := story-wii-weights.bin story-wii-tokenizer.bin \
	chat-wii-weights.bin chat-wii-tokenizer.bin
MODEL_STAMP := .model-data-stamp
CHAT_MODEL_STAMP := .chat-model-data-stamp
CONVERTED_STAMP := .converted-stamp
CHAT_CONVERTED_STAMP := .chat-converted-stamp

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

$(CHAT_MODEL_STAMP): download_chat_model.py
	python3 download_chat_model.py
	touch $@

config.json model.safetensors tokenizer.json: $(MODEL_STAMP)
	@test -f $@

chat-config.json chat-model.safetensors chat-tokenizer.json: $(CHAT_MODEL_STAMP)
	@test -f $@

$(CONVERTED_STAMP): config.json model.safetensors tokenizer.json convert_story_raw.py
	python3 convert_story_raw.py
	touch $@

$(CHAT_CONVERTED_STAMP): chat-config.json chat-model.safetensors chat-tokenizer.json convert_chat_model.py
	python3 convert_chat_model.py
	touch $@

story-wii-weights.bin story-wii-tokenizer.bin: $(CONVERTED_STAMP)
	@test -f $@

chat-wii-weights.bin chat-wii-tokenizer.bin: $(CHAT_CONVERTED_STAMP)
	@test -f $@

story_data.o: $(DATA) story_data.S

wii-ran-ai-hbc.zip: $(TARGET).dol meta.xml icon.png LICENSE package_hbc.py
	python3 package_hbc.py

clean:
	rm -f $(OBJS) *.d $(TARGET).elf $(TARGET).dol $(TARGET).map \
		wii-ran-ai-hbc.zip $(MODEL_STAMP) $(CHAT_MODEL_STAMP) \
		$(CONVERTED_STAMP) $(CHAT_CONVERTED_STAMP)

.PHONY: all clean
