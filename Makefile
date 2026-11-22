TARGET := storybananamind-wii
OBJS := wii_story.o story_data.o
DATA := story-wii-weights.bin story-wii-tokenizer.bin \
	chat-wii-weights.bin chat-wii-tokenizer.bin
MEDIUM_CHAT_DATA := medium-chat-wii-weights.bin medium-chat-wii-tokenizer.bin
MODEL_STAMP := .model-data-stamp
CHAT_MODEL_STAMP := .chat-model-data-stamp
MEDIUM_CHAT_MODEL_STAMP := .medium-chat-model-stamp
CONVERTED_STAMP := .converted-stamp
CHAT_CONVERTED_STAMP := .chat-converted-stamp
MEDIUM_CHAT_CONVERTED_STAMP := .medium-chat-converted-stamp

DEVKITPRO ?= /opt/devkitpro
DEVKITPPC ?= $(DEVKITPRO)/devkitPPC
DEPSDIR := .
include $(DEVKITPPC)/wii_rules

INCLUDE := -I$(LIBOGC_INC)
FATDIR ?= $(DEVKITPRO)/libogc
INCLUDE += -I$(FATDIR)/include
CFLAGS := -g -O3 -Wall -Wextra $(MACHDEP) $(INCLUDE)
CXXFLAGS := $(CFLAGS) -std=gnu++11
LD := $(CXX)
LDFLAGS := $(MACHDEP) -Wl,-Map,$(TARGET).map
LIBPATHS := -L$(LIBOGC_LIB) -L$(FATDIR)/lib/wii
LIBS := -lwiikeyboard -lwiiuse -lbte -lfat -logc -lm

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

$(MEDIUM_CHAT_MODEL_STAMP): download_medium_chat_model.py
	python3 download_medium_chat_model.py
	touch $@

medium-chat-config.json medium-chat-model.safetensors medium-chat-tokenizer.json: $(MEDIUM_CHAT_MODEL_STAMP)
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

$(MEDIUM_CHAT_CONVERTED_STAMP): medium-chat-config.json medium-chat-model.safetensors medium-chat-tokenizer.json convert_chat_model.py convert_medium_chat_model.py
	python3 convert_medium_chat_model.py
	touch $@

medium-chat-wii-weights.bin medium-chat-wii-tokenizer.bin: $(MEDIUM_CHAT_CONVERTED_STAMP)
	@test -f $@

story_data.o: $(DATA) story_data.S

wii-ran-ai-hbc.zip: $(TARGET).dol meta.xml icon.png LICENSE package_hbc.py $(MEDIUM_CHAT_DATA)
	python3 package_hbc.py

clean:
	rm -f $(OBJS) *.d $(TARGET).elf $(TARGET).dol $(TARGET).map \
		wii-ran-ai-hbc.zip $(MODEL_STAMP) $(CHAT_MODEL_STAMP) \
		$(MEDIUM_CHAT_MODEL_STAMP) $(CONVERTED_STAMP) \
		$(CHAT_CONVERTED_STAMP) $(MEDIUM_CHAT_CONVERTED_STAMP)

.PHONY: all clean
