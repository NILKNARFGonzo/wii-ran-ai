	.section .rodata
	.balign 32
	.global story_weights_start
story_weights_start:
	.incbin "story-wii-weights.bin"
	.global story_weights_end
story_weights_end:

	.balign 32
	.global story_tokenizer_start
story_tokenizer_start:
	.incbin "story-wii-tokenizer.bin"
	.global story_tokenizer_end
story_tokenizer_end:
