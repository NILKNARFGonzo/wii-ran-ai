#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef GEKKO
#include <gccore.h>
#include <ogc/console.h>
#include <wiikeyboard/keyboard.h>
#include <wiikeyboard/usbkeyboard.h>
#include <wiiuse/wpad.h>
#endif

namespace {

struct Tensor {
    uint8_t type;
    std::vector<uint32_t> shape;
    float scale;
    const uint8_t* data;
    size_t byte_count;

    size_t elements() const {
        size_t count = 1;
        for (size_t i = 0; i < shape.size(); ++i) count *= shape[i];
        return count;
    }

    float at(size_t index) const {
        if (type == 1) {
            uint32_t bits = (static_cast<uint32_t>(data[index * 4]) << 24) |
                            (static_cast<uint32_t>(data[index * 4 + 1]) << 16) |
                            (static_cast<uint32_t>(data[index * 4 + 2]) << 8) |
                            static_cast<uint32_t>(data[index * 4 + 3]);
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        return static_cast<int8_t>(data[index]) * scale;
    }
};

struct MergeRule {
    uint32_t rank;
    uint16_t result;
};

class BinaryReader {
public:
    explicit BinaryReader(const char* path)
        : file_(std::fopen(path, "rb")), data_(nullptr), size_(0), position_(0) {
        if (!file_) throw std::runtime_error(std::string("Cannot open ") + path);
    }

    BinaryReader(const uint8_t* data, size_t size)
        : file_(nullptr), data_(data), size_(size), position_(0) {}

    ~BinaryReader() {
        if (file_) std::fclose(file_);
    }

    bool read(void* output, size_t size) {
        if (file_) return std::fread(output, 1, size, file_) == size;
        if (size > size_ - position_) return false;
        std::memcpy(output, data_ + position_, size);
        position_ += size;
        return true;
    }

    bool read_view(const uint8_t*& output, size_t size) {
        if (file_ || size > size_ - position_) return false;
        output = data_ + position_;
        position_ += size;
        return true;
    }

    bool read_u8(uint8_t& value) { return read(&value, 1); }

    bool read_u16(uint16_t& value) {
        uint8_t b[2];
        if (!read(b, sizeof(b))) return false;
        value = (static_cast<uint16_t>(b[0]) << 8) | b[1];
        return true;
    }

    bool read_u32(uint32_t& value) {
        uint8_t b[4];
        if (!read(b, sizeof(b))) return false;
        value = (static_cast<uint32_t>(b[0]) << 24) |
                (static_cast<uint32_t>(b[1]) << 16) |
                (static_cast<uint32_t>(b[2]) << 8) | b[3];
        return true;
    }

    bool read_f32(float& value) {
        uint32_t bits;
        if (!read_u32(bits)) return false;
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

private:
    FILE* file_;
    const uint8_t* data_;
    size_t size_;
    size_t position_;
};

struct Tokenizer {
    std::vector<std::string> tokens;
    std::unordered_map<uint32_t, MergeRule> merges;
    int byte_ids[256];

    static uint32_t pair_key(uint16_t left, uint16_t right) {
        return (static_cast<uint32_t>(left) << 16) | right;
    }

    void load(const char* path) {
        BinaryReader reader(path);
        load(reader);
    }

    void load(const uint8_t* data, size_t size) {
        BinaryReader reader(data, size);
        load(reader);
    }

    void load(BinaryReader& reader) {
        char magic[4];
        uint32_t vocab_size, merge_count;
        if (!reader.read(magic, sizeof(magic)) ||
            std::memcmp(magic, "TKB1", 4) != 0 ||
            !reader.read_u32(vocab_size) || !reader.read_u32(merge_count)) {
            throw std::runtime_error("Invalid tokenizer header");
        }
        tokens.resize(vocab_size);
        for (uint32_t i = 0; i < vocab_size; ++i) {
            uint16_t length;
            if (!reader.read_u16(length)) {
                throw std::runtime_error("Truncated tokenizer vocabulary");
            }
            tokens[i].resize(length);
            if (length && !reader.read(&tokens[i][0], length)) {
                throw std::runtime_error("Truncated tokenizer token");
            }
        }
        for (uint32_t rank = 0; rank < merge_count; ++rank) {
            uint16_t left, right, result;
            if (!reader.read_u16(left) || !reader.read_u16(right) ||
                !reader.read_u16(result)) {
                throw std::runtime_error("Truncated tokenizer merges");
            }
            merges[pair_key(left, right)] = MergeRule{rank, result};
        }

        for (int i = 0; i < 256; ++i) byte_ids[i] = -1;
        for (uint32_t id = 0; id < tokens.size(); ++id) {
            if (tokens[id].size() == 1) {
                byte_ids[static_cast<uint8_t>(tokens[id][0])] = static_cast<int>(id);
            }
        }
        for (int i = 0; i < 256; ++i) {
            if (byte_ids[i] < 0) throw std::runtime_error("Tokenizer is missing byte tokens");
        }
    }

    static bool is_space(unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }

    static int char_class(unsigned char c) {
        if (is_space(c)) return 0;
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80) return 1;
        if (c >= '0' && c <= '9') return 2;
        return 3;
    }

    static size_t utf8_char_width(unsigned char c) {
        if ((c & 0x80) == 0) return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 1;
    }

    bool contraction(const std::string& text, size_t pos, size_t& length) const {
        static const char* suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
        for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
            size_t n = std::strlen(suffixes[i]);
            if (pos + n > text.size()) continue;
            bool match = true;
            for (size_t j = 0; j < n; ++j) {
                char a = text[pos + j];
                char b = suffixes[i][j];
                if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
                if (a != b) match = false;
            }
            if (match) {
                length = n;
                return true;
            }
        }
        return false;
    }

    void encode_piece(const std::string& piece, std::vector<int>& output) const {
        std::vector<uint16_t> ids;
        ids.reserve(piece.size());
        for (size_t i = 0; i < piece.size(); ++i) {
            ids.push_back(static_cast<uint16_t>(byte_ids[static_cast<uint8_t>(piece[i])]));
        }
        while (ids.size() > 1) {
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            size_t best_at = ids.size();
            uint16_t best_result = 0;
            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                std::unordered_map<uint32_t, MergeRule>::const_iterator rule =
                    merges.find(pair_key(ids[i], ids[i + 1]));
                if (rule != merges.end() && rule->second.rank < best_rank) {
                    best_rank = rule->second.rank;
                    best_at = i;
                    best_result = rule->second.result;
                }
            }
            if (best_at == ids.size()) break;
            ids[best_at] = best_result;
            ids.erase(ids.begin() + best_at + 1);
        }
        for (size_t i = 0; i < ids.size(); ++i) output.push_back(ids[i]);
    }

    std::vector<int> encode(const std::string& text) const {
        std::vector<int> output;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t start = pos;
            if (text[pos] == '\'') {
                size_t length = 0;
                if (contraction(text, pos, length)) {
                    encode_piece(text.substr(pos, length), output);
                    pos += length;
                    continue;
                }
            }

            if (is_space(static_cast<unsigned char>(text[pos]))) {
                size_t run_end = pos;
                while (run_end < text.size() && is_space(static_cast<unsigned char>(text[run_end]))) {
                    ++run_end;
                }
                if (run_end == text.size()) {
                    encode_piece(text.substr(pos), output);
                    break;
                }
                if (run_end - pos > 1) {
                    encode_piece(text.substr(pos, run_end - pos - 1), output);
                    pos = run_end - 1;
                }
                start = pos++;
                if (pos == text.size()) {
                    encode_piece(text.substr(start), output);
                    break;
                }
            }

            int kind = char_class(static_cast<unsigned char>(text[pos]));
            if (kind == 0) {
                encode_piece(text.substr(pos, 1), output);
                ++pos;
                continue;
            }
            size_t end = pos;
            while (end < text.size() &&
                   char_class(static_cast<unsigned char>(text[end])) == kind) {
                size_t width = utf8_char_width(static_cast<unsigned char>(text[end]));
                end += std::min(width, text.size() - end);
            }
            if (end == pos) ++end;
            encode_piece(text.substr(start, end - start), output);
            pos = end;
        }
        return output;
    }

    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] >= 0 && static_cast<size_t>(ids[i]) < tokens.size()) {
                result += tokens[ids[i]];
            }
        }
        return result;
    }

};

struct Model {
    uint32_t layers;
    uint32_t vocab_size;
    uint32_t context;
    uint32_t hidden;
    uint32_t heads;
    std::unordered_map<std::string, Tensor> tensors;
#ifndef GEKKO
    std::vector<uint8_t> owned_bytes;

    void load(const char* path) {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) throw std::runtime_error(std::string("Cannot open ") + path);
        const std::streamsize size = input.tellg();
        if (size <= 0) throw std::runtime_error("Weights file is empty");
        input.seekg(0, std::ios::beg);
        owned_bytes.resize(static_cast<size_t>(size));
        if (!input.read(reinterpret_cast<char*>(&owned_bytes[0]), size)) {
            throw std::runtime_error("Could not read weights file");
        }
        load(&owned_bytes[0], owned_bytes.size());
    }
#endif

    void load(const uint8_t* data, size_t size) {
        BinaryReader reader(data, size);
        tensors.clear();
        char magic[4];
        uint32_t version, tensor_count;
        if (!reader.read(magic, sizeof(magic)) ||
            std::memcmp(magic, "SWM1", 4) != 0 ||
            !reader.read_u32(version) || version != 1 ||
            !reader.read_u32(layers) || !reader.read_u32(vocab_size) ||
            !reader.read_u32(context) || !reader.read_u32(hidden) ||
            !reader.read_u32(heads) || !reader.read_u32(tensor_count)) {
            throw std::runtime_error("Invalid weights header");
        }

        for (uint32_t i = 0; i < tensor_count; ++i) {
            uint16_t name_length;
            uint8_t type, rank;
            if (!reader.read_u16(name_length) ||
                !reader.read_u8(type) || !reader.read_u8(rank)) {
                throw std::runtime_error("Truncated tensor record");
            }
            std::string name(name_length, '\0');
            if (name_length && !reader.read(&name[0], name_length)) {
                throw std::runtime_error("Truncated tensor name");
            }
            Tensor tensor;
            tensor.type = type;
            for (uint8_t d = 0; d < rank; ++d) {
                uint32_t dimension;
                if (!reader.read_u32(dimension)) {
                    throw std::runtime_error("Truncated tensor shape");
                }
                tensor.shape.push_back(dimension);
            }
            uint32_t payload_size;
            if (!reader.read_f32(tensor.scale) || !reader.read_u32(payload_size)) {
                throw std::runtime_error("Truncated tensor metadata");
            }
            if (type != 1 && type != 2) throw std::runtime_error("Unknown tensor type");
            size_t element_count = 1;
            for (size_t d = 0; d < tensor.shape.size(); ++d) {
                if (tensor.shape[d] &&
                    element_count > std::numeric_limits<size_t>::max() / tensor.shape[d]) {
                    throw std::runtime_error("Tensor shape overflow");
                }
                element_count *= tensor.shape[d];
            }
            const size_t expected_size = element_count * (type == 1 ? 4 : 1);
            if (payload_size != expected_size) {
                throw std::runtime_error("Tensor payload size does not match shape");
            }
            if (!reader.read_view(tensor.data, payload_size)) {
                throw std::runtime_error("Truncated tensor payload");
            }
            tensor.byte_count = payload_size;
            tensors[name] = tensor;
        }
        if (tensors.size() != tensor_count) throw std::runtime_error("Duplicate tensor name");
        require("transformer.wte.weight");
        require("transformer.wpe.weight");
    }

    const Tensor& get(const std::string& name) const {
        std::unordered_map<std::string, Tensor>::const_iterator it = tensors.find(name);
        if (it == tensors.end()) throw std::runtime_error("Missing tensor: " + name);
        return it->second;
    }

    void require(const std::string& name) const { (void)get(name); }

private:
};

void layer_norm(const std::vector<float>& input, std::vector<float>& output,
                const Tensor& weight, const Tensor& bias) {
    const size_t n = input.size();
    double mean = 0.0;
    for (size_t i = 0; i < n; ++i) mean += input[i];
    mean /= n;
    double variance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = input[i] - mean;
        variance += d * d;
    }
    variance /= n;
    const float inv = 1.0f / std::sqrt(static_cast<float>(variance) + 1e-5f);
    output.resize(n);
    for (size_t i = 0; i < n; ++i) {
        output[i] = (input[i] - static_cast<float>(mean)) * inv * weight.at(i) + bias.at(i);
    }
}

void linear(const Tensor& weight, const Tensor& bias,
            const std::vector<float>& input, std::vector<float>& output) {
    const size_t in = weight.shape[0];
    const size_t out = weight.shape[1];
    if (input.size() != in || bias.elements() != out) {
        throw std::runtime_error("Linear tensor shape mismatch");
    }
    output.resize(out);
    for (size_t j = 0; j < out; ++j) output[j] = bias.at(j);
    for (size_t i = 0; i < in; ++i) {
        const float x = input[i];
        const size_t row = i * out;
        for (size_t j = 0; j < out; ++j) output[j] += x * weight.at(row + j);
    }
}

struct Cache {
    std::vector<std::vector<float> > keys;
    std::vector<std::vector<float> > values;
};

int model_step(const Model& model, Cache& cache, int token, uint32_t position) {
    const uint32_t d_model = model.hidden;
    const uint32_t head_dim = d_model / model.heads;
    const Tensor& token_embedding = model.get("transformer.wte.weight");
    const Tensor& position_embedding = model.get("transformer.wpe.weight");
    std::vector<float> x(d_model);
    for (uint32_t i = 0; i < d_model; ++i) {
        x[i] = token_embedding.at(static_cast<size_t>(token) * d_model + i) +
               position_embedding.at(static_cast<size_t>(position) * d_model + i);
    }

    if (cache.keys.empty()) {
        cache.keys.resize(model.layers);
        cache.values.resize(model.layers);
        for (uint32_t l = 0; l < model.layers; ++l) {
            cache.keys[l].reserve(static_cast<size_t>(model.context) * d_model);
            cache.values[l].reserve(static_cast<size_t>(model.context) * d_model);
        }
    }

    for (uint32_t layer = 0; layer < model.layers; ++layer) {
        const std::string prefix = "transformer.h." + std::to_string(layer) + ".";
        std::vector<float> norm1;
        layer_norm(x, norm1, model.get(prefix + "ln_1.weight"), model.get(prefix + "ln_1.bias"));

        std::vector<float> qkv;
        linear(model.get(prefix + "attn.c_attn.weight"),
               model.get(prefix + "attn.c_attn.bias"), norm1, qkv);
        std::vector<float>& keys = cache.keys[layer];
        std::vector<float>& values = cache.values[layer];
        for (uint32_t i = 0; i < d_model; ++i) {
            keys.push_back(qkv[d_model + i]);
            values.push_back(qkv[2 * d_model + i]);
        }

        std::vector<float> attended(d_model, 0.0f);
        std::vector<float> scores(position + 1);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        for (uint32_t h = 0; h < model.heads; ++h) {
            const uint32_t base = h * head_dim;
            float maximum = -std::numeric_limits<float>::infinity();
            for (uint32_t t = 0; t <= position; ++t) {
                double dot = 0.0;
                const size_t cache_base = static_cast<size_t>(t) * d_model + base;
                for (uint32_t j = 0; j < head_dim; ++j) {
                    dot += qkv[base + j] * keys[cache_base + j];
                }
                scores[t] = static_cast<float>(dot) * scale;
                if (scores[t] > maximum) maximum = scores[t];
            }
            double denominator = 0.0;
            for (uint32_t t = 0; t <= position; ++t) {
                scores[t] = std::exp(scores[t] - maximum);
                denominator += scores[t];
            }
            const float inv_denominator = 1.0f / static_cast<float>(denominator);
            for (uint32_t t = 0; t <= position; ++t) {
                const float probability = scores[t] * inv_denominator;
                const size_t value_base = static_cast<size_t>(t) * d_model + base;
                for (uint32_t j = 0; j < head_dim; ++j) {
                    attended[base + j] += probability * values[value_base + j];
                }
            }
        }

        std::vector<float> projected;
        linear(model.get(prefix + "attn.c_proj.weight"),
               model.get(prefix + "attn.c_proj.bias"), attended, projected);
        for (uint32_t i = 0; i < d_model; ++i) x[i] += projected[i];

        std::vector<float> norm2;
        layer_norm(x, norm2, model.get(prefix + "ln_2.weight"), model.get(prefix + "ln_2.bias"));
        std::vector<float> mlp;
        linear(model.get(prefix + "mlp.c_fc.weight"),
               model.get(prefix + "mlp.c_fc.bias"), norm2, mlp);
        const float gelu_scale = std::sqrt(2.0f / 3.14159265358979323846f);
        for (size_t i = 0; i < mlp.size(); ++i) {
            const float value = mlp[i];
            mlp[i] = 0.5f * value *
                     (1.0f + std::tanh(gelu_scale * (value + 0.044715f * value * value * value)));
        }
        std::vector<float> mlp_projected;
        linear(model.get(prefix + "mlp.c_proj.weight"),
               model.get(prefix + "mlp.c_proj.bias"), mlp, mlp_projected);
        for (uint32_t i = 0; i < d_model; ++i) x[i] += mlp_projected[i];
    }

    std::vector<float> final;
    layer_norm(x, final, model.get("transformer.ln_f.weight"),
               model.get("transformer.ln_f.bias"));
    int best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t token_id = 0; token_id < model.vocab_size; ++token_id) {
        double logit = 0.0;
        const size_t base = static_cast<size_t>(token_id) * d_model;
        for (uint32_t i = 0; i < d_model; ++i) {
            logit += final[i] * token_embedding.at(base + i);
        }
        if (logit > best_logit) {
            best_logit = static_cast<float>(logit);
            best_token = static_cast<int>(token_id);
        }
    }
    return best_token;
}

#ifndef GEKKO
std::string read_prompt(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return "Once upon a time, there was a little cat named Lily.";
    std::string text;
    std::getline(input, text, '\0');
    return text;
}
#endif

std::string generate(const Model& model, const Tokenizer& tokenizer,
                     const std::string& prompt_text, int max_new_tokens) {
    std::vector<int> prompt = tokenizer.encode(prompt_text);
    if (prompt.empty()) prompt.push_back(0);
    if (prompt.size() > model.context) {
        prompt.erase(prompt.begin(), prompt.end() - model.context);
    }

    Cache cache;
    std::vector<int> generated;
    int next = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        next = model_step(model, cache, prompt[i], static_cast<uint32_t>(i));
    }
    for (int i = 0; i < max_new_tokens && prompt.size() + generated.size() < model.context; ++i) {
        if (next == 0) break;
        generated.push_back(next);
        next = model_step(model, cache, next,
                          static_cast<uint32_t>(prompt.size() + generated.size() - 1));
    }
    return tokenizer.decode(generated);
}

#ifndef GEKKO
void run(const char* weights_path, const char* tokenizer_path,
         const char* prompt_path, int max_new_tokens) {
    Model model;
    Tokenizer tokenizer;
    model.load(weights_path);
    tokenizer.load(tokenizer_path);
    std::cout << generate(model, tokenizer, read_prompt(prompt_path), max_new_tokens)
              << std::endl;
}
#endif

}  // namespace

#ifdef GEKKO
extern "C" {
extern const uint8_t story_weights_start[];
extern const uint8_t story_weights_end[];
extern const uint8_t story_tokenizer_start[];
extern const uint8_t story_tokenizer_end[];
}

namespace {

std::string prompt_text;
const size_t kMaxPromptLength = 256;
const char* kKeyboardRows[] = {
    "1234567890",
    "QWERTYUIOP",
    "ASDFGHJKL ",
    "ZXCVBNM.,?!"
};
int key_row = 0;
int key_column = 0;

void on_usb_key(char symbol) {
    if (symbol == '\b' || symbol == 0x7f) {
        if (!prompt_text.empty()) prompt_text.erase(prompt_text.size() - 1);
    } else if (symbol == '\r' || symbol == '\n') {
        return;
    } else if (symbol >= 32 && symbol <= 126 &&
               prompt_text.size() < kMaxPromptLength) {
        prompt_text += symbol;
    }
}

void move_cursor(int row_delta, int column_delta) {
    key_row = (key_row + row_delta + 4) % 4;
    key_column = (key_column + column_delta + 10) % 10;
    for (int attempts = 0; attempts < 40 &&
         kKeyboardRows[key_row][key_column] == ' '; ++attempts) {
        key_column = (key_column + (column_delta < 0 ? 9 : 1)) % 10;
    }
}

void draw_input_screen(bool usb_keyboard) {
    consoleClear();
    std::printf("Wii Ran AI - type a story prompt\n\n");
    std::printf("Prompt (max %u chars):\n%s_\n\n",
                static_cast<unsigned>(kMaxPromptLength), prompt_text.c_str());
    if (usb_keyboard) {
        std::printf("USB keyboard connected: type directly\n");
    } else {
        std::printf("USB keyboard not detected; use Wii Remote\n");
    }
    std::printf("\n");
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 10; ++column) {
            const char key = kKeyboardRows[row][column];
            if (key == ' ') {
                std::printf("    ");
            } else if (row == key_row && column == key_column) {
                std::printf("[%c] ", key);
            } else {
                std::printf(" %c  ", key);
            }
        }
        std::printf("\n");
    }
    std::printf("\nRemote: D-pad move, A type, B delete\n");
    std::printf("Minus: space   Plus: generate   Home: quit\n");
}

}  // namespace

int main() {
    VIDEO_Init();
    WPAD_Init();
    GXRModeObj* mode = VIDEO_GetPreferredMode(NULL);
    void* framebuffer = MEM_K0_TO_K1(SYS_AllocateFramebuffer(mode));
    console_init(framebuffer, 20, 20, mode->fbWidth, mode->xfbHeight,
                 mode->fbWidth * VI_DISPLAY_PIX_SZ);
    VIDEO_Configure(mode);
    VIDEO_SetNextFramebuffer(framebuffer);
    VIDEO_SetBlack(FALSE);
    VIDEO_Flush();
    VIDEO_WaitVSync();
    if (mode->viTVMode & VI_NON_INTERLACE) VIDEO_WaitVSync();
    std::ios::sync_with_stdio(false);
    std::printf("\x1b[2;0HStoryBananaMind Wii demo\n");
    try {
        std::printf("Loading model...\n");
        Model model;
        Tokenizer tokenizer;
        model.load(story_weights_start,
                   static_cast<size_t>(story_weights_end - story_weights_start));
        tokenizer.load(story_tokenizer_start,
                       static_cast<size_t>(story_tokenizer_end - story_tokenizer_start));

        const bool keyboard_ready = KEYBOARD_Init(on_usb_key) >= 0;
        int previous_buttons = 0;
        std::string previous_prompt;
        while (true) {
            draw_input_screen(keyboard_ready && USBKeyboard_IsConnected());
            bool generate_story = false;
            while (!generate_story) {
                WPAD_ScanPads();
                const int buttons = WPAD_ButtonsDown(0);
                if (buttons & WPAD_BUTTON_HOME) {
                    if (keyboard_ready) KEYBOARD_Deinit();
                    return 0;
                }
                if (buttons & WPAD_BUTTON_LEFT) move_cursor(0, -1);
                if (buttons & WPAD_BUTTON_RIGHT) move_cursor(0, 1);
                if (buttons & WPAD_BUTTON_UP) move_cursor(-1, 0);
                if (buttons & WPAD_BUTTON_DOWN) move_cursor(1, 0);
                if (buttons & WPAD_BUTTON_A) {
                    char selected = kKeyboardRows[key_row][key_column];
                    if (selected && prompt_text.size() < kMaxPromptLength) {
                        prompt_text += selected;
                    }
                }
                if ((buttons & WPAD_BUTTON_B) && !prompt_text.empty()) {
                    prompt_text.erase(prompt_text.size() - 1);
                }
                if ((buttons & WPAD_BUTTON_MINUS) &&
                    prompt_text.size() < kMaxPromptLength) {
                    prompt_text += ' ';
                }
                if (buttons & WPAD_BUTTON_PLUS) generate_story = true;
                if (buttons != previous_buttons || prompt_text != previous_prompt) {
                    draw_input_screen(keyboard_ready && USBKeyboard_IsConnected());
                }
                previous_buttons = buttons;
                previous_prompt = prompt_text;
                VIDEO_WaitVSync();
            }

            draw_input_screen(keyboard_ready && USBKeyboard_IsConnected());
            std::printf("\n\nGenerating...\n");
            const std::string continuation = generate(model, tokenizer, prompt_text, 80);
            std::printf("\n\n%s\n\n", continuation.c_str());
            std::printf("Press A or Plus to enter another prompt, Home to quit.\n");
            while (true) {
                WPAD_ScanPads();
                const int buttons = WPAD_ButtonsDown(0);
                if (buttons & WPAD_BUTTON_HOME) {
                    if (keyboard_ready) KEYBOARD_Deinit();
                    return 0;
                }
                if (buttons & (WPAD_BUTTON_A | WPAD_BUTTON_PLUS)) {
                    prompt_text.clear();
                    break;
                }
                VIDEO_WaitVSync();
            }
        }
    } catch (const std::exception& e) {
        std::printf("Error: %s\n", e.what());
    }
    std::printf("\n\nPress HOME to exit.\n");
    while (1) VIDEO_WaitVSync();
    return 0;
}
#else
int main(int argc, char** argv) {
    const char* weights = argc > 1 ? argv[1] : "story-wii-weights.bin";
    const char* tokenizer = argc > 2 ? argv[2] : "story-wii-tokenizer.bin";
    const char* prompt = argc > 3 ? argv[3] : "prompt.txt";
    const int max_new_tokens = argc > 4 ? std::atoi(argv[4]) : 32;
    try {
        run(weights, tokenizer, prompt, max_new_tokens);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
#endif
