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
    std::vector<uint8_t> special;
    std::vector<uint32_t> special_ids;
    bool digit_aware;
    int eos_id;
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
            (std::memcmp(magic, "TKB1", 4) != 0 &&
             std::memcmp(magic, "TKB2", 4) != 0) ||
            !reader.read_u32(vocab_size) || !reader.read_u32(merge_count)) {
            throw std::runtime_error("Invalid tokenizer header");
        }
        digit_aware = std::memcmp(magic, "TKB2", 4) == 0;
        uint32_t special_count = 0;
        if (digit_aware) {
            uint32_t eos;
            if (!reader.read_u32(special_count) || !reader.read_u32(eos)) {
                throw std::runtime_error("Truncated tokenizer metadata");
            }
            eos_id = static_cast<int>(eos);
        } else {
            eos_id = 0;
        }
        tokens.resize(vocab_size);
        special.assign(vocab_size, 0);
        for (uint32_t i = 0; i < vocab_size; ++i) {
            uint16_t length;
            uint8_t is_special = 0;
            if ((digit_aware && !reader.read_u8(is_special)) ||
                !reader.read_u16(length)) {
                throw std::runtime_error("Truncated tokenizer vocabulary");
            }
            special[i] = is_special;
            if (is_special) special_ids.push_back(i);
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
        if (special_count > vocab_size) {
            throw std::runtime_error("Invalid special-token count");
        }

        for (int i = 0; i < 256; ++i) byte_ids[i] = -1;
        for (uint32_t id = 0; id < tokens.size(); ++id) {
            if (!special[id] && tokens[id].size() == 1) {
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
            size_t special_at = text.size();
            uint32_t special_id = 0;
            if (digit_aware) {
                for (size_t s = 0; s < special_ids.size(); ++s) {
                    const uint32_t id = special_ids[s];
                    if (!tokens[id].empty() &&
                        pos + tokens[id].size() <= text.size() &&
                        text.compare(pos, tokens[id].size(), tokens[id]) == 0 &&
                        tokens[id].size() < special_at - pos) {
                        special_at = pos + tokens[id].size();
                        special_id = id;
                    }
                }
            }
            if (special_at != text.size()) {
                output.push_back(static_cast<int>(special_id));
                pos = special_at;
                continue;
            }
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
                if (digit_aware && kind == 2) break;
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
                if (!special[ids[i]]) result += tokens[ids[i]];
            }
        }
        return result;
    }

};

struct Model {
    uint32_t architecture;
    uint32_t layers;
    uint32_t vocab_size;
    uint32_t context;
    uint32_t hidden;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t intermediate;
    uint32_t head_dim;
    float rope_theta;
    float rms_eps;
    float embedding_scale;
    int eos_id;
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
        if (!reader.read(magic, sizeof(magic))) {
            throw std::runtime_error("Invalid weights header");
        }
        if (std::memcmp(magic, "SWM1", 4) == 0) {
            architecture = 1;
            if (!reader.read_u32(version) || version != 1 ||
                !reader.read_u32(layers) || !reader.read_u32(vocab_size) ||
                !reader.read_u32(context) || !reader.read_u32(hidden) ||
                !reader.read_u32(heads) || !reader.read_u32(tensor_count)) {
                throw std::runtime_error("Invalid story weights header");
            }
            kv_heads = heads;
            intermediate = hidden * 4;
            head_dim = hidden / heads;
            rope_theta = 0.0f;
            rms_eps = 1e-5f;
            embedding_scale = 1.0f;
            eos_id = 0;
        } else if (std::memcmp(magic, "BNM1", 4) == 0) {
            architecture = 2;
            uint32_t bos_id, eos_id_value;
            if (!reader.read_u32(version) || version != 1 ||
                !reader.read_u32(layers) || !reader.read_u32(vocab_size) ||
                !reader.read_u32(context) || !reader.read_u32(hidden) ||
                !reader.read_u32(heads) || !reader.read_u32(kv_heads) ||
                !reader.read_u32(intermediate) || !reader.read_u32(head_dim) ||
                !reader.read_f32(rope_theta) || !reader.read_f32(rms_eps) ||
                !reader.read_u32(bos_id) || !reader.read_u32(eos_id_value) ||
                !reader.read_u32(tensor_count)) {
                throw std::runtime_error("Invalid BananaMind weights header");
            }
            eos_id = static_cast<int>(eos_id_value);
            embedding_scale = std::sqrt(static_cast<float>(hidden));
        } else {
            throw std::runtime_error("Unknown weights file format");
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
        if (architecture == 1) {
            require("transformer.wpe.weight");
        } else {
            require("transformer.ln_f.weight");
        }
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

void model_step(const Model& model, Cache& cache, int token, uint32_t position,
                std::vector<float>& logits) {
    if (model.architecture == 2) {
        const uint32_t d_model = model.hidden;
        const uint32_t kv_width = model.kv_heads * model.head_dim;
        const Tensor& token_embedding = model.get("transformer.wte.weight");
        std::vector<float> x(d_model);
        for (uint32_t i = 0; i < d_model; ++i) {
            x[i] = token_embedding.at(static_cast<size_t>(token) * d_model + i) *
                   model.embedding_scale;
        }

        if (cache.keys.empty()) {
            cache.keys.resize(model.layers);
            cache.values.resize(model.layers);
            for (uint32_t l = 0; l < model.layers; ++l) {
                cache.keys[l].reserve(static_cast<size_t>(model.context) * kv_width);
                cache.values[l].reserve(static_cast<size_t>(model.context) * kv_width);
            }
        }

        const auto normalize = [&model](
                                   const std::vector<float>& input,
                                   const Tensor& weight,
                                   std::vector<float>& output) {
            double squares = 0.0;
            for (size_t i = 0; i < input.size(); ++i) squares += input[i] * input[i];
            const float scale = 1.0f / std::sqrt(
                static_cast<float>(squares / input.size()) + model.rms_eps);
            output.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = input[i] * scale * weight.at(i);
            }
        };
        const auto project = [](const Tensor& weight,
                                const std::vector<float>& input,
                                std::vector<float>& output) {
            const size_t in = weight.shape[0];
            const size_t out = weight.shape[1];
            if (input.size() != in) throw std::runtime_error("Projection tensor shape mismatch");
            output.assign(out, 0.0f);
            for (size_t i = 0; i < in; ++i) {
                const float value = input[i];
                const size_t row = i * out;
                for (size_t j = 0; j < out; ++j) {
                    output[j] += value * weight.at(row + j);
                }
            }
        };

        std::vector<float> rope_cos(model.head_dim / 2);
        std::vector<float> rope_sin(model.head_dim / 2);
        for (uint32_t j = 0; j < model.head_dim; j += 2) {
            const float angle = static_cast<float>(position) /
                std::pow(model.rope_theta, static_cast<float>(j) / model.head_dim);
            rope_cos[j / 2] = std::cos(angle);
            rope_sin[j / 2] = std::sin(angle);
        }

        for (uint32_t layer = 0; layer < model.layers; ++layer) {
            const std::string prefix = "transformer.h." + std::to_string(layer) + ".";
            std::vector<float> norm1;
            normalize(x, model.get(prefix + "ln_1.weight"), norm1);
            std::vector<float> q, k, v;
            project(model.get(prefix + "attn.q_proj.weight"), norm1, q);
            project(model.get(prefix + "attn.k_proj.weight"), norm1, k);
            project(model.get(prefix + "attn.v_proj.weight"), norm1, v);

            for (uint32_t h = 0; h < model.heads; ++h) {
                const size_t base = static_cast<size_t>(h) * model.head_dim;
                double squares = 0.0;
                for (uint32_t j = 0; j < model.head_dim; ++j) {
                    squares += q[base + j] * q[base + j];
                }
                const float scale = 1.0f / std::sqrt(
                    static_cast<float>(squares / model.head_dim) + model.rms_eps);
                const Tensor& q_norm = model.get(prefix + "attn.q_norm.weight");
                for (uint32_t j = 0; j < model.head_dim; ++j) {
                    q[base + j] *= scale * q_norm.at(j);
                }
            }
            for (uint32_t h = 0; h < model.kv_heads; ++h) {
                const size_t base = static_cast<size_t>(h) * model.head_dim;
                double squares = 0.0;
                for (uint32_t j = 0; j < model.head_dim; ++j) {
                    squares += k[base + j] * k[base + j];
                }
                const float scale = 1.0f / std::sqrt(
                    static_cast<float>(squares / model.head_dim) + model.rms_eps);
                const Tensor& k_norm = model.get(prefix + "attn.k_norm.weight");
                for (uint32_t j = 0; j < model.head_dim; ++j) {
                    k[base + j] *= scale * k_norm.at(j);
                }
            }

            for (uint32_t h = 0; h < model.heads; ++h) {
                const size_t base = static_cast<size_t>(h) * model.head_dim;
                for (uint32_t j = 0; j < model.head_dim; j += 2) {
                    const float cosine = rope_cos[j / 2];
                    const float sine = rope_sin[j / 2];
                    const float even = q[base + j];
                    const float odd = q[base + j + 1];
                    q[base + j] = even * cosine - odd * sine;
                    q[base + j + 1] = even * sine + odd * cosine;
                }
            }
            for (uint32_t h = 0; h < model.kv_heads; ++h) {
                const size_t base = static_cast<size_t>(h) * model.head_dim;
                for (uint32_t j = 0; j < model.head_dim; j += 2) {
                    const float cosine = rope_cos[j / 2];
                    const float sine = rope_sin[j / 2];
                    const float even = k[base + j];
                    const float odd = k[base + j + 1];
                    k[base + j] = even * cosine - odd * sine;
                    k[base + j + 1] = even * sine + odd * cosine;
                }
            }

            std::vector<float>& keys = cache.keys[layer];
            std::vector<float>& values = cache.values[layer];
            keys.insert(keys.end(), k.begin(), k.end());
            values.insert(values.end(), v.begin(), v.end());
            std::vector<float> attended(d_model, 0.0f);
            std::vector<float> scores(position + 1);
            const float attention_scale = 1.0f /
                std::sqrt(static_cast<float>(model.head_dim));
            const uint32_t heads_per_kv = model.heads / model.kv_heads;
            for (uint32_t h = 0; h < model.heads; ++h) {
                const uint32_t kv_head = h / heads_per_kv;
                const size_t query_base = static_cast<size_t>(h) * model.head_dim;
                const size_t kv_base = static_cast<size_t>(kv_head) * model.head_dim;
                float maximum = -std::numeric_limits<float>::infinity();
                for (uint32_t t = 0; t <= position; ++t) {
                    double dot = 0.0;
                    const size_t cache_base = static_cast<size_t>(t) * kv_width + kv_base;
                    for (uint32_t j = 0; j < model.head_dim; ++j) {
                        dot += q[query_base + j] * keys[cache_base + j];
                    }
                    scores[t] = static_cast<float>(dot) * attention_scale;
                    maximum = std::max(maximum, scores[t]);
                }
                double denominator = 0.0;
                for (uint32_t t = 0; t <= position; ++t) {
                    scores[t] = std::exp(scores[t] - maximum);
                    denominator += scores[t];
                }
                for (uint32_t t = 0; t <= position; ++t) {
                    const float probability =
                        scores[t] / static_cast<float>(denominator);
                    const size_t value_base = static_cast<size_t>(t) * kv_width + kv_base;
                    for (uint32_t j = 0; j < model.head_dim; ++j) {
                        attended[query_base + j] +=
                            probability * values[value_base + j];
                    }
                }
            }
            std::vector<float> projected;
            project(model.get(prefix + "attn.o_proj.weight"), attended, projected);
            for (uint32_t i = 0; i < d_model; ++i) x[i] += projected[i];

            std::vector<float> norm2;
            normalize(x, model.get(prefix + "ln_2.weight"), norm2);
            std::vector<float> gate, up;
            project(model.get(prefix + "mlp.w_gate.weight"), norm2, gate);
            project(model.get(prefix + "mlp.w_up.weight"), norm2, up);
            for (size_t i = 0; i < gate.size(); ++i) {
                const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
                gate[i] = gate[i] * sigmoid * up[i];
            }
            std::vector<float> mlp;
            project(model.get(prefix + "mlp.w_down.weight"), gate, mlp);
            for (uint32_t i = 0; i < d_model; ++i) x[i] += mlp[i];
        }

        std::vector<float> final;
        normalize(x, model.get("transformer.ln_f.weight"), final);
        logits.resize(model.vocab_size);
        for (uint32_t token_id = 0; token_id < model.vocab_size; ++token_id) {
            float logit = 0.0f;
            const size_t base = static_cast<size_t>(token_id) * d_model;
            for (uint32_t i = 0; i < d_model; ++i) {
                logit += final[i] * token_embedding.at(base + i);
            }
            logits[token_id] = logit;
        }
        return;
    }

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
    logits.resize(model.vocab_size);
    for (uint32_t token_id = 0; token_id < model.vocab_size; ++token_id) {
        float logit = 0.0f;
        const size_t base = static_cast<size_t>(token_id) * d_model;
        for (uint32_t i = 0; i < d_model; ++i) {
            logit += final[i] * token_embedding.at(base + i);
        }
        logits[token_id] = logit;
    }
}

uint32_t next_random(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int sample_next_token(const std::vector<float>& logits,
                      const std::vector<int>& history,
                      size_t prompt_length,
                      uint32_t& random_state,
                      bool use_sampling) {
    std::vector<float> adjusted = logits;
    const float repetition_penalty = use_sampling ? 1.15f : 1.1f;
    for (size_t i = prompt_length; i < history.size(); ++i) {
        const int token = history[i];
        if (token >= 0 && static_cast<size_t>(token) < adjusted.size()) {
            float& score = adjusted[token];
            score = score < 0.0f ? score * repetition_penalty
                                 : score / repetition_penalty;
        }
    }

    if (use_sampling && history.size() >= 2) {
        const int previous_a = history[history.size() - 2];
        const int previous_b = history[history.size() - 1];
        for (size_t i = 0; i + 2 < history.size(); ++i) {
            if (history[i] == previous_a && history[i + 1] == previous_b) {
                const int repeated = history[i + 2];
                if (repeated >= 0 && static_cast<size_t>(repeated) < adjusted.size()) {
                    adjusted[repeated] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    if (!use_sampling) {
        return static_cast<int>(std::distance(
            adjusted.begin(), std::max_element(adjusted.begin(), adjusted.end())));
    }

    const size_t top_k = 16;
    std::vector<size_t> candidates;
    for (size_t i = 0; i < adjusted.size(); ++i) {
        if (!std::isfinite(adjusted[i])) continue;
        size_t insert_at = candidates.size();
        for (size_t j = 0; j < candidates.size(); ++j) {
            if (adjusted[i] > adjusted[candidates[j]]) {
                insert_at = j;
                break;
            }
        }
        if (insert_at < top_k) {
            candidates.insert(candidates.begin() + insert_at, i);
            if (candidates.size() > top_k) candidates.pop_back();
        } else if (candidates.size() < top_k) {
            candidates.push_back(i);
        }
    }
    if (candidates.empty()) return static_cast<int>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));

    const float temperature = 0.8f;
    const float maximum = adjusted[candidates[0]];
    std::vector<double> probabilities(candidates.size());
    double total = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp((adjusted[candidates[i]] - maximum) / temperature);
        total += probabilities[i];
    }
    const double sample = (next_random(random_state) / 4294967296.0) * total;
    double cumulative = 0.0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        cumulative += probabilities[i];
        if (sample < cumulative) return static_cast<int>(candidates[i]);
    }
    return static_cast<int>(candidates[0]);
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
    std::vector<int> history = prompt;
    std::vector<float> logits;
    uint32_t random_state = 0x9e3779b9u;
    for (size_t i = 0; i < prompt.size(); ++i) {
        random_state ^= static_cast<uint32_t>(prompt[i]) + 0x9e3779b9u +
                        (random_state << 6) + (random_state >> 2);
    }
    if (random_state == 0) random_state = 1;
    for (size_t i = 0; i < prompt.size(); ++i) {
        model_step(model, cache, prompt[i], static_cast<uint32_t>(i), logits);
    }
    for (int i = 0; i < max_new_tokens && prompt.size() + generated.size() < model.context; ++i) {
        const int next = sample_next_token(
            logits, history, prompt.size(), random_state, model.architecture == 1);
        if (next == model.eos_id) break;
        generated.push_back(next);
        history.push_back(next);
        model_step(model, cache, next,
                   static_cast<uint32_t>(prompt.size() + generated.size() - 1), logits);
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
extern const uint8_t chat_weights_start[];
extern const uint8_t chat_weights_end[];
extern const uint8_t chat_tokenizer_start[];
extern const uint8_t chat_tokenizer_end[];
}

namespace {

std::string prompt_text;
std::vector<std::pair<std::string, std::string> > chat_history;
const size_t kMaxPromptLength = 256;
const size_t kChatWiiContext = 384;
const size_t kChatMaxReply = 64;
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

void draw_model_menu(int selected) {
    consoleClear();
    std::printf("Wii Ran AI - select a model\n\n");
    std::printf("%s StoryBananaMind-7.5M (stories)\n",
                selected == 0 ? ">" : " ");
    std::printf("%s BananaMind 2 Nano Chat (assistant)\n\n",
                selected == 1 ? ">" : " ");
    std::printf("D-pad: select   A: choose   Home: quit\n");
}

std::string make_chat_prompt(const Tokenizer& tokenizer,
                             const std::string& user_message) {
    const std::string system_prompt =
        "You are BananaMind, a friendly and concise assistant. "
        "Answer directly, be honest when you are unsure, and keep explanations clear.";
    size_t first_turn = 0;
    std::string prompt;
    while (true) {
        prompt = "<|bos|><|system|>\n" + system_prompt + "\n";
        for (size_t i = first_turn; i < chat_history.size(); ++i) {
            prompt += "<|user|>\n" + chat_history[i].first +
                      "<|assistant|>\n" + chat_history[i].second +
                      "<|eos|>\n";
        }
        prompt += "<|user|>\n" + user_message + "<|assistant|>\n";
        if (tokenizer.encode(prompt).size() + kChatMaxReply <= kChatWiiContext) return prompt;
        if (first_turn >= chat_history.size()) {
            throw std::runtime_error("Message is too long for the Wii chat context");
        }
        ++first_turn;
    }
}

void draw_input_screen(bool usb_keyboard, bool chat_model) {
    consoleClear();
    std::printf(chat_model ? "BananaMind Chat - enter a message\n\n"
                           : "Wii Ran AI - type a story prompt\n\n");
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
    try {
        int selected_model = 0;
        while (true) {
            draw_model_menu(selected_model);
            WPAD_ScanPads();
            const int buttons = WPAD_ButtonsDown(0);
            if (buttons & WPAD_BUTTON_HOME) return 0;
            if (buttons & WPAD_BUTTON_UP) selected_model = (selected_model + 1) % 2;
            if (buttons & WPAD_BUTTON_DOWN) selected_model = (selected_model + 1) % 2;
            if (buttons & WPAD_BUTTON_A) break;
            VIDEO_WaitVSync();
        }
        const bool chat_model = selected_model == 1;
        std::printf("\nLoading %s...\n", chat_model
                    ? "BananaMind 2 Nano Chat" : "StoryBananaMind-7.5M");
        Model model;
        Tokenizer tokenizer;
        if (chat_model) {
            model.load(chat_weights_start,
                       static_cast<size_t>(chat_weights_end - chat_weights_start));
            tokenizer.load(chat_tokenizer_start,
                           static_cast<size_t>(chat_tokenizer_end - chat_tokenizer_start));
        } else {
            model.load(story_weights_start,
                       static_cast<size_t>(story_weights_end - story_weights_start));
            tokenizer.load(story_tokenizer_start,
                           static_cast<size_t>(story_tokenizer_end - story_tokenizer_start));
        }

        const bool keyboard_ready = KEYBOARD_Init(on_usb_key) >= 0;
        int previous_buttons = 0;
        std::string previous_prompt;
        while (true) {
            draw_input_screen(keyboard_ready && USBKeyboard_IsConnected(), chat_model);
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
                    draw_input_screen(keyboard_ready && USBKeyboard_IsConnected(), chat_model);
                }
                previous_buttons = buttons;
                previous_prompt = prompt_text;
                VIDEO_WaitVSync();
            }

            draw_input_screen(keyboard_ready && USBKeyboard_IsConnected(), chat_model);
            std::printf("\n\n%s...\n", chat_model ? "Thinking" : "Generating");
            std::string continuation;
            try {
                const std::string active_prompt = chat_model
                    ? make_chat_prompt(tokenizer, prompt_text) : prompt_text;
                continuation = generate(
                    model, tokenizer, active_prompt,
                    chat_model ? static_cast<int>(kChatMaxReply) : 80);
            } catch (const std::exception& e) {
                consoleClear();
                std::printf("Could not generate a reply:\n%s\n\n", e.what());
                std::printf("Edit the message and try again. Press HOME to quit.\n");
                VIDEO_WaitVSync();
                continue;
            }
            if (chat_model) {
                chat_history.push_back(std::make_pair(prompt_text, continuation));
                if (chat_history.size() > 8) chat_history.erase(chat_history.begin());
                std::printf("\nYou: %s\n\nAssistant: %s\n\n",
                            prompt_text.c_str(), continuation.c_str());
            } else {
                std::printf("\n\n%s\n\n", continuation.c_str());
            }
            std::printf("Press A or Plus to enter another %s, Home to quit.\n",
                        chat_model ? "message" : "prompt");
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
