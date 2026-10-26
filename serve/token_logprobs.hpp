#pragma once
#include "serve/json.hpp"
#include "serve/tokenizer.hpp"
#include "serve/sampler.hpp"
#include <string>
#include <vector>

namespace strix {
struct TokenLogprob {
    int32_t id = 0;
    double logprob = 0;
    std::string bytes;
    std::vector<std::pair<int32_t, double>> top;
    bool reasoning = false;
    int64_t text_offset = 0;
};
// Full target logits, before temperature/top-k/top-p and before token constraints.
TokenLogprob token_logprob(const LogitRows &rows, int64_t row, int32_t id, int top, const Tokenizer &tok);
json::Value chat_token_logprob(const TokenLogprob &p, const Tokenizer &tok, bool as_ids = false);
json::Value chat_logprobs(const std::vector<TokenLogprob> &p, const Tokenizer &tok, bool as_ids = false);
json::Value legacy_logprobs(const std::vector<TokenLogprob> &p, const Tokenizer &tok, bool as_ids = false);
// Count complete UTF-8 characters in a byte prefix; a token can end inside a character.
int64_t complete_utf8_characters(const std::string &bytes);
}
