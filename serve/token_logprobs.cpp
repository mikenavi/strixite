#include "serve/token_logprobs.hpp"
#include "serve/unicode.hpp"
#include "common/check.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace strix {
TokenLogprob token_logprob(const LogitRows &rows, int64_t row, int32_t id, int top, const Tokenizer &tok) {
    STRIX_CHECK(rows.cands == 0 && row >= 0 && row < rows.rows && rows.row >= tok.size(), "logprobs require full target logits");
    const auto &full = rows.unmasked.empty() ? rows.full : rows.unmasked;
    STRIX_CHECK(full.size() == (size_t)(rows.rows * rows.row) && id >= 0 && id < tok.size() && top >= 0 && top <= 20,
                "invalid logprobs row/token/count");
    const float *x = full.data() + row * rows.row;
    double max = -std::numeric_limits<double>::infinity();
    for (int32_t i=0; i<tok.size(); ++i) { STRIX_CHECK(!std::isnan(x[i]), "NaN target logit"); max=std::max(max,(double)x[i]); }
    STRIX_CHECK(std::isfinite(max), "no finite target logit maximum");
    double sum=0; for (int32_t i=0; i<tok.size(); ++i) sum+=std::exp((double)x[i]-max);
    const double logz=std::log(sum);
    TokenLogprob out; out.id=id; out.bytes=tok.token_bytes(id); out.logprob=(double)x[id]-max-logz;
    STRIX_CHECK(std::isfinite(out.logprob), "non-finite selected token logprob");
    if (top) {
        std::vector<int32_t> order(tok.size()); std::iota(order.begin(),order.end(),0);
        const int n=std::min<int>(top,tok.size());
        std::partial_sort(order.begin(),order.begin()+n,order.end(),[&](int32_t a,int32_t b){return x[a]>x[b] || (x[a]==x[b] && a<b);});
        for (int i=0; i<n; ++i) if (std::isfinite(x[order[i]])) out.top.emplace_back(order[i],(double)x[order[i]]-max-logz);
    }
    return out;
}
namespace {
json::Value item(int32_t id,double lp,const Tokenizer &tok,bool as_ids) {
    const std::string &raw=tok.token_bytes(id);
    auto value=json::Value::object(); value.set("token",json::Value::string(as_ids ? "token_id:"+std::to_string(id) : unicode::replace_invalid_utf8(raw,nullptr)));
    value.set("token_id",json::Value::integer(id)); value.set("logprob",json::Value::number(lp));
    auto bytes=json::Value::array(); for(unsigned char c:raw)bytes.push(json::Value::integer(c));value.set("bytes",std::move(bytes));return value;
}
}
json::Value chat_token_logprob(const TokenLogprob &p,const Tokenizer &tok,bool as_ids) {
    auto out=item(p.id,p.logprob,tok,as_ids); auto top=json::Value::array();
    for(const auto &[id,lp]:p.top)top.push(item(id,lp,tok,as_ids));out.set("top_logprobs",std::move(top));return out;
}
json::Value chat_logprobs(const std::vector<TokenLogprob> &p,const Tokenizer &tok,bool as_ids) {
    auto out=json::Value::object(), content=json::Value::array(), reasoning=json::Value::array();
    for(const auto &v:p)(v.reasoning ? reasoning : content).push(chat_token_logprob(v,tok,as_ids));
    out.set("content",std::move(content));out.set("refusal",json::Value());
    if(!reasoning.as_array("reasoning").empty())out.set("reasoning_content",std::move(reasoning));return out;
}
json::Value legacy_logprobs(const std::vector<TokenLogprob> &p,const Tokenizer &tok,bool as_ids) {
    auto out=json::Value::object(), tokens=json::Value::array(), values=json::Value::array(), alternatives=json::Value::array(), offsets=json::Value::array(), ids=json::Value::array();
    for(const auto &v:p) {
        tokens.push(json::Value::string(as_ids ? "token_id:"+std::to_string(v.id) : unicode::replace_invalid_utf8(v.bytes,nullptr)));values.push(json::Value::number(v.logprob));
        auto top=json::Value::object();for(const auto &[id,lp]:v.top)top.set(as_ids ? "token_id:"+std::to_string(id) : unicode::replace_invalid_utf8(tok.token_bytes(id),nullptr),json::Value::number(lp));
        alternatives.push(std::move(top));offsets.push(json::Value::integer(v.text_offset));ids.push(json::Value::integer(v.id));
    }
    out.set("tokens",std::move(tokens));out.set("token_logprobs",std::move(values));out.set("top_logprobs",std::move(alternatives));out.set("text_offset",std::move(offsets));out.set("token_ids",std::move(ids));return out;
}
int64_t complete_utf8_characters(const std::string &s) {
    int64_t count=0;size_t at=0;
    while(at<s.size()) { unsigned char c=s[at];size_t n=c<0x80?1:(c>=0xc2&&c<=0xdf?2:(c>=0xe0&&c<=0xef?3:(c>=0xf0&&c<=0xf4?4:1)));
        if(at+n>s.size())break;bool ok=true;for(size_t i=1;i<n;++i)if(((unsigned char)s[at+i]&0xc0)!=0x80)ok=false;
        if(ok && n>=3) {
            const unsigned char second=s[at+1];
            if((c==0xe0 && second<0xa0) || (c==0xed && second>=0xa0) || (c==0xf0 && second<0x90) || (c==0xf4 && second>=0x90))ok=false;
        }
        at+=ok?n:1;++count; }
    return count;
}
}
