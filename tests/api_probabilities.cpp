#include "serve/http_server.hpp"
#include "serve/openai.hpp"
#include "serve/token_logprobs.hpp"
#include "serve/log.hpp"
#include <httplib.h>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
using namespace strix;

namespace {
// Checked JSON access for test assertions only; production uses the native Value API.
struct J : json::Value {
    J(json::Value value) : json::Value(std::move(value)) {}
    static J parse(const std::string &s) { return json::Value::parse(s); }
    J at(std::string_view key) const { auto p=json::Value::find(key); if(!p)throw std::runtime_error("missing JSON key: "+std::string(key)); return *p; }
    std::optional<J> find(std::string_view key) const { auto p=json::Value::find(key); if(p)return J(*p);return {}; }
    std::vector<J> as_array() const { std::vector<J> out;for(const auto &v:json::Value::as_array("test"))out.emplace_back(v);return out; }
    int64_t as_int() const { return json::Value::as_int("test"); }
    double as_double() const { return json::Value::as_double("test"); }
};
int checks = 0;
void require(bool value, const std::string &why) { ++checks; if(!value) throw std::runtime_error(why); }
class FakeBackend : public LmBackend {
public:
    explicit FakeBackend(const Tokenizer &t): tok(t), target(t.encode("Москва")[0]), alternative(t.byte_token('A')) { snapshots.fill(-1); }
    int64_t capacity() const override { return 8192; }
    int64_t max_chunk() const override { return 4096; }
    int64_t logits_row() const override { return tok.size()+3; }
    int64_t pos() const override { return position; }
    void reset() override { position=0; snapshots.fill(-1); }
    std::vector<float> row(bool draft=false) const {
        std::vector<float> out(logits_row(),-1000);
        out[target]=draft ? 20 : 5; out[alternative]=4;
        if(draft && reject_drafts) {out[target]=0;out[alternative]=20;}
        out.back()=1000; // Padded vocabulary logits must never enter the denominator or top alternatives.
        if(draft)out.resize(tok.size());
        return out;
    }
    std::vector<float> forward(const std::vector<int32_t> &ids,bool logits) override { return forward(ids,(int64_t)(logits ? 1 : 0)); }
    std::vector<float> forward(const std::vector<int32_t> &ids,int64_t n) override {
        ++forwards; position+=ids.size(); std::vector<float> all;
        auto values=row();for(int64_t i=0;i<n;++i)all.insert(all.end(),values.begin(),values.end());return all;
    }
    bool has_mtp() const override {return true;}
    std::vector<float> forward_mtp(int32_t,int64_t) override {return row(true);}
    std::vector<float> forward_verify(const std::vector<int32_t> &ids,int64_t n) override {
        pending=position;auto all=forward(ids,n);
        if(verify_end>=0)for(int64_t i=0;i<n;++i) {
            auto at=all.begin()+i*logits_row();std::fill(at,at+logits_row(),-1000);
            at[verify_end]=5;at[alternative]=4;at[logits_row()-1]=1000;
        }
        return all;
    }
    void keep_verify() override {pending=-1;}
    void drop_verify() override {if(pending>=0)position=pending;pending=-1;}
    void save_snapshot(int slot) override {snapshots.at(slot)=position;}
    int64_t snapshot_pos(int slot) const override {return snapshots.at(slot);}
    bool can_restore_snapshot(int slot) const override {return snapshots.at(slot)>=0;}
    void restore_snapshot(int slot) override {position=snapshots.at(slot);}
    void export_snapshot(int,HostBuffer&,int64_t=0) override {throw std::runtime_error("unexpected disk export");}
    uint64_t snapshot_bytes(int,int64_t=0) const override {return 0;}
    void import_state(const HostBuffer&,int64_t,int64_t=0) override {throw std::runtime_error("unexpected disk import");}
    std::string state_fingerprint() const override {return "cpu-api-fixture";}
    std::string describe() const override {return "CPU API fixture, no model or GPU";}
    const Tokenizer &tok;
    int32_t target,alternative;
    std::atomic<int> forwards{0};
    bool reject_drafts=false;
    int32_t verify_end=-1;
private:
    int64_t position=0,pending=-1;
    std::array<int64_t,4> snapshots;
};
}
int main(int argc,char **argv) {
 try {
    require(argc==2 || (argc==3 && std::string(argv[2])=="--serve"),"tokenizer path required");
    set_log_level(LogLevel::Error);
    Tokenizer tok(argv[1]); FakeBackend backend(tok);
    Engine::Options options; options.mtp_draft=3;options.think_nudge=false;
    Engine engine(backend,tok,options);
    ServerConfig cfg;cfg.host="127.0.0.1";cfg.port=0;cfg.model_id="qwen38-flash-next-125b-a6b";cfg.thinking_default=false;
    HttpServer server(engine,tok,cfg); int port=server.bind();if(argc==3){std::cout<<"PORT "<<port<<std::endl;server.serve();return 0;}std::thread worker([&]{server.serve();});
    struct Stop {HttpServer &server;std::thread &worker;~Stop(){server.stop();worker.join();}} stop{server,worker};
    httplib::Client client("127.0.0.1",port);client.set_connection_timeout(2);client.set_read_timeout(10);
    const auto post=[&](const char *path,const std::string &text,int status=200) {
        auto response=client.Post(path,text,"application/json");require((bool)response,"HTTP response missing");
        require(response->status==status,"HTTP status "+std::to_string(response->status)+": "+response->body);
        return response->body;
    };
    for(const std::string text:{"Привет, мир!","Ёж и е\xcc\x88ж","hello 😀","<|im_start|>",""}) {
        auto req=json::Value::object();req.set("content",json::Value::string(text));
        auto result=J::parse(post("/tokenize",req.dump()));auto expected=tok.encode(text);
        require(result.at("count").as_int()==(int64_t)expected.size(),"token count");
        const auto &ids=result.at("tokens").as_array();require(ids.size()==expected.size(),"token IDs size");
        for(size_t i=0;i<ids.size();++i)require(ids[i].as_int()==expected[i],"native tokenizer mismatch");
    }
    require(backend.forwards==0,"tokenize touched model backend");
    auto pieces=J::parse(post("/v1/tokenize",R"({"content":"Привет","with_pieces":true})"));
    require(pieces.at("tokens").as_array()[0].at("bytes").is_array(),"pieces bytes missing");
    auto plain=J::parse(post("/tokenize",R"({"content":"<|im_start|>","parse_special":false})"));
    require(plain.at("count").as_int()>1,"parse_special false");
    post("/tokenize",R"({"content":42})",400);post("/tokenize",R"({"content":"a","text":"b"})",400);
    post("/tokenize",R"({"content":"a","with_pieces":1})",400);post("/tokenize",R"({})",400);
    const double expected=-std::log1p(std::exp(-1.0));
    const auto check_item=[&](const J &item) {
        require(item.at("token_id").as_int()==backend.target,"wrong selected token metadata");
        require(std::fabs(item.at("logprob").as_double()-expected)<1e-9,"target normalization or MTP row mismatch");
        const auto &top=item.at("top_logprobs").as_array();require(top.size()==2,"top alternatives count");
        require(top[0].at("token_id").as_int()==backend.target,"wrong best alternative");
        require(top[1].at("token_id").as_int()==backend.alternative,"wrong second alternative");
        require(std::fabs(top[1].at("logprob").as_double()-(expected-1))<1e-9,"second logprob");
    };
    const std::string request=R"({"messages":[{"role":"user","content":"Ответь"}],"enable_thinking":false,"max_tokens":8,"temperature":0,"logprobs":true,"top_logprobs":2})";
    for(bool rejected:{false,true}) {
        backend.reject_drafts=rejected;
        auto result=J::parse(post("/v1/chat/completions",request));
        const auto &probs=result.at("choices").as_array()[0].at("logprobs").at("content").as_array();
        require(probs.size()==8,"lost/double-counted accepted or rejected MTP token");for(const auto &p:probs)check_item(p);
    }
    std::string streaming=request;streaming.pop_back();streaming+=",\"stream\":true}";
    std::string response=post("/v1/chat/completions",streaming);std::istringstream frames(response);std::string line;int count=0;
    while(std::getline(frames,line))if(line.starts_with("data: ")&&line!="data: [DONE]") {
        auto frame=J::parse(line.substr(6));const auto choices=frame.find("choices");
        if(choices&&!choices->as_array().empty())if(const auto lp=choices->as_array()[0].find("logprobs"))
            for(const auto &p:lp->at("content").as_array()){check_item(p);++count;}
    }
    require(count==8,"stream probabilities missing");
    auto raw=J::parse(post("/v1/completions",R"({"prompt":"a","max_tokens":8,"temperature":0,"logprobs":2})"));
    const auto &lp=raw.at("choices").as_array()[0].at("logprobs");require(lp.at("tokens").as_array().size()==8,"legacy length");
    for(const auto &p:lp.at("token_logprobs").as_array())require(std::fabs(p.as_double()-expected)<1e-9,"legacy normalization");
    auto ids=J::parse(post("/v1/completions",R"({"prompt":[65,66],"max_tokens":1,"temperature":0,"logprobs":0})"));
    require(ids.at("choices").as_array()[0].at("logprobs").at("token_logprobs").as_array().size()==1,"token-ID prompt/zero alternatives");
    auto named=J::parse(post("/v1/completions",R"({"prompt":"a","max_tokens":1,"temperature":0,"logprobs":2,"return_tokens_as_token_ids":true})"));
    require(named.at("choices").as_array()[0].at("logprobs").at("tokens").as_array()[0].as_string("token")=="token_id:"+std::to_string(backend.target),"named token reporting IDs");
    post("/v1/completions",R"({"prompt":[999999],"max_tokens":1})",400);
    post("/v1/completions",R"({"prompt":"a","logprobs":21,"max_tokens":1})",400);
    post("/v1/chat/completions",R"({"messages":[{"role":"user","content":"a"}],"logprobs":1,"max_tokens":1})",400);
    post("/v1/chat/completions",R"({"messages":[{"role":"user","content":"a"}],"top_logprobs":2,"max_tokens":1})",400);
    post("/v1/chat/completions",R"({"messages":[{"role":"user","content":"a"}],"logprobs":true,"max_tokens":4097})",400);
    auto thinking=J::parse(post("/v1/chat/completions",R"({"messages":[{"role":"user","content":"Думай"}],"enable_thinking":true,"max_thinking_tokens":8,"max_tokens":128,"temperature":0,"logprobs":true,"top_logprobs":2})"));
    auto thinking_lp=thinking.at("choices").as_array()[0].at("logprobs");
    require(thinking_lp.at("reasoning_content").as_array().size()==8,"reasoning probabilities or inserted budget text counted incorrectly");
    for(const auto &p:thinking_lp.at("reasoning_content").as_array())check_item(p);
    require(!thinking_lp.at("content").as_array().empty(),"answer probabilities lost after reasoning close");
    for(const auto &p:thinking_lp.at("content").as_array())check_item(p);
    const auto plain_answer=J::parse(post("/v1/chat/completions",R"({"messages":[{"role":"user","content":"Ответь"}],"enable_thinking":false,"max_tokens":8,"temperature":0})"));
    require(!plain_answer.at("choices").as_array()[0].find("logprobs"),"probabilities enabled by default");
    const std::string raw_stream=post("/v1/completions",R"({"prompt":"stream","max_tokens":8,"temperature":0,"logprobs":2,"stream":true})");
    std::istringstream raw_frames(raw_stream);count=0;
    while(std::getline(raw_frames,line))if(line.starts_with("data: ")&&line!="data: [DONE]") {
        auto frame=J::parse(line.substr(6));auto choices=frame.find("choices");
        if(choices&&!choices->as_array().empty())if(auto lp=choices->as_array()[0].find("logprobs");lp&&!lp->is_null())
            for(const auto &p:lp->at("token_logprobs").as_array()){require(std::fabs(p.as_double()-expected)<1e-9,"raw streamed logprob");++count;}
    }
    require(count==8,"raw streamed logprobs missing");
    // A stop as the very first raw token still has a probability distribution.
    const int32_t normal_target=backend.target;
    for(const char *name:{"<|endoftext|>","<|im_end|>"}) {
        backend.target=tok.id_of(name);
        auto end=J::parse(post("/v1/completions",std::string(R"({"prompt":"immediate terminal )")+name+R"(","max_tokens":1,"temperature":0,"logprobs":2})"));
        const auto choice=end.at("choices").as_array()[0];
        require(choice.at("text").as_string("text").empty(),"terminal token leaked into output text");
        const auto &terminal=choice.at("logprobs").at("token_ids").as_array();
        require(terminal.size()==1,"missing first-token terminal logprobs: "+std::string(name));
        require(terminal[0].as_int()==backend.target,"wrong terminal token ID");
        require(choice.at("finish_reason").as_string("finish")=="stop","terminal finish reason");
    }
    backend.target=normal_target;
    for(const char *name:{"<|endoftext|>","<|im_end|>","<tool_call>","</tool_call>","</think>"}) {
        backend.target=tok.id_of(name);
        auto result=J::parse(post("/v1/completions",std::string(R"({"prompt":"structural distribution )")+name+R"(","max_tokens":1,"temperature":0,"logprobs":2,"return_tokens_as_token_ids":true})"));
        auto metadata=result.at("choices").as_array()[0].at("logprobs");
        require(metadata.at("token_ids").as_array().size()==1,"raw structural token distribution missing");
        require(metadata.at("token_ids").as_array()[0].as_int()==backend.target,"raw structural token ID");
        require(std::fabs(metadata.at("token_logprobs").as_array()[0].as_double()-expected)<1e-9,"raw structural normalization");
    }
    backend.target=normal_target;
    for(const char *name:{"<|endoftext|>","<|im_end|>"}) {
        backend.verify_end=tok.id_of(name);backend.reject_drafts=false;
        auto result=J::parse(post("/v1/completions",std::string(R"({"prompt":"verify terminal )")+name+R"(","max_tokens":8,"temperature":0,"logprobs":2})"));
        auto metadata=result.at("choices").as_array()[0].at("logprobs");
        const auto values=metadata.at("token_ids").as_array();
        require(values.size()==2,"MTP rejection lost terminal probability");
        require(values[0].as_int()==normal_target && values[1].as_int()==backend.verify_end,"MTP terminal row identity");
        require(std::fabs(metadata.at("token_logprobs").as_array()[1].as_double()-expected)<1e-9,"MTP terminal target normalization");
        backend.verify_end=-1;
        backend.target=tok.id_of(name);
        auto frames=post("/v1/completions",std::string(R"({"prompt":"stream terminal )")+name+R"(","max_tokens":1,"temperature":0,"logprobs":2,"stream":true})");
        std::istringstream input(frames);int terminal_count=0;
        while(std::getline(input,line))if(line.starts_with("data: ")&&line!="data: [DONE]") {
            auto frame=J::parse(line.substr(6));auto choices=frame.find("choices");
            if(choices&&!choices->as_array().empty())if(auto lp=choices->as_array()[0].find("logprobs");lp&&!lp->is_null())
                for(const auto &id:lp->at("token_ids").as_array()){require(id.as_int()==backend.target,"stream terminal identity");++terminal_count;}
        }
        require(terminal_count==1,"stream lost/doubled terminal distribution");
        auto chat=J::parse(post("/v1/chat/completions",R"({"messages":[{"role":"user","content":"terminal chat"}],"enable_thinking":false,"max_tokens":1,"temperature":0,"logprobs":true,"top_logprobs":2})"));
        require(chat.at("choices").as_array()[0].at("logprobs").at("content").as_array().empty(),"raw terminal fix leaked structural token into chat metadata");
        backend.target=normal_target;
    }
    // A grammar mask must not renormalize the reported underlying target distribution.
    LogitRows rows=LogitRows::from_full(backend.row(),backend.logits_row());
    LogitMasks mask;mask.rows=1;mask.words=(rows.row+31)/32;mask.bits.assign(mask.words,0);mask.preserve_unmasked=true;
    mask.bits[backend.alternative/32]|=1u<<(backend.alternative%32);apply_masks(rows,mask);
    auto masked=token_logprob(rows,0,backend.alternative,2,tok);require(std::fabs(masked.logprob-(expected-1))<1e-9,"mask changed raw target distribution");
    J partial=chat_token_logprob(TokenLogprob{tok.byte_token(0xd0),-1,std::string(1,(char)0xd0),{}},tok);
    require(partial.at("bytes").as_array()[0].as_int()==0xd0,"split Cyrillic byte not preserved");
    require(complete_utf8_characters("\xd0")==0 && complete_utf8_characters("\xd0\xaf")==1,"split Unicode offsets");
    std::cout<<"PASS: "<<checks<<" CPU API/probability assertions, including MTP accept/reject and streaming\n";
 }catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
