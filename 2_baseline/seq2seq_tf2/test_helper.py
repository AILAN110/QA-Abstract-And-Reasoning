import tensorflow as tf
import numpy as np
from seq2seq_tf2.batcher import output_to_words
from tqdm import tqdm
import math


def greedy_decode(model, dataset, vocab, params):
    # 存储结果
    batch_size = params["batch_size"]
    results = []

    sample_size = 20000
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = sample_size // batch_size + 1
    datasets = iter(dataset)
    for i in tqdm(range(steps_epoch)):
        # enc_data,_=next(iter(dataset))
        try:
            enc_data, _ = next(datasets)
        except Exception as e:
            print("StopIter!")
            break

        # results += batch_greedy_decode(model, enc_data, vocab, params)
        results += beam_decode(model, enc_data, vocab, params)
    return results


def batch_greedy_decode(model, enc_data, vocab, params):
    # 判断输入长度
    batch_data = enc_data["enc_input"]
    batch_size = enc_data["enc_input"].shape[0]
    # 开辟结果存储list
    predicts = [''] * batch_size
    # inputs = batch_data # shape=(3, 115)
    inputs = tf.convert_to_tensor(batch_data)
    # hidden = [tf.zeros((batch_size, params['enc_units']))]
    # enc_output, enc_hidden = model.encoder(inputs, hidden)
    enc_output, enc_hidden = model.call_encoder(inputs)  # 编码

    dec_hidden = enc_hidden
    # dec_input = tf.expand_dims([vocab.word_to_id(vocab.START_DECODING)] * batch_size, 1)
    dec_input = tf.constant([2] * batch_size)  # <start>
    dec_input = tf.expand_dims(dec_input, axis=1)  # [batch_sz,1]

    context_vector, _ = model.attention(dec_hidden, enc_output)
    for t in range(params['max_dec_len']):
        # 单步预测
        _, pred, dec_hidden = model.decoder(dec_input,
                                            dec_hidden,
                                            enc_output,
                                            context_vector)
        context_vector, _ = model.attention(dec_hidden, enc_output)
        predicted_ids = tf.argmax(pred, axis=-1).numpy().tolist()
        # print(predicted_ids.numpy().tolist())
        for index, predicted_id in enumerate(predicted_ids):
            predicts[index] += vocab.id_to_word(predicted_id) + ' '

        # using teacher forcing
        dec_input = tf.expand_dims(predicted_ids, 1)

    results = []
    for predict in predicts:
        # 去掉句子前后空格
        predict = predict.strip()
        # 句子小于max len就结束了 截断vocab.word_to_id('[STOP]')
        if '[STOP]' in predict:
            # 截断stop
            predict = predict[:predict.index('[STOP]')]
        # 保存结果
        results.append(predict)
    return results


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens):
        # list of all the tokens from time 0 to the current time step t
        self.tokens = tokens  # 所有tokens
        # list of the log probabilities of the tokens of the tokens
        self.log_probs = log_probs  # 当前概率
        # decoder state after the last token decoding
        self.state = state  # 当前状态
        # attention dists of all the tokens
        self.attn_dists = attn_dists  # 所有attn
        # generation probability of all the tokens
        self.p_gens = p_gens  # 没用到
        # self.coverage = coverage

        # self.abstract = ""
        # self.text = ""
        # self.real_abstract = ""

    def extend(self, token, log_prob, state, attn_dist, p_gen):
        """Method to extend the current hypothesis by adding the next decoded token and all
        the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          state=state,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist],
                          # we  add the attention dist of the decoded token
                          p_gens=self.p_gens + [p_gen],  # we add the p_gen
                          )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def beam_decode(model, batch, vocab, params):
    def decode_onestep(enc_inp, enc_outputs, dec_input, dec_state, enc_extended_inp,
                       batch_oov_len, enc_pad_mask, use_coverage, prev_coverage):
        """
            Method to decode the output step by step (used for beamSearch decoding)
            Args:
                sess : tf.Session object
                batch : current batch, shape = [beam_size, 1, vocab_size( + max_oov_len if pointer_gen)]
                (for the beam search decoding, batch_size = beam_size)
                enc_outputs : hiddens outputs computed by the encoder LSTM
                dec_state : beam_size-many list of decoder previous state, LSTMStateTuple objects,
                shape = [beam_size, 2, hidden_size]
                dec_input : decoder_input, the previous decoded batch_size-many words, shape = [beam_size, embed_size]
                cov_vec : beam_size-many list of previous coverage vector
            Returns: A dictionary of the results of all the ops computations (see below for more details)
        """
        # print("enc_outputs:",enc_outputs.shape)
        # print("dec_state:",dec_state.shape)
        # print("enc_inp:",enc_inp.shape)
        # print("enc_extended_inp:",enc_extended_inp.shape)
        # print("dec_input:",dec_input.shape)
        dec_tar = tf.ones(shape=(params["beam_size"], 1))
        final_dists, dec_hidden = model(enc_outputs,  # shape=(32, 200, 128)
                                        dec_input,  # shape=(3, 1)
                                        dec_state,  # shape=(3, 128)
                                        dec_tar)  # shape=(200, )
        # enc_extended_inp)  # shape=(200, )
        # batch_oov_len,  # shape=()
        # enc_pad_mask,  # shape=(3, 115)
        # use_coverage,
        # prev_coverage)  # shape=(3, 115, 1)

        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dists), k=params["beam_size"] * 2)
        top_k_log_probs = tf.math.log(top_k_probs)

        results = {"dec_state": dec_hidden,
                   # "attention_vec": attentions,  # [batch_sz, max_len_x, 1]
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   # "p_gen": p_gens,
                   }
        return results

    # print(batch)
    params["batch_size"]=params["beam_size"]
    dataset = batch
    res = []
    for k in range(params["batch_size"]):
        # enc_input = tf.expand_dims(dataset["enc_input"][k], axis=1)
        enc_input = dataset["enc_input"][k]
        enc_ = tf.squeeze(tf.stack([[enc_input] * params["beam_size"]], axis=0))
        print(enc_)
        enc_outputs, state = model.call_encoder(enc_)  # 全部编码
        hyps = [Hypothesis(tokens=[vocab.word_to_id('[START]')],
                           log_probs=[0.0],
                           state=state[0],
                           p_gens=[],
                           attn_dists=[]) for _ in range(params['beam_size'])]
        # print('hyps', hyps)
        results = []  # list to hold the top beam_size hypothesises
        steps = 0  # initial step

        while steps < params['max_dec_steps'] and len(results) < params['beam_size']:  # 一次beam_search
            # print('step is ', steps)
            latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
            latest_tokens = [t if t in range(params['vocab_size']) else vocab.word_to_id('[UNK]') for t in
                             latest_tokens]  # [batch]
            states = [h.state for h in hyps]  # [batch]
            dec_input = tf.expand_dims(latest_tokens, axis=1)  # shape=(beam, 1)
            dec_states = tf.stack(states, axis=0)  # shape=[beam,128]
            returns = decode_onestep(dataset['enc_input'][k],  # shape=(3, 115)
                                     enc_outputs,  # shape=(3, 115, 256)
                                     dec_input,  # shape=(3, 1)
                                     dec_states,  # shape=(3, 256)
                                     dataset['extended_enc_input'][k],  # shape=(3, 115)
                                     dataset['max_oov_len'],  # shape=()
                                     dataset['sample_encoder_pad_mask'][k],  # shape=(3, 115)
                                     True,  # true
                                     prev_coverage=None)  # shape=(3, 115, 1)
            topk_ids, topk_log_probs, new_states = returns['top_k_ids'], \
                                                   returns['top_k_log_probs'], \
                                                   returns['dec_state']
            # returns['attention_vec'], \
            # returns["p_gen"], \
            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(hyps)
            num = 1
            # 获取3x3x2种可能性
            for i in range(num_orig_hyps):
                h, new_state = hyps[i], new_states[i]
                num += 1
                for j in range(params['beam_size'] * 2):
                    new_hyp = h.extend(token=topk_ids[i, j].numpy(),
                                       log_prob=topk_log_probs[i, j],
                                       state=new_state,
                                       attn_dist=None,
                                       p_gen=[],
                                       )
                    all_hyps.append(new_hyp)
            hyps = []
            # 取前3种
            sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
            for h in sorted_hyps:
                if h.latest_token == vocab.word_to_id('[STOP]'):
                    if steps >= params['min_dec_steps']:
                        results.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                    break
            steps += 1

        if len(results) == 0:
            results = hyps

        hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
        best_hyp = hyps_sorted[0]  # 取最优
        best_hyp.abstract = " ".join(output_to_words(best_hyp.tokens, vocab, dataset["article_oovs"][0])[1:-1])
        best_hyp.text = dataset["article"].numpy()[0].decode()
        print('best_hyp is ', best_hyp.abstract)
        res.append(best_hyp.abstract)
    return res
