import tensorflow as tf
import numpy as np
from seq2seq_pgn_tf2.batcher import output_to_words
from tqdm import tqdm
import math


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        # list of all the tokens from time 0 to the current time step t
        self.tokens = tokens
        # list of the log probabilities of the tokens of the tokens
        self.log_probs = log_probs
        # decoder state after the last token decoding
        self.state = state
        # attention dists of all the tokens
        self.attn_dists = attn_dists
        # generation probability of all the tokens
        self.p_gens = p_gens
        self.coverage = coverage

        # self.abstract = ""
        # self.text = ""
        # self.real_abstract = ""

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """Method to extend the current hypothesis by adding the next decoded token and all
        the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          state=state,  # we update the state
                          attn_dists=self.attn_dists + [attn_dist],
                          # we  add the attention dist of the decoded token
                          p_gens=self.p_gens + [p_gen],  # we add the p_gen
                          coverage=coverage)

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
                       batch_oov_len, enc_pad_mask, use_coverage, prev_coverage, past_input, repetition_penalty=1.2):
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
        outputs = model(enc_outputs,  # shape=(3, 115, 256)
                        dec_state,  # shape=(3, 256)
                        enc_inp,  # shape=(3, 115)
                        enc_extended_inp,  # shape=(3, 115)
                        dec_input,  # shape=(3, 1)
                        batch_oov_len,  # shape=()
                        enc_pad_mask,  # shape=(3, 115)
                        use_coverage,
                        prev_coverage)  # shape=(3, 115, 1)
        final_dists = outputs["logits"]  # 预测 (dec_len,batch,vocab_size)
        dec_hidden = outputs["dec_hidden"]
        attentions = outputs["attentions"]
        coverages = outputs["coverages"]
        p_gens = outputs["p_gens"]
        # print("final_dists=",final_dists)
        next_token_logits = tf.squeeze(final_dists)
        past_tokens = tf.convert_to_tensor(past_input)
        # print("next_token_logits=",next_token_logits.shape)
        # print("past_tokens=",past_tokens)
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:  # 生成出现的词就惩罚,有问题，生成内容会改变
            next_token_logits_penalties = _create_next_token_logits_penalties(
                past_tokens, next_token_logits, repetition_penalty
            )
            # print(next_token_logits_penalties)
            next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)
        # final_dists shape=(3, 1, 30000)
        # top_k_probs shape=(3, 6)
        # top_k_ids shape=(3, 6)
        # print(next_token_logits)
        no_repeat_ngram_size=10
        cur_len=dec_input.shape[1]
        next_token_logits = tf.nn.log_softmax(next_token_logits, axis=-1)
        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            num_batch_hypotheses = params["beam_size"]
            banned_tokens = calc_banned_ngram_tokens(
                past_tokens, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            # create banned_tokens boolean mask
            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append(
                    [True if token in banned_tokens_slice else False for token in range(next_token_logits.shape[-1])]
                )
        bad_words_ids=[[vocab.word_to_id('[UNK]')]]
        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(past_input, bad_words_ids)

            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append(
                    [True if token in banned_tokens_slice else False for token in range(next_token_logits.shape[-1])]
                )

            next_token_logits = set_tensor_by_indices_to_value(
                next_token_logits, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float("inf")
            )

        oov_probs = tf.convert_to_tensor([[1]]* params["beam_size"],dtype=tf.float32) \
            if next_token_logits.shape[1] == params["vocab_size"] else next_token_logits[:,params["vocab_size"]:]
        if next_token_logits.shape[1] == params["vocab_size"]:
            _, top_1_oov_id = tf.nn.top_k(oov_probs, k=1)
        else:
            top_1_oov_id=tf.convert_to_tensor([[1-params["vocab_size"]]]* params["beam_size"],dtype=tf.int32)


        top_k_probs, top_k_ids = tf.nn.top_k(next_token_logits, k=params["beam_size"] * 2)
        top_k_log_probs = tf.math.log(top_k_probs)
        # dec_hidden shape = (3, 256)
        # attentions, shape = (3, 115)
        # p_gens shape = (3, 1)
        # coverages,shape = (3, 115, 1)
        results = {"dec_state": dec_hidden,
                   "attention_vec": attentions,  # [batch_sz, max_len_x, 1]
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs,
                   "p_gen": p_gens,
                   "coverages": coverages,
                   "oov_id": top_1_oov_id
                   }
        return results

    # end of the nested class

    # We run the encoder once and then we use the results to decode each time step token
    # state shape=(3, 256), enc_outputs shape=(3, 115, 256)
    enc_input = batch[0]["enc_input"]
    # print(enc_input.shape)
    enc_outputs, state = model.call_encoder(enc_input)
    # Initial Hypothesises (beam_size many list)
    hyps = [Hypothesis(tokens=[vocab.word_to_id('[START]')],
                       log_probs=[0.0],
                       state=state[0],
                       p_gens=[],
                       attn_dists=[],
                       coverage=np.zeros([enc_input.shape[1], 1], dtype=np.float32)) for _ in
            range(params['batch_size'])]
    results = []  # list to hold the top beam_size hypothesises
    steps = 0  # initial step
    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
        # we replace all the oov is by the unknown token
        latest_tokens = [t if t in range(params['vocab_size']) else vocab.word_to_id('[UNK]') for t in latest_tokens]
        # we collect the last states for each hypothesis
        states = [h.state for h in hyps]
        # prev_coverage = [h.coverage for h in hyps]  # list of coverage vectors (or None)
        # prev_coverage = tf.convert_to_tensor(prev_coverage)

        # we decode the top likely 2 x beam_size tokens tokens at time step t for each hypothesis
        # model, batch, vocab, params
        dec_input = tf.expand_dims(latest_tokens, axis=1)  # shape=(3, 1)   #decode不能用最后一个
        # 自定义
        dec_tokens = [h.tokens for h in hyps]

        # 有问题
        # print(dec_tokens)
        def handle(line):
            res = [dd if dd in range(params['vocab_size']) else vocab.word_to_id('[UNK]') for dd in line]
            return res

        dec_tokens = [handle(d) for d in dec_tokens]
        # print(dec_tokens)
        dec_inputs = tf.convert_to_tensor(dec_tokens, dtype=tf.int32)
        dec_inputs = tf.reshape(dec_inputs, shape=(params["beam_size"], -1))
        # 自定义
        dec_states = tf.stack(states, axis=0)
        returns = decode_onestep(batch[0]['enc_input'],  # shape=(3, 115)
                                 enc_outputs,  # shape=(3, 115, 256)
                                 dec_input,  # shape=(3, 1)
                                 dec_states,  # shape=(3, 256)
                                 batch[0]['extended_enc_input'],  # shape=(3, 115)
                                 batch[0]['max_oov_len'],  # shape=()
                                 batch[0]['sample_encoder_pad_mask'],  # shape=(3, 115)
                                 params['is_coverage'],  # true
                                 past_input=dec_inputs,
                                 prev_coverage=None)  # shape=(3, 115, 1)
        topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, oov_ids = returns['top_k_ids'], \
                                                                                          returns['top_k_log_probs'], \
                                                                                          returns['dec_state'], \
                                                                                          returns['attention_vec'], \
                                                                                          returns["p_gen"], \
                                                                                          returns['coverages'],\
                                                                                          returns["oov_id"]

        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        num = 1
        for i in range(num_orig_hyps):
            h = hyps[i]
            new_state = new_states[i]
            attn_dist = attn_dists[i]
            p_gen = p_gens[i]
            new_coverage_i = new_coverage[i]
            num += 1
            for j in range(params['beam_size'] * 2):
                # we extend each hypothesis with each of the top k tokens
                # (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                new_hyp = h.extend(token=topk_ids[i, j].numpy(),
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i
                                   )
                all_hyps.append(new_hyp)
        # in the following lines, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
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

    # At the end of the loop we return the most likely hypothesis, which holds the most likely ouput sequence,
    # given the input fed to the model
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)
    best_hyp = hyps_sorted[0]
    best_hyp = result_index2text(best_hyp, vocab, batch)
    # best_hyp.abstract = " ".join(output_to_words(best_hyp.tokens, vocab, batch[0]["article_oovs"][0])[1:-1])
    # best_hyp.text = batch[0]["article"].numpy()[0].decode()
    return best_hyp


def result_index2text(hyp, vocab, batch):
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    hyp.article = batch[0]["article"].numpy()[0].decode()

    words = []
    for index in hyp.tokens:
        if index != 2 and index != 3:
            if index < (len(article_oovs) + vocab.size()):
                if index < vocab.size():
                    words.append(vocab.id_to_word(index))
                else:
                    words.append(article_oovs[index - vocab.size()].decode())
            else:
                print('error values id :{}'.format(index))
    hyp.abstract = " ".join(words)
    return hyp


# 惩罚率
def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty):
    # create logit penalties for already seen input_ids
    token_penalties = np.ones(shape_list(logits))
    prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]  # 之前出现的token_ids
    for i, prev_input_id in enumerate(prev_input_ids):
        logit_penalized = logits[i].numpy()[prev_input_id]  # 被惩罚
        logit_penalties = np.zeros(logit_penalized.shape)
        # if previous logit score is < 0 then multiply repetition penalty else divide
        logit_penalties[logit_penalized < 0] = repetition_penalty  # x，让它变小
        logit_penalties[logit_penalized > 0] = 1 / repetition_penalty  # 除，让它变小
        # print("0 -- ",logit_penalties)
        np.put(token_penalties[i], prev_input_id, logit_penalties)
        # print("1 -- ",logit_penalties)
    return tf.convert_to_tensor(token_penalties, dtype=tf.float32)


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    # Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].numpy().tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].numpy().tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens

def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.numpy().tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def set_tensor_by_indices_to_value(tensor, indices, value):
    # create value_tensor since tensor value assignment is not possible in TF
    value_tensor = tf.zeros_like(tensor) + value
    return tf.where(indices, value_tensor, tensor)