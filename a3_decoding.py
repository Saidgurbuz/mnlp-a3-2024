import torch
from typing import Any, Dict
from a3_utils import *
import copy
import torch.nn.functional as F

class GreedySearchDecoderForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using greedy decoding. 
        This means that we always pick the next token with the highest score/probability.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping (i.e. if the next token is an EOS 
            (end-of-sentence) token, you should stop decoding) or stops at 
            max_new_tokens.
        - It only handles inputs of batch size = 1.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: greedy decoded best sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens)
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################

        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # Hint (#1): There are 2 ways to pass the inputs to the model. Please open the
        # [GPT2LMHeadModel documentation](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel) 
        # and read the `Parameters` and `Returns` sections while looking at these hints.
        # Either of these approaches is fine since we don't expect the most efficient solution:
        #
        #   1. **If recomputing the past decoder processes at each decoding step:**
        #       Since the tokenizer's output dictionary keys matches these `Parameter` 
        #       i.e. arguments of the model you can directly do:
        #       ```python
        #       >> self.model(**inputs)
        #       ```
        #       Just be careful and think about how you modify the "input_ids" 
        #       and "attention_mask" keys across decoding steps. 
        
        #   2. **If using cached decoder hidden states at each decoding step:**
        #       To speed up the process (although *not required*) you can also get 
        #       the computed key/values hidden-states *so far* with `use_cache=True`
        #       where in the first step you may need to do:
        #       ```python
        #       >> self.model(**inputs, use_cache=True)
        #       ```
        #       This will return an extra dictionary entry called "past_key_values".
        #       In the next steps you would do, assuming your previous output 
        #       dict is called `outputs`:
        #       ```python
        #       >> self.model(**inputs, use_cache=True, past_key_values=outputs["past_key_values"])
        #       ```
        #       Again, be careful as to how you modify the "input_ids" and 
        #       "attention_mask" keys across decoding steps. In particular the 
        #       cached setting expects them to be different from the non-cached 
        #       setting. Read the `Parameters` and `Returns` sections of the 
        #       GPT2LMHeadModel carefully.
        #
        # Hint (#2): You can implement and use the `self.prepare_next_inputs` 
        #   function in `a3_utils.py` inherited by all decoding and sampling classes 
        #   (although you are not required to) to reduce repeated code and make it
        #   more readable. There isn't a unique solution for this so use it as you wish
        #   or create another function in this super class.
        ########################################################################
        temp_inputs = copy.deepcopy(inputs)
        generated_sequence = inputs['input_ids'].tolist()[0]
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(**temp_inputs)
                logits = outputs.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            temp_inputs = self.prepare_next_inputs(temp_inputs, next_token)
            generated_sequence.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        return torch.LongTensor([generated_sequence]).to(self.model.device)


class BeamSearchDecoderForCausalLM(GeneratorForCausalLM):
    ###########################################################################
    # NOTE: Caution - do not modify the args to the class + the args of 
    # the sample function.
    # 
    # However, feel free to add as many helper functions in this class as you want.
    ###########################################################################
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Inherits variables and helper functions from GeneratorForCausalLM.
        """
        super().__init__(model, tokenizer)
    
    @torch.no_grad()
    def search(
        self,
        inputs: dict,
        max_new_tokens: int,
        num_beams: int,
        num_return_sequences=1,
        length_penalty: float = 0.0
    ) -> dict: 
        """Generates sequences of token ids with self.model, 
        (which has a language modeling head) using beam search. This means that 
        given a probability distribution over the possible next tokens and 
        a beam width (here num_beams), needs to keep track of the most probable 
        num_beams candidates. (Hint: use log probabilities!)

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens. 
        - It only handles inputs of batch size = 1.
        - It only handles beam size > 1.
        - It includes a length_penalty variable that controls the score assigned 
            to a long generation. This is implemented by exponiating the amount 
            of newly generated tokens to this value. Then, divide the score which 
            can be calculated as the sum of the log probabilities so far.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)
            num_beams (int): number of beams for beam search
            num_return_sequences (int, optional):
                the amount of best sequences to return. Cannot be more than beam size.
                Defaults to 1.
            length_penalty (float, optional): 
                exponential penalty to the length that is used with beam-based generation. 
                It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. 
                Defaults to 0.0.

        Returns:
            dict: dictionary with two key values:
                    - "sequences": torch.LongTensor depicting the best generated sequences (token ID tensor) 
                        * shape (num_return_sequences, maximum_generated_sequence_length)
                        * ordered from best scoring sequence to worst
                        * if a sequence has reached end of the sentence, 
                          you can fill the rest of the tensor row with the pad token ID
                    - "sequences_scores": length penalized log probability score list, ordered by best score to worst
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(
            inputs, 
            max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        if constraint_return is None:
            return None
        else:
            max_new_tokens = constraint_return
            
        self.model.eval()
        ########################################################################
        
        ########################################################################
        # TODO: Implement me! Read the docstring above carefully for expected features.
        #
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################
        input_ids = inputs['input_ids']
        original_input_length = input_ids.size(1)
        input_ids = input_ids.expand(num_beams, -1).to(self.model.device)

        all_candidates = []
        worst_score = 0.0

        beam_scores = torch.zeros((num_beams)).to(input_ids.device)
        # to break the ties in the first iteration
        beam_scores[1:] = -1e7

        for _ in range(max_new_tokens):
            all_beam_scores, all_beam_next_tokens = self._get_next_beams(input_ids, num_beams, beam_scores)
            current_seqs = []
            for rank, (beam_score, whole_token_id) in enumerate(zip(all_beam_scores[0], all_beam_next_tokens[0])):
                
                # since I flatten all the beams, I first get actual beam index
                beam_idx = whole_token_id // self.model.config.vocab_size
                token_id = whole_token_id % self.model.config.vocab_size
                if token_id.item() == self.eos_token_id:
                    # print('EOS token reached')
                    is_not_better_beam = rank >= num_beams
                    if is_not_better_beam:
                        continue
                    actual_score = beam_score.item() / ((input_ids[beam_idx].size(0) - original_input_length) ** length_penalty)
                    if len(all_candidates) < num_beams or actual_score > worst_score:
                        new_beam = {
                            'input_ids': input_ids[beam_idx].clone(),
                            'score': actual_score
                        }
                        all_candidates.append(new_beam)

                        if(len(all_candidates) > num_beams):
                            sorted_candidate_scores = sorted((x['score'], idx) for idx, x in enumerate(all_candidates))
                            all_candidates.pop(sorted_candidate_scores[0][1])
                            worst_score = sorted_candidate_scores[1][0]
                        else:
                            worst_score = actual_score if actual_score < worst_score else worst_score
                else:
                    current_seqs.append({ 'beam_idx': beam_idx, 'token_id': token_id, 'beam_score': beam_score })      

                # once we have all the beams filled, we will break the loop
                if len(current_seqs) == num_beams:
                    break
            
            current_score = all_beam_scores[0].max().item() / ((input_ids.size(1) + 1 - original_input_length) ** length_penalty)

            # check if we have enough candidates to select from and we are sure that the remaining beams will not be better
            if len(all_candidates) >= num_beams and current_score < worst_score:
                break

            beam_scores = beam_scores.new_tensor([x['beam_score'] for x in current_seqs])
            beam_tokens = input_ids.new_tensor([x['token_id'] for x in current_seqs])
            beam_indices = input_ids.new_tensor([x['beam_idx'] for x in current_seqs])
            input_ids = input_ids[beam_indices]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            # I am checking if all the beams have reached the end of the sentence (early stopping)
            is_all_finished = True
            for token in all_beam_next_tokens[0]:
                if token != self.eos_token_id:
                    is_all_finished = False
                    break
            if is_all_finished:
                break

        # I added remaining beams to the candidates if beams are not filled
        if len(all_candidates) < num_beams:
            for beam_idx in range(num_beams):
                final_score = beam_scores[beam_idx].item()
                final_tokens = input_ids[beam_idx]
                
                actual_final_score = final_score / ((final_tokens.size(0) - original_input_length) ** length_penalty)
                if len(all_candidates) < num_beams or actual_final_score > worst_score:
                    new_beam = {
                        'input_ids': final_tokens.clone(),
                        'score': actual_final_score
                    }
                    all_candidates.append(new_beam)

                    if(len(all_candidates) > num_beams):
                        sorted_candidate_scores = sorted((x['score'], idx) for idx, x in enumerate(all_candidates))
                        all_candidates.pop(sorted_candidate_scores[0][1])
                        worst_score = sorted_candidate_scores[1][0]
                    else:
                        worst_score = actual_final_score if actual_final_score < worst_score else worst_score

        sequences = [beam['input_ids'] for beam in all_candidates[:num_return_sequences]]
        padded_sequences = self._pad_sequences(sequences)

        return {
            'sequences': torch.LongTensor(padded_sequences).squeeze(1).to(input_ids.device),
            'sequences_scores': [beam['score'] for beam in all_candidates[:num_return_sequences]]
        }

    def _get_next_beams(self, input_ids, num_beams=1, beam_scores=None):
        attention_mask = torch.ones(input_ids.size()).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        next_token_logits = logits[:, -1, :]
        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
        next_token_log_probs = next_token_log_probs + beam_scores[:, None]
        next_token_log_probs = next_token_log_probs.view(1, -1)

        # the reason that I used 2 * num_beams is that we want to have enough candidates to select from
        # if some of them are not valid due to reaching the end of the sentence (early stopping), worst case scenario is that
        # we will have num_beams end_of_sentence candidates, so we will have num_beams valid candidates to select from
        top_k_log_probs, top_k_tokens = torch.topk(next_token_log_probs, k=2*num_beams, dim=-1)
        return top_k_log_probs, top_k_tokens

    def _pad_sequences(self, sequences):
        max_len = max([len(seq) for seq in sequences])
        padded_sequences = [list(seq) + [self.pad_token_id] * (max_len - len(seq)) for seq in sequences]
        return padded_sequences

def main():
    ############################################################################
    # NOTE: You can use this space for testing but you are not required to do so!
    ############################################################################
    seed = 421
    torch.manual_seed(seed)
    torch.set_printoptions(precision=16)
    model_name = "vicgalle/gpt2-alpaca-gpt4"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


if __name__ == '__main__':
    main()