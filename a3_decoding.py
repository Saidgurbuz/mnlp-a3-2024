import torch
from typing import Any, Dict
from a3_utils import *


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

        pass


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
        pass


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