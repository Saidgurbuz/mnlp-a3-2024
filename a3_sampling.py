import torch
from typing import Any, Dict
from a3_utils import *
import copy


class TopKSamplerForCausalLM(GeneratorForCausalLM):
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
    def sample(
        self,
        inputs: dict,
        top_k: int,
        temperature: float,
        max_new_tokens: int,
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-k sampling. 
        This means that we sample the next token from the top-k scoring tokens 
        by using their probability values.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - It only handles top_k => 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.
        
        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_k (int): the number of highest probability vocabulary tokens 
                         to keep for top-k filtering/sampling
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                  (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-k sampled sequence made of token ids of size (1,generated_seq_len)
                              This should include the starting pad token!
        """
        ########################################################################
        # NOTE: Don't change this part, it's to help you debug!
        constraint_return = self.input_constraints(inputs, max_new_tokens, top_k=top_k)
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
        temp_inputs = copy.deepcopy(inputs)
        generated_sequence = inputs['input_ids'].tolist()[0]
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(**temp_inputs)
                logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
            # Here we get the top k probabilities and indices
            next_token_probs, next_token_indices = torch.topk(next_token_probs, top_k)
            # Then we sample from the top k probabilities based on weighted probabilities
            next_token = next_token_indices[0][torch.multinomial(next_token_probs, 1)[0]]
            temp_inputs = self.prepare_next_inputs(temp_inputs, next_token)
            generated_sequence.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        return torch.LongTensor([generated_sequence]).to(self.model.device)


class TopPSamplerForCausalLM(GeneratorForCausalLM):
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
    def sample(
        self,
        inputs: dict,
        top_p: float,
        temperature: float,
        max_new_tokens: int
    ) -> torch.LongTensor:
        """Generates sequences of token ids with self.model 
        (which has a language modeling head) using top-p sampling. 
        This means that we sample the next token from the smallest set of most 
        probable tokens with probabilities that cumulatively add up to top_p *or higher*.
        If there are no tokens falling in the top_p cumulative probability mass 
        (e.g. because the top scoring tokens probability is larger than top_p) 
        then samples the top scoring token.

        - This function always does early stopping and does not handle the case 
            where we don't do early stopping, or stops at max_new_tokens.
        - It only handles inputs of batch size = 1.
        - The temperature variable modulates the distribution we sample 
            from, by scaling the logits before softmax.

        Args:
            inputs (dict): the tokenized input dictionary returned by the tokenizer
            top_p (float): the cumulative probability mass to select the smallest 
                           set of most probable tokens with probabilities that 
                           cumulatively add up to top_p or higher.
            temperature (float): the value used to modulate the next token probabilities, 
                                 scales logits before softmax
            max_new_tokens (int): the maximum numbers of new tokens to generate 
                                (i.e. not including the initial input tokens)

        Returns:
            torch.LongTensor: top-p sampled sequence made of token ids of size (1,generated_seq_len)
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
        # For hints, read the todo statement in GreedySearchDecoderForCausalLM.
        ########################################################################
        temp_inputs = copy.deepcopy(inputs)
        generated_sequence = inputs['input_ids'].tolist()[0]
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(**temp_inputs)
                logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = self._top_p_sampling(next_token_probs, top_p)
            temp_inputs = self.prepare_next_inputs(temp_inputs, next_token)
            generated_sequence.append(next_token.item())
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        return torch.LongTensor([generated_sequence]).to(self.model.device)
    
    def _top_p_sampling(self, next_token_probs, top_p):
        sorted_probs, sorted_indices = torch.sort(next_token_probs, descending=True)
        for i in range(1, len(sorted_probs[0])):
            if torch.sum(sorted_probs[0][:i]) > top_p:
                sorted_probs = sorted_probs[:,:i]
                break
        next_token_indices = torch.multinomial(sorted_probs, 1)[0]
        next_token = sorted_indices[0][next_token_indices.item()].unsqueeze(0)
        return next_token




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