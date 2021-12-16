"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class BartForMultiConditionalGeneration(BartForConditionalGeneration):

    def multi_encode(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict=None
    ):
        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D)
        # (B, N, L) -> (B*N, L) -> (B, N*L)
        B = input_ids.size(0)  # batch-size
        N = input_ids.size(1)  # num-docs
        L = input_ids.size(2)  # max_len
        if input_ids.size() != attention_mask.size():
            raise ValueError(
                f"Input ids different shape ({input_ids.size()}) than attention mask ({attention_mask.size()})"
            )
        input_ids = input_ids.contiguous().view(B * N, L)
        attention_mask = attention_mask.contiguous().view(B * N, L)
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        if return_dict:
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = encoder_outputs[0]
        # hidden_states: (B * N, L, D)
        D = hidden_states.size(2)
        stacked_source_reps = hidden_states.contiguous().view(B, N * L, D)
        if return_dict:
            encoder_outputs = BaseModelOutput(last_hidden_state=stacked_source_reps)
        else:
            encoder_outputs = (stacked_source_reps,)
        stacked_source_mask = attention_mask.contiguous().view(B, N * L)
        return encoder_outputs, stacked_source_mask

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        encoder_outputs, attention_mask = self.multi_encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return super().generate(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            **kwargs
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):

        if input_ids is None:
            if encoder_outputs is None:
                raise ValueError("Encoder outputs is required when no input ids passed")
        else:
            encoder_outputs, attention_mask = self.multi_encode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict = return_dict
                # encoder_outputs=encoder_outputs
            )

        output = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return output
