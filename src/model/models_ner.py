from torchcrf import CRF
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import XLNetModel, XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetForTokenClassificationOutput
from transformers import AutoModel, AutoConfig, RobertaPreTrainedModel, RobertaModel, RobertaConfig
from transformers.modeling_outputs import TokenClassifierOutput

   
class CustomRobertaForTokenClassification(RobertaPreTrainedModel):
    def __init__(self, config, custom_layers):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.num_LSTM = custom_layers['LSTM'] if 'LSTM' in custom_layers.keys() else 0
        self.CRF = 'CRF' in custom_layers.keys() and custom_layers['CRF'] == True

        if self.num_LSTM:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size//2, num_layers = self.num_LSTM, bidirectional=True, dropout = 0.5)
        
        if self.CRF:
            self.crf = CRF(config.num_labels, batch_first=True)
            self.crf.reset_parameters()
        
        self.classifier = nn.Linear(self.config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
         
        if self.num_LSTM:
            # Forward propagate through LSTM
            sequence_output, _ = self.lstm(sequence_output)
        
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None

        if self.CRF:
              if labels is not None:
                  loss = -self.crf(emissions=logits, tags=labels, mask=attention_mask.byte(), reduction='token_mean')
                  logits = self.crf.decode(logits)
              else:
                  logits = self.crf.decode(logits)
              logits = torch.Tensor(logits)
        else:
              if labels is not None:
                  loss_fct = CrossEntropyLoss()
                  loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  
    
    
class CustomXLNetForTokenClassification(XLNetPreTrainedModel):
    def __init__(self, config, custom_layers):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = XLNetModel(config)
        
        self.num_LSTM = custom_layers['LSTM'] if 'LSTM' in custom_layers.keys() else 0
        self.CRF = 'CRF' in custom_layers.keys() and custom_layers['CRF'] == True

        if self.num_LSTM:
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size//2, num_layers = self.num_LSTM, bidirectional=True, dropout=0.5)
        
        if self.CRF:
            self.crf = CRF(config.num_labels, batch_first=True)
            self.crf.reset_parameters()
        
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_mems: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ) -> Union[Tuple, XLNetForTokenClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where *num_choices* is the size of the second dimension of the input tensors. (see *input_ids* above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_mems=use_mems,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
                
        if self.num_LSTM:
            # Forward propagate through LSTM
            sequence_output, _ = self.lstm(sequence_output) 

        logits = self.classifier(sequence_output)
        loss = None
        
        if self.CRF:
            if labels is not None:
                loss = -self.crf(emissions=logits, tags=labels, mask=attention_mask.byte(), reduction='token_mean')
                logits = self.crf.decode(logits)
            else:
                logits = self.crf.decode(logits)
            logits = torch.Tensor(logits)
        else:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XLNetForTokenClassificationOutput(
            loss=loss,
            logits=logits,
            mems=outputs.mems,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ) 


from typing import Union, List, Optional
import torch
import numpy as np
import copy
import re
from transformers.pipelines import AggregationStrategy, TokenClassificationPipeline

class SlidingWindowNERPipeline(TokenClassificationPipeline):
    """Modified version of TokenClassificationPipeline that uses a sliding
    window approach to fit long texts into the limited position embeddings of a
    transformer.
    """

    def __init__(self, aggregation_strategy, window_length: Optional[int] = None,
                 stride: Optional[int] = None, viterbi=False, *args, **kwargs):
        super(SlidingWindowNERPipeline, self).__init__(
            *args, **kwargs)
        self.viterbi = viterbi
        self.window_length = window_length or self.tokenizer.model_max_length
        if stride is None:
            self.stride = self.window_length // 2
        elif stride == 0:
            self.stride = self.window_length
        elif 0 < stride <= self.window_length:
            self.stride = stride
        else:
            raise ValueError("`stride` must be a positive integer no greater "
                             "than `window_length`")
        if aggregation_strategy == 'simple':
            self.aggregation_strategy = AggregationStrategy.SIMPLE
        elif aggregation_strategy == 'first':
            self.aggregation_strategy = AggregationStrategy.FIRST
        elif aggregation_strategy == 'average':
            self.aggregation_strategy = AggregationStrategy.AVERAGE
        elif aggregation_strategy == 'max':
            self.aggregation_strategy = AggregationStrategy.MAX
        else:
            self.aggregation_strategy = AggregationStrategy.NONE
        self.ignore_labels=["O"]

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy)
            with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `aggregation_strategy` is not :obj:`"none"`.
            - **index** (:obj:`int`, only present when ``aggregation_strategy="none"``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []
        num_labels = self.model.num_labels

        for i, sentence in enumerate(_inputs):

            # Manage correct placement of the tensors
            with self.device_placement():
                tokens = self.tokenizer(
                    sentence,
                    padding=True,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    return_special_tokens_mask=True,
                    add_special_tokens=True,
                    return_offsets_mapping=self.tokenizer.is_fast
                )
                if self.tokenizer.is_fast:
                    offset_mapping = \
                        tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = \
                    tokens.pop("special_tokens_mask").cpu().numpy()[0]

                if self.framework == "tf":
                    raise ValueError("SlidingWindowNERPipeline does not "
                                     "support TensorFlow models.")
                # Forward inference pass
                with torch.no_grad():
                    #tokens = self.ensure_tensor_on_device(**tokens)
                    tokens.to(self.device)
                    # Get logits (i.e. tag scores)
                    entities = np.zeros(tokens['input_ids'].shape[1:] +
                                        (num_labels,))
                    writes = np.zeros(entities.shape)
                    
                    for start in range(0, tokens['input_ids'].shape[1] - 1, self.stride):
                        end = start + self.window_length - 2

                        window_input_ids = torch.cat([
                            torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device),
                            tokens['input_ids'][:, start:end],
                            torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device)
                        ], dim=1)
                        window_logits = self.model(input_ids=window_input_ids)[0][0].cpu().numpy()
                        entities[start:end] += window_logits[1:-1]
                        writes[start:end] += 1
                        
                    entities = entities / writes

                    input_ids = tokens["input_ids"].cpu().numpy()[0]
                    if self.viterbi:
                        crf = CRF(self.model.num_labels, batch_first=True)
                        scores = crf.decode(torch.tensor(entities).float())
                    else:    
                        scores = np.exp(entities) / np.exp(entities).sum(
                            -1, keepdims=True)
                    pre_entities = self.gather_pre_entities(
                        sentence, input_ids, scores, offset_mapping,
                        special_tokens_mask, aggregation_strategy=self.aggregation_strategy)
                    grouped_entities = self.aggregate(
                        pre_entities, self.aggregation_strategy)
                    if self.aggregation_strategy != AggregationStrategy.NONE:
                    # Filter anything that is in self.ignore_labels
                        entities = [
                            entity
                            for entity in grouped_entities
                            if entity.get("entity", None) not in self.ignore_labels
                            and entity.get("entity_group", None) not in
                            self.ignore_labels
                        ]
                        answers.append(entities)
                    else:
                        answers.append(grouped_entities)

        if len(answers) == 1:
            return answers[0]
        return answers
      
class CrfSlidingWindowNERPipeline(SlidingWindowNERPipeline):
    """Modified version of SlidingWindowNERPipeline made for models that
    use a CRF.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy)
            with the following keys:

            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word (it is named `entity_group` when
              `aggregation_strategy` is not :obj:`"none"`.
            - **index** (:obj:`int`, only present when ``aggregation_strategy="none"``) -- The index of the
              corresponding token in the sentence.
            - **start** (:obj:`int`, `optional`) -- The index of the start of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
            - **end** (:obj:`int`, `optional`) -- The index of the end of the corresponding entity in the sentence.
              Only exists if the offsets are available within the tokenizer
        """

        _inputs, offset_mappings = self._args_parser(inputs, **kwargs)

        answers = []
        num_labels = self.model.num_labels

        for i, sentence in enumerate(_inputs):

            # Manage correct placement of the tensors
            with self.device_placement():
                tokens = self.tokenizer(
                    sentence,
                    #padding=True,
                    return_attention_mask=False,
                    return_tensors=self.framework,
                    return_special_tokens_mask=True,
                    add_special_tokens=False,
                    return_offsets_mapping=self.tokenizer.is_fast
                )
                if self.tokenizer.is_fast:
                    offset_mapping = \
                        tokens.pop("offset_mapping").cpu().numpy()[0]
                elif offset_mappings:
                    offset_mapping = offset_mappings[i]
                else:
                    offset_mapping = None

                special_tokens_mask = \
                    tokens.pop("special_tokens_mask").cpu().numpy()[0]

                if self.framework == "tf":
                    raise ValueError("SlidingWindowNERPipeline does not "
                                     "support TensorFlow models.")
                # Forward inference pass
                with torch.no_grad():
                    #tokens = self.ensure_tensor_on_device(**tokens)
                    tokens.to(self.device)
                    # Get logits (i.e. tag scores)
                    entities = np.zeros(tokens['input_ids'].shape[1:])
                    writes = np.zeros(entities.shape)
                    
                    
                    if tokens['input_ids'].shape[1] >= self.window_length - 2:
                        for start in range(
                                0, tokens['input_ids'].shape[1] - 1,
                                self.stride):
                            end = start + self.window_length - 2

                            window_input_ids = torch.cat([
                                torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device),
                                tokens['input_ids'][:, start:end],
                                torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device)
                            ], dim=1)
                            window_logits = self.model(
                                input_ids=window_input_ids)[0][0].cpu().numpy()

                            entities[start:end] = window_logits[1:-1]
                    else:
                        window_input_ids = torch.cat([
                                torch.tensor([[self.tokenizer.cls_token_id]]).to(self.device),
                                tokens['input_ids'][:,],
                                torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device)
                            ], dim=1)
                        window_logits = self.model(input_ids=window_input_ids)[0][0].cpu().numpy()
                        entities = window_logits[1:-1]
                    
                    input_ids = tokens["input_ids"].cpu().numpy()[0]
                    scores = entities
                    
                    
                    pre_entities = self.gather_pre_entities(
                        sentence, input_ids, scores, offset_mapping,
                        special_tokens_mask, aggregation_strategy=self.aggregation_strategy)
                    
                    
                    entities = []
                    for pre_entity in pre_entities:
                        entity_idx = pre_entity["scores"]
                        #score = pre_entity["scores"][entity_idx]
                        entity = {
                            "entity": self.model.config.id2label[entity_idx],
                            "score": 0,
                            "index": pre_entity["index"],
                            "word": pre_entity["word"],
                            "start": pre_entity["start"],
                            "end": pre_entity["end"],
                        }
                        entities.append(entity)
                    if self.aggregation_strategy != AggregationStrategy.NONE:
                        grouped_entities = self.group_entities(entities)
                    
                    if self.aggregation_strategy != AggregationStrategy.NONE:
                    # Filter anything that is in self.ignore_labels
                        entities = [
                            entity
                            for entity in grouped_entities
                            if entity.get("entity", None) not in self.ignore_labels
                            and entity.get("entity_group", None) not in
                            self.ignore_labels
                        ]
                        answers.append(entities)
                    else:
                        answers.append(entities)

        if len(answers) == 1:
            return answers[0]
        return answers
    
def escape(pattern):
    """Escape special characters in a string, except for single space."""
    special_chars_map = {i: '\\' + chr(i) for i in b'()[]{}?*+-|^$\\.&~#\t\n\r\v\f'}
    if isinstance(pattern, str):
        return pattern.translate(special_chars_map)
    else:
        pattern = str(pattern, 'latin1')
        return pattern.translate(special_chars_map).encode('latin1')
    

def remap_predictions(df, df_clean, predictions):
    """Method to remap the indexes of the predictions on cleaned DataFrame that correspond 
       to the original DataFrame."""
    
    predictions_cp = copy.copy(predictions)

    for row, row_clean, i in zip(df.iloc, df_clean.iloc, range(len(predictions_cp))):
        
        pred = predictions_cp[i]
        context = row['context']
        context_clean = row_clean['context']
        
        text = [row_clean['context'][p['start']:p['end']] for p in pred]
        
        start = []
        end = []
        off = 0
        
        s_temp = [p['start'] for p in pred]
        try:
            for t,tmp in zip(text,s_temp):
                
                while True: # avoid infinite loops
                    s1 = escape(t)
                    s2 = context[off:]
                    match = re.search(r'\s*'.join(s1.split()), s2)
                    s, e = match.start(), match.end()
                    
                    string_clean = context_clean[:tmp]
                    string = context[:s+off]
                    if re.sub('\s+', '', string) == re.sub('\s+', '', string_clean):
                       break
                    off += e
                
                start.append(s+off)
                end.append(e+off)
                off = end[-1]
                    
            for j in range(len(pred)):
                predictions_cp[i][j]['start'] = start[j]
                predictions_cp[i][j]['end'] = end[j]
        except Exception as e:
            print(e)
            print("Corrispondence not found")
    return predictions_cp


def remap_predictions_context(context, context_clean, pred):
    """Method to remap the indexes of the predictions on cleaned context that correspond 
        to the original context."""
    text = [context_clean[p['start']:p['end']] for p in pred]
    start,end = [], []
    off = 0
    
    s_temp = [p['start'] for p in pred]
    try:
        for t,tmp in zip(text,s_temp):
            while True:
                s1 = escape(t)
                s2 = context[off:]
                match = re.search(r'\s*'.join(s1.split()), s2)
                s, e = match.start(), match.end()
                
                string_clean = context_clean[:tmp]
                string = context[:s+off]
                if re.sub('\s+', '', string) == re.sub('\s+', '', string_clean): break
                off += e
            
            start.append(s+off)
            end.append(e+off)
            off = end[-1]
                
        for j in range(len(pred)):
            pred[j]['start'] = start[j]
            pred[j]['end'] = end[j]
            pred[j]['word'] = context[start[j]:end[j]]
    except Exception as e:
        print(e)
        print("Corrispondence not found")
    return pred




