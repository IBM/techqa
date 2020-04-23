import json
import logging
from copy import deepcopy
from os import path
from random import random
import re
from datetime import datetime
from tqdm import tqdm
import torch
from typing import Optional, Iterable, List, Tuple, Dict

from transformers import PreTrainedTokenizer, RobertaTokenizer
from torch.utils.data import TensorDataset

TEXT_ENCODING = 'utf-8'

class TechQaInputFeature(object):
    """A single input vector for the model."""

    __slots__ = ['qid', 'doc_id', 'doc_span_index', 'tokens', 'token_to_original_start_offset',
                 'token_to_original_end_offset',
                 'input_ids', 'input_mask', 'segment_ids', 'start_position', 'end_position']

    def __init__(self, qid: str, doc_id: str, doc_span_index: int, tokens: List[str],
                 token_to_original_start_offset: Dict[int, int], input_mask: List[int],
                 token_to_original_end_offset: Dict[int, int], input_ids: List[int],
                 segment_ids: List[int], start_position: Optional[int] = None,
                 end_position: Optional[int] = None):
        self.qid = qid
        self.doc_id = doc_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_original_start_offset = token_to_original_start_offset
        self.token_to_original_end_offset = token_to_original_end_offset
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return "%s(%s)" % (
            self.__class__.__name__,
            ",".join('%s=%s' %
                     (attribute, getattr(self, attribute)) for attribute in self.__slots__))


class Span(object):
    __slots__ = ['start', 'end']

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

        if self.start != -1 and self.start > self.end:
            raise ValueError('Inconsistent span start (%d) and end (%d)' % (start, end))

    def is_null_span(self):
        return self.start == -1 or self.end == -1

    def __contains__(self, other):
        if self.is_null_span():
            return False
        elif isinstance(other, Span):
            if other.start >= self.start and other.end < self.end:
                return True
            else:
                return False
        else:
            return self.end > other >= self.start

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return "Span[boundaries=[%s,%s)]" % (self.start, self.end)

    def __hash__(self):
        return hash((self.start, self.end))


_NULL_SPAN = Span(start=-1, end=-1)


class WordTokenizerWithCharOffsetTracking(object):
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions

    @classmethod
    def tokenize(cls, text: str) -> List[Tuple[str, Span]]:
        def inner(text):
            for match in re.finditer(r'\S+', text):
                span = Span(start=match.start(), end=match.end())
                tok = match.group(0)
                yield (tok, span)

        return list(inner(text))


def _get_tokenized_text_and_mapping_to_original_offsets(
        text: str, tokenizer: PreTrainedTokenizer,
        max_number_of_tokens: Optional[int] = None) -> Tuple[List[str], List[Span]]:
    tokens = list()
    token_idx_to_char_boundaries = []

    # We do a first pass white space / punctuation tokenization using the non-destructive
    # spacy tokenizer so that we can maintain a map back to the original character offsets
    # and then do a second pass with the tokenizer of choice to convert into model
    # specific word pieces
    for word, char_offset_boundaries_for_this_word in WordTokenizerWithCharOffsetTracking.tokenize(
            text):
        word_pieces = tokenizer.tokenize(word)

        if max_number_of_tokens is not None and len(tokens) + len(word_pieces) > \
                max_number_of_tokens:
            logging.warning('Truncating text after %d word piece tokens' % len(tokens))
            break

        for word_piece in tokenizer.tokenize(word):
            tokens.append(word_piece)
            token_idx_to_char_boundaries.append(char_offset_boundaries_for_this_word)

    return tokens, token_idx_to_char_boundaries


def _map_answer_boundaries_to_doc_span_vector_offsets(
        original_answer_span: Span, size_of_question_segment: int, doc_span: Span) -> Span:
    # TODO: Think harder about the situation where the correct answer boundaries
    #  partially overlap with the document span, currently we still mark those spans as not
    #  having an answers...but perhaps that's sending bad signals to the training algorithm??
    if original_answer_span in doc_span:
        answer_span_within_answer_segment = \
            Span(start=original_answer_span.start - doc_span.start + size_of_question_segment,
                 end=original_answer_span.end - doc_span.start + size_of_question_segment)
    else:
        # Point to the [CLS] token; Re-visit if we need to support xlnet which doesn't keep the
        # [CLS] token at position 0
        answer_span_within_answer_segment = Span(start=0, end=0)

    return answer_span_within_answer_segment


def _split_into_spans(total_length: int, window_length: int, stride_length: int) -> List[Span]:
    spans = list()
    if total_length <= window_length:
        logging.debug('Window size (%d) for splitting is larger than the total document size: %d' %
                      (window_length, total_length))
        spans.append(Span(start=0, end=total_length))
    else:
        start_offset = 0
        while start_offset < total_length:
            spans.append(Span(start_offset,
                              end=min(start_offset + window_length, total_length)))
            start_offset += stride_length
    return spans


def _map_answer_char_offsets_to_token_boundaries(
        answer_span_in_char_offsets: Span, token_idx_to_char_offset_boundaries: List):
    # NOTE: the character offsets may not align exactly with our token boundaries so we find the
    # nearest aligned token boundaries in order to specify the span

    start_token_idx = None
    end_token_idx = None
    for token_idx, token_char_boundaries in enumerate(token_idx_to_char_offset_boundaries):
        if start_token_idx is None:
            if answer_span_in_char_offsets.start in token_char_boundaries:
                # We found that there is a token that contains (inclusive) the specified start
                # char offset
                start_token_idx = token_idx
            elif answer_span_in_char_offsets.start < token_char_boundaries.start:
                # We use the first token that starts after the specified start char offset
                start_token_idx = token_idx

        if end_token_idx is None:
            # NOTE: annotations use exclusive end offset, but we want inclusive token end offset
            if token_char_boundaries.start >= answer_span_in_char_offsets.end:
                # We found the first token that starts _after_ the specified end char offset, so
                # use the previous token
                end_token_idx = max(0, token_idx - 1)
            elif (answer_span_in_char_offsets.end - 1) in token_char_boundaries:
                # We found that there is a token that contains (exclusive) the specified end
                # char offset
                end_token_idx = token_idx

        if start_token_idx is not None and end_token_idx is not None:
            break

    if start_token_idx is None:
        raise RuntimeError('Unable to map the answer character offsets %s to an appropriate'
                           ' token offset boundary using %s' % (
                               answer_span_in_char_offsets, token_idx_to_char_offset_boundaries))
    elif end_token_idx is None:
        # assume that the end character offset takes us beyond the last tokenized word (e.g.
        # if the document has trailing white space
        end_token_idx = len(token_idx_to_char_offset_boundaries) - 1

    return Span(start=start_token_idx, end=end_token_idx)
      
def _prepare_query_segment(query: Dict[str, str], max_query_length: int, sep_tokens: List[str],
                           tokenizer: PreTrainedTokenizer) -> List[str]:
    try:
        query_title, _ = _get_tokenized_text_and_mapping_to_original_offsets(
            text=query['QUESTION_TITLE'],
            tokenizer=tokenizer)

        query_body, _ = _get_tokenized_text_and_mapping_to_original_offsets(
            text=query['QUESTION_TEXT'],
            tokenizer=tokenizer)

        if len(query_title) + len(query_body) + len(sep_tokens) > max_query_length:
            query_title = query_title[: max_query_length // 2]
        query_tokens = query_title + sep_tokens
        query_tokens += query_body[:max_query_length - len(query_tokens)]
    except Exception as ex:
        raise ValueError('Unable to parse query tokens from query `%s`: %s' % (query, ex)) from ex

    return query_tokens

def generate_features_for_example(
        qid: str, query: dict, doc: dict, doc_id: str, answer_span: Span,
        tokenizer: PreTrainedTokenizer, max_seq_length: int, doc_stride: int,
        max_query_length: int, negative_span_subsampling_probability: float,
        add_doc_title_to_passage: bool = False) -> \
        List[TechQaInputFeature]:
    features = list()

    between_text_segment_separator = [tokenizer.sep_token]
    if isinstance(tokenizer, RobertaTokenizer):
        # Roberta uses 2 sep tokens to separate segments
        between_text_segment_separator.append(tokenizer.sep_token)

    query_segment = _prepare_query_segment(query=query, tokenizer=tokenizer,
                                           sep_tokens=between_text_segment_separator,
                                           max_query_length=max_query_length)
    if len(query_segment) < 1:
        logging.warning('No query tokens left after tokenization for qid %s' % qid)
        return features

    query_segment = [tokenizer.cls_token] + query_segment + between_text_segment_separator
    context_size = max_seq_length - len(query_segment) - 1
    if add_doc_title_to_passage:
        document_title_tokens, _ = _get_tokenized_text_and_mapping_to_original_offsets(
            text=doc['title'],
            tokenizer=tokenizer)[: context_size // 2]

        # better to add doc title to query segment rather than doc (i.e. segment id == [0])
        query_segment += document_title_tokens + between_text_segment_separator
        context_size -= len(document_title_tokens) + len(between_text_segment_separator)
    
    document_tokens, token_idx_to_char_boundaries = \
        _get_tokenized_text_and_mapping_to_original_offsets(
            text=doc['text'],
            tokenizer=tokenizer)

    if len(document_tokens) < 1:
        logging.warning('No document tokens left after tokenization for doc id %s' % doc['_id'])
        return features
    sliding_window_spans = _split_into_spans(total_length=len(document_tokens),
                                             window_length=context_size,
                                             stride_length=doc_stride)

    if not answer_span.is_null_span():
        answer_span = _map_answer_char_offsets_to_token_boundaries(answer_span,
                                                                   token_idx_to_char_boundaries)

    features = []

    # For each span, create a new feature vector
    for doc_span_index, doc_span in enumerate(sliding_window_spans):
        # Create a copy of the question segment for each doc span
        tokens = deepcopy(query_segment)
        segment_ids = [0] * len(tokens)

        size_of_question_segment = len(segment_ids)

        # Map the original token index offsets to the feature vector containing the current doc
        # span, the query, and the special tokens
        answer_span_within_context = _map_answer_boundaries_to_doc_span_vector_offsets(
            original_answer_span=answer_span,
            size_of_question_segment=size_of_question_segment, doc_span=doc_span)
        span_token_to_original_start_offset = {
            i - doc_span.start + size_of_question_segment: token_idx_to_char_boundaries[i].start for i
            in range(doc_span.start, doc_span.end)}
        span_token_to_original_end_offset = {
            i - doc_span.start + size_of_question_segment: token_idx_to_char_boundaries[i].end for i
            in range(doc_span.start, doc_span.end)}

        span_has_answer = answer_span_within_context.start >= size_of_question_segment

        if not span_has_answer and random() > negative_span_subsampling_probability:
            logging.debug('Dropping span since it\'s a negative instance')
        else:
            # Add document tokens
            for i in range(doc_span.start, doc_span.end):
                tokens.append(document_tokens[i])
                segment_ids.append(1)

            # Add the final separator token to indicate end of passage
            tokens.append(tokenizer.sep_token)
            segment_ids.append(1)

            # Initialize the input ids & masks (for padding)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                # Mask is 0 for padding tokens
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            features.append(TechQaInputFeature(
                qid=qid, doc_id=doc_id, doc_span_index=doc_span_index, tokens=tokens,
                token_to_original_start_offset=span_token_to_original_start_offset,
                token_to_original_end_offset=span_token_to_original_end_offset,
                input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                start_position=answer_span_within_context.start,
                end_position=answer_span_within_context.end))
    return features

def create_features(input_query_file: str, input_corpus_file: str, max_seq_length: int,
                    max_query_length: int, doc_stride: int, tokenizer: PreTrainedTokenizer,
                    negative_subsampling_probability_when_has_answer: Optional[float] = None,
                    negative_subsampling_probability_when_no_answer: Optional[float] = None,
                    add_doc_title_to_passage: bool = False):
    with open(input_corpus_file, encoding=TEXT_ENCODING) as infile:
        logging.info('Loading corpus text from %s' % infile)
        document_by_id = json.load(infile)
        logging.info('Loaded %d documents from corpus' % len(document_by_id))

    with open(input_query_file, encoding=TEXT_ENCODING) as infile:
        logging.info('Loading queries and annotations from %s' % infile)
        queries = json.load(infile)
        logging.info('Loaded %d queries from %s' % (len(queries), infile))

    num_generated_features = 0
    if negative_subsampling_probability_when_has_answer is None:
        negative_subsampling_probability_when_has_answer = 1.0
    if negative_subsampling_probability_when_no_answer is None:
        negative_subsampling_probability_when_no_answer = 1.0

    features = []

    for i, query in enumerate(tqdm(queries, desc='Featurizing queries')):
        qid = "The %d th query" % i
        try:
            qid = query['QUESTION_ID']
            correct_doc_id = None
            if 'ANSWERABLE' in query and query['ANSWERABLE'] == 'Y':
                # Add positive examples and keep track of the doc id so that we don't featurize it again from the answer
                # pool
                correct_doc_id = query['DOCUMENT']
                for input_feature in generate_features_for_example(
                        qid=qid, doc_id=query['DOCUMENT'],
                        query=query, doc=document_by_id[query['DOCUMENT']],
                        answer_span=Span(start=int(query['START_OFFSET']),
                                         end=int(query['END_OFFSET'])),
                        tokenizer=tokenizer, max_seq_length=max_seq_length,
                        doc_stride=doc_stride, max_query_length=max_query_length,
                        negative_span_subsampling_probability=
                        negative_subsampling_probability_when_has_answer,
                        add_doc_title_to_passage=add_doc_title_to_passage):
                    num_generated_features += 1
                    features.append(input_feature)

            # Add negative examples from the answer pool
            for doc_id in query['DOC_IDS']:
                doc_id = doc_id.strip()
                if doc_id != correct_doc_id:
                    if doc_id not in document_by_id:
                        logging.warning('Document %s is specified in the DOC_IDS for question %s'
                                        ' but does not exist in the corpus!!' % (qid, doc_id))
                    else:
                        for input_feature in generate_features_for_example(
                                qid=qid, doc_id=doc_id, query=query, doc=document_by_id[doc_id],
                                answer_span=_NULL_SPAN,
                                tokenizer=tokenizer, max_seq_length=max_seq_length,
                                doc_stride=doc_stride, max_query_length=max_query_length,
                                negative_span_subsampling_probability=
                                negative_subsampling_probability_when_no_answer,
                                add_doc_title_to_passage=add_doc_title_to_passage):
                            num_generated_features += 1
                            features.append(input_feature)

        except Exception as ex:
            logging.warning('Error featurizing query %s. We will skip this query: %s' % (qid, ex))

    logging.info('Generated %d features from %d queries' %
                 (num_generated_features, len(queries)))
    return features


def _derive_feature_cache_name(
        input_query_file: str, input_corpus_file: str, max_seq_len: int,
        max_query_length: int, doc_stride: int, tokenizer: PreTrainedTokenizer,
        negative_subsampling_probability_when_has_answer: Optional[float] = None,
        negative_subsampling_probability_when_no_answer: Optional[float] = None,
        add_doc_title_to_passage: bool = False) -> str:
    cache_filename = '{0}_{1}_features_with_{2}_max_seq_len_{3}_stride_{4}_max_q_len_{5}_' \
                     'has_answer_ssrate_{6}_no_answer_ssrate_{7}_doc_title_{8}'.format(
        path.splitext(input_query_file)[0],
        path.splitext(path.basename(input_corpus_file))[0],
        tokenizer.__class__.__name__,
        max_seq_len,
        doc_stride,
        max_query_length,
        negative_subsampling_probability_when_has_answer,
        negative_subsampling_probability_when_no_answer,
        'Y' if add_doc_title_to_passage else 'N')

    if hasattr(tokenizer, 'basic_tokenizer') and hasattr(tokenizer.basic_tokenizer,
                                                         'do_lower_case'):
        cache_filename += '_lower_case_{0}'.format(tokenizer.basic_tokenizer.do_lower_case)

    return cache_filename

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if evaluate:
        input_query_file = args.predict_file
        negative_subsampling_probability_when_has_answer = None
        negative_subsampling_probability_when_no_answer = None
    else:
        input_query_file = args.train_file
        negative_subsampling_probability_when_has_answer = args.negative_sampling_prob_when_has_answer
        negative_subsampling_probability_when_no_answer = args.negative_sampling_prob_when_no_answer

    return  get_tech_qa_features(
                            input_query_file=input_query_file,
                            input_corpus_file=args.input_corpus_file,
                            max_seq_len=args.max_seq_length,
                            max_query_length=args.max_query_length,
                            doc_stride=args.doc_stride,
                            tokenizer=tokenizer,
                            negative_subsampling_probability_when_no_answer=negative_subsampling_probability_when_no_answer,
                            negative_subsampling_probability_when_has_answer=negative_subsampling_probability_when_has_answer,
                            add_doc_title_to_passage=args.add_doc_title_to_passage,
                            output_examples=output_examples
                        )

def get_dataset_from_features(features):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    all_feature_indices = torch.arange(all_input_ids.size(0), dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_feature_indices)
    
    return dataset

def get_tech_qa_features(
        input_query_file: str, input_corpus_file: str, max_seq_len: int,
        max_query_length: int, doc_stride: int, tokenizer: PreTrainedTokenizer,
        negative_subsampling_probability_when_has_answer: Optional[float] = None,
        negative_subsampling_probability_when_no_answer: Optional[float] = None,
        add_doc_title_to_passage: bool = False,
        output_examples: bool = False
        ) -> \
        Iterable[TechQaInputFeature]:
    logging.info('**** Generating feature caches for examples from %s ****\n' %
                 input_query_file)
    logging.info('Using corpus: %s' % input_corpus_file)
    logging.info('Using tokenizer: %s' % tokenizer.__class__.__name__)
    logging.info('Using max sequence length: %s' % max_seq_len)
    logging.info('Using document stride: %s' % doc_stride)
    logging.info('Using maximum query length: %s' % max_query_length)
    logging.info(
        'When example has answer, using negative instance subsampling probability: %s' %
        negative_subsampling_probability_when_has_answer)
    logging.info(
        'When example has NO answer, using negative instance subsampling probability: %s' %
        negative_subsampling_probability_when_no_answer)

    logging.info("add doc title to passage: %d" % (add_doc_title_to_passage))

    cache_file = _derive_feature_cache_name(
        max_seq_len=max_seq_len, max_query_length=max_query_length,
        doc_stride=doc_stride, tokenizer=tokenizer, input_corpus_file=input_corpus_file,
        input_query_file=input_query_file,
        negative_subsampling_probability_when_no_answer=
        negative_subsampling_probability_when_no_answer,
        negative_subsampling_probability_when_has_answer=
        negative_subsampling_probability_when_has_answer,
        add_doc_title_to_passage=add_doc_title_to_passage)

    if not path.isfile(cache_file):
        logging.info('Did not find previously cached features (%s) for %s, generating them now' %
                     (cache_file, input_query_file))
        features = create_features(
                input_query_file=input_query_file,
                input_corpus_file=input_corpus_file,
                tokenizer=tokenizer, max_seq_length=max_seq_len,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                negative_subsampling_probability_when_has_answer=
                negative_subsampling_probability_when_has_answer,
                negative_subsampling_probability_when_no_answer=
                negative_subsampling_probability_when_no_answer,
                add_doc_title_to_passage=add_doc_title_to_passage
                )

        dataset = get_dataset_from_features(features)
        
        with open(input_query_file, encoding=TEXT_ENCODING) as infile:
            gold_dict = {q['QUESTION_ID']: q for q in json.load(infile)}
        
        logging.info("Saving features into cached file %s", cache_file)
        torch.save({"features": features, "dataset": dataset, "examples": gold_dict}, cache_file)         
    else:
        logging.info('Skipping featurization and loading features from cache: %s' % cache_file)
        features_and_dataset = torch.load(cache_file)
        dataset = features_and_dataset["dataset"]
        if output_examples:
            gold_dict = features_and_dataset["examples"]
            features = features_and_dataset["features"]

    if output_examples:
        return dataset, features, gold_dict
    return dataset