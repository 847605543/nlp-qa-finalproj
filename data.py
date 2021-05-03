"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset, input_cuda


PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size, lower = True):
        self.words = self._initialize(samples, vocab_size, lower)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size, lower):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for (_, passage, question, _, _) in samples:
            for token in itertools.chain(passage, question):
                if lower:
                    vocab[token.lower()] += 1
                else:
                    for t in token:
                        vocab[t] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self, args, path, lower = True):
        self.args = args
        self.char_wise = False
        self.meta, self.elems = load_dataset(path)
        if args.bert:
            from transformers import AutoTokenizer
            # self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
            self.tokenizer = AutoTokenizer.from_pretrained('./bert/bert_tiny', model_max_length=512)
        else:
            self.tokenizer = None
        self.samples = self._create_samples(True)
        self.c_samples = self._create_samples(False)
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

    def _create_samples(self, lower = True):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        samples = []
        for elem in self.elems:
            # Unpack the context paragraph. Shorten to max sequence length.
            if self.args.bert:
                offsets_mapping = self.tokenizer(elem['context'], return_offsets_mapping=True, truncation=True).offset_mapping
            else:
                if lower:
                    passage = [
                    token.lower() for (token, offset) in elem['context_tokens']
                    ][:self.args.max_context_length]
                else:
                    passage = [
                        token for (token, offset) in elem['context_tokens']
                    ][:self.args.max_context_length]

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for qa in elem['qas']:
                qid = qa['qid']

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.ss
                if self.args.bert:
                    answer_start = answer_end = -1
                    char_start, char_end = qa['detected_answers'][0]['char_spans'][0]
                    # add 1 to offset since position is inclusive
                    char_end += 1
                    for token_idx, (token_start, token_end) in enumerate(offsets_mapping):
                        # print((token_start, token_end))
                        if answer_start >= 0 and answer_end >= 0:
                            break
                        if token_start == char_start:
                            answer_start = token_idx
                        if token_end == char_end:
                            answer_end = token_idx
                    samples.append(
                        (qid, elem['context'], qa['question'], answer_start, answer_end)
                    )
                else:
                    if lower:
                        question = [
                            token.lower() for (token, offset) in qa['question_tokens']
                        ][:self.args.max_question_length]
                    else:
                        question = [
                            token for (token, offset) in qa['question_tokens']
                        ][:self.args.max_question_length]
                    answers = qa['detected_answers']
                    
                    answer_start, answer_end = answers[0]['token_spans'][0]
                    samples.append(
                        (qid, passage, question, answer_start, answer_end)
                    )
                
        return samples

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        if self.args.char_cat:
            passages_c = []
            questions_c = []
        start_positions = []
        end_positions = []
        passs = []
        print('hello')
        self.max_word_len = 0
        self.max_question_word_len = 0
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, answer_start, answer_end = self.samples[idx]


            # Convert words to tensor.
            if self.args.bert:
                # leave tokenizer to batching
                passage_ids = passage
                question_ids = question
            else:
                passs.append(passage)
                passage_ids = torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(passage)
                )
                question_ids = torch.tensor(
                    self.tokenizer.convert_tokens_to_ids(question)
                )
            if self.args.char_cat:
                passage_c_ids = []
                for w in passage:
                    if len(w)>self.max_word_len:
                        self.max_word_len = min(args.max_word_len,len(w))
                    passage_c_ids.append(self.alphabet_tokenizer.convert_tokens_to_ids(w))
                question_c_ids = []
                for w in question:
                    if len(w) > self.max_word_len:
                        self.max_word_len = min(args.max_word_len,len(w))
                    question_c_ids.append(self.alphabet_tokenizer.convert_tokens_to_ids(w))
                questions_c.append(question_c_ids)
                passages_c.append(passage_c_ids)
            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)
        if self.args.char_cat:
            return zip(passages, questions, start_positions, end_positions,passages_c,questions_c)
        return zip(passages, questions, start_positions, end_positions)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            if self.args.char_cat:
                passages_c = []
                questions_c = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            if not self.args.bert:
                max_passage_length = 0
                max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])

                if self.args.char_cat:
                    passages_c.append(current_batch[ii][4])
                    questions_c.append(current_batch[ii][5])

                start_positions[ii] = current_batch[ii][2]
                end_positions[ii] = current_batch[ii][3]

                if not self.args.bert:
                    max_passage_length = max(
                        max_passage_length, len(current_batch[ii][0])
                    )
                    max_question_length = max(
                        max_question_length, len(current_batch[ii][1])
                    )
                if self.args.char_cat:
                    max_passage_word_length = 0
                    max_question_word_length = 0
                    #print(current_batch[ii][4])
                    for w in current_batch[ii][4]:
                        #print(self.alphabet_tokenizer.convert_ids_to_tokens(w),len(w))
                        #print(w)
                        max_passage_word_length = max(
                            max_passage_word_length, len(w)
                        )
                    for w in current_batch[ii][5]:
                        max_question_word_length = max(
                            max_question_word_length, len(w)
                        )


            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            if self.args.bert:
                padded_passages = self.tokenizer(passages, padding=True, truncation=True, return_tensors="pt")
                padded_questions = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
            else:
                padded_passages = torch.zeros(bsz, max_passage_length)
                padded_questions = torch.zeros(bsz, max_question_length)
                if self.args.char_cat:
                    padded_word_passages = torch.zeros((bsz, max_passage_length,self.max_word_len))
                    padded_word_questions = torch.zeros((bsz, max_question_length,self.max_word_len))
                # Pad passages and questions
                for iii, passage_question in enumerate(zip(passages, questions)):
                    passage, question = passage_question
                    padded_passages[iii][:len(passage)] = passage
                    padded_questions[iii][:len(question)] = question

                if self.args.char_cat:
                    for iv, passage_question in enumerate(zip(passages_c, questions_c)):
                        passage, question = passage_question
                        for v,p in enumerate(passage):
                            #print(len(p))
                            #print(self.alphabet_tokenizer.convert_ids_to_tokens(p))
                            #print(max_question_word_length)
                            #print(max_passage_word_length)
                            padded_word_passages[iv][v][:min(len(p),self.max_word_len)] = torch.Tensor(p[:min(len(p),self.max_word_len)])
                        for v,q in enumerate(question):
                            padded_word_questions[iv][v][:min(len(q),self.max_word_len)] = torch.Tensor(q[:min(len(q),self.max_word_len)])

            # Create an input dictionary
            if self.args.char_cat:
                batch_dict = {
                    'passages': input_cuda(self.args, padded_passages) if self.args.bert else cuda(self.args, padded_passages).long(),
                    'questions': input_cuda(self.args, padded_questions) if self.args.bert else cuda(self.args, padded_questions).long(),
                    'start_positions': cuda(self.args, start_positions).long(),
                    'end_positions': cuda(self.args, end_positions).long(),
                    'test': passages, # TODO: remove later
                    'passages_c': cuda(self.args, padded_word_passages).long(),
                    'questions_c': cuda(self.args, padded_word_questions).long()

            }
            else:
                batch_dict = {
                    'passages': input_cuda(self.args, padded_passages) if self.args.bert else cuda(self.args, padded_passages).long(),
                    'questions': input_cuda(self.args, padded_questions) if self.args.bert else cuda(self.args, padded_questions).long(),
                    'start_positions': cuda(self.args, start_positions).long(),
                    'end_positions': cuda(self.args, end_positions).long(),
                    'test': passages # TODO: remove later
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    # def register_tokenizer(self, tokenizer):
    #     """
    #     Stores `Tokenizer` object as an instance variable.

    #     Args:
    #         tokenizer: If `True`, shuffle examples. Default: `False`
    #     """
    #     self.tokenizer = tokenizer

    def register_tokenizer(self, tokenizer, alphabet_tokenizer=None):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer
        self.alphabet_tokenizer = alphabet_tokenizer
        self.char_wise = True

    def __len__(self):
        return len(self.samples)
