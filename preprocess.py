def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples, tokenizer, max_len=1024):
    """
    After running tokenizer, word ids can get misaligned
    need to re-align BIO labels, i.e. make sure split-up words
    get tagged as I-, [CLS] and [SEP] etc. are given sentinel values
    """
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        max_length=max_len
    )
    
    old_labels = examples['labels_cat']   # categorical labels should be here already
    new_labels = []
    for i, labels in enumerate(old_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels
    return tokenized_inputs
