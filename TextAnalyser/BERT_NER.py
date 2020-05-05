# Tutorial: https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
# Dataset: https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
# https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py

import pandas as pd
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns

print(torch.__version__)
print(transformers.__version__)

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_data(training_data):
    """
    Prepare the data.
    """

    data = pd.read_csv(training_data, encoding="latin1").fillna(method="ffill")
    print(data.tail(10))

    getter = SentenceGetter(data)

    # Get sentences
    sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
    print(sentences[0])

    # Get labels
    labels = [[s[2] for s in sentence] for sentence in getter.sentences]
    print(labels[0])

    # Add PAD to tag values and enumerate
    tag_values = list(set(data["Tag"].values))
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    print(tag2idx)

    return sentences, labels, tag_values, tag2idx


def load_and_prepare_data(tokenizer, MAX_LEN, BATCH_SIZE, sentences, labels):
    """
    Limit our sequence length to 75 tokens and
    use a batch size of 32 as suggested by the BERT paper.
    Note, that Bert supports sequences of up to 512 tokens.
    """

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(tokenizer, sent, labs)
        for sent, labs in zip(sentences, labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    # Convert tokens to indixes ["This", "is", "it"] -> [1188, 1110, 1122] and add padding
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Convert tags to indixes
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")

    attention_masks = [[float(i != tag2idx["PAD"]) for i in ii] for ii in input_ids]

    # Split the dataset to use 10% to validate the model
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)

    # Convert the dataset to torch tensors
    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    # The last step is to define the dataloaders.
    # We shuffle the data at training time with the RandomSampler and at test time we just pass them sequentially with the SequentialSampler.
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

    return train_dataloader, valid_dataloader


def get_hyperparameters(model, FULL_FINETUNING):
    # Before the fine-tuning process, setup the optimizer and add the parameters it should update.
    # Choose AdamW optimizer.
    # Add some weight_decay as regularization to the main weight matrices.

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters


def train_validate(model, optimizer, scheduler, device, EPOCHS, MAX_GRAD_NORM, train_dataloader, valid_dataloader):

    # Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []

    # 1. Training
    # Perform one full pass over the training set.
    for _ in trange(EPOCHS, desc="Epoch"):

        # Put the model into training mode.
        model.train()

        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0:
                print(step)

            # Add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()

            # Forward pass to calculate the loss (outputs a prediction, given the input)
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels
                            )

            # Get the loss
            loss = outputs[0]

            # Backward pass to calculate the gradients
            loss.backward()

            # Track train loss
            total_loss += loss.item()

            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)

            # Update parameters
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # 2. Validation
        # Put the model into evaluation mode
        model.eval()

        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)

            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            eval_accuracy += flat_accuracy(logits, label_ids)
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        # Print validation loss and accuracy
        eval_loss = eval_loss / nb_eval_steps
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        # Print validation F1-score
        pred_tags = [tag_values[p_i] for p in predictions for p_i in p]
        valid_tags = [tag_values[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

    return loss_values, validation_loss_values


def visualize(loss_values, validation_loss_values):
    """
    Visualise training loss.
    """

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o', label="training loss")
    plt.plot(validation_loss_values, 'r-o', label="validation loss")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def predict(test_sentence, tokenizer, model):
    """
    Extract entities from a given sentence.
    """

    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).cuda()

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

    # Join BPE split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    # Print out the results
    for token, label in zip(new_tokens, new_labels):
        print("{}\t{}".format(label, token))


if __name__ == "__main__":

    sentences, labels, tag_values, tag2idx = load_data(training_data="./data/NER/ner_dataset.csv")

    # Set up constants
    MAX_LEN = 75
    BATCH_SIZE = 32
    EPOCHS = 3
    MAX_GRAD_NORM = 1.0
    NUM_LABELS = len(tag_values)
    FULL_FINETUNING = True

    # Specify device data for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer
    print("Loading the tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    # Load and prepare data
    print("Loading training and validation data into DataLoaders...")
    train_dataloader, valid_dataloader = load_and_prepare_data(tokenizer, MAX_LEN,
                                                                BATCH_SIZE, sentences, labels)

    # Load the pre-trained bert-base-cased model and provide the number of possible labels.
    print("Loading the pre-trained bert-base-cased model..")
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=NUM_LABELS,
        output_attentions = False,
        output_hidden_states = False
    )
    # Pass the model parameters to the GPU
    model.to(device) # OR model.cuda()

    # Set hyperparameters (optimizer, weight decay, learning rate)
    print("Initializing optimizer and setting hyperparameters...")
    optimizer_grouped_parameters = get_hyperparameters(model, FULL_FINETUNING)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )

    # Total number of training steps is number of batches * number of epochs.
    print("Setting up scheduler...")
    total_steps = len(train_dataloader) * EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,

        num_training_steps=total_steps
    )

    # Training the model
    print("Training the model...")
    loss_values, validation_loss_values = train_validate(model, optimizer, scheduler,
                                                        device, EPOCHS, MAX_GRAD_NORM,
                                                        train_dataloader, valid_dataloader)

    # 5. Visualise training losses
    visualize(loss_values, validation_loss_values)

    # 6. Apply model to test sentence
    test_sentence = """
    Mr. Trump’s tweets began just moments after a Fox News report by Mike Tobin, a
    reporter for the network, about protests in Minnesota and elsewhere.
    """
    predict(test_sentence, tokenizer, model)

    # Save the model and tokenizer
    model.save_pretrained('./models/NER/')
    tokenizer.save_pretrained('./models/NER/')
