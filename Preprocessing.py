from typing import Dict
import pandas as pd
import json
import os
import sentencepiece as spm
import logging


# get configuration for the preprocessing of the corpus and embeddings
CORPUS_CONFIG_PATH = 'config/corpus_config.json'
BPE_MODEL_CONFIG_PATH = 'config/bpe_model_config.json'

logging.basicConfig(
    level=logging.DEBUG,                   # Set the minimum log level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger('TransformerLogger')


# this function is used to replace newline symbols and
def normalize_text(sequence: str):
    # Replace newline characters with space
    text = sequence.replace('\n', ' ')
    return text


# this method normalizes each sequence in the src and target language of the corpus
def normalize_corpus(df_corpus: pd.DataFrame):
    df_normalized = df_corpus.copy(deep=True)
    df_normalized['en'] = df_normalized['en'].apply(normalize_text)
    df_normalized['de'] = df_normalized['de'].apply(normalize_text)
    return df_normalized


def prepare_corpus(corpus_config: Dict = None):
    logger.info('Preparing the corpus from language files.')
    # check if proprietary config was passed
    if corpus_config is None:
        # Load the config file
        with open(CORPUS_CONFIG_PATH, "r") as f:
            corpus_config = json.load(f)

    corpus_path = corpus_config.get('corpuse_path', 'corpus')
    corpus_data = corpus_config.get('datasets', [])

    # check if corpus exists already
    if os.path.exists(os.path.join(corpus_path,'corpus.pickle')) and os.path.exists(os.path.join(corpus_path,'corpus_normalized.txt')):
        normalized_path = os.path.join(corpus_path,'corpus_normalized.txt')

        logger.info(f'Normalized corpus exists already at {normalized_path}. Existing file is used instead.')

        return pd.read_csv(normalized_path, names=['en', 'de']), normalized_path

    # prepare lists to hold all sentences in the files
    english_sentences = []
    german_sentences = []

    # prepare name for the stored corpus
    corpus_name = 'corpus'

    # prepare paths to files
    for single_c in corpus_data:

        src_file_path = single_c['src']
        tgt_file_path = single_c['tgt']

        #corpus_name += '_' + src_file_path.split(".")[0]

        src_file = os.path.join(corpus_path, src_file_path)
        tgt_file = os.path.join(corpus_path, tgt_file_path)

        # Loading the two files into arrays split by new lines
        with open(src_file, 'r', encoding='utf-8') as f:
            english_sentences.extend(f.read().splitlines())

        with open(tgt_file, 'r', encoding='utf-8') as f:
            german_sentences.extend(f.read().splitlines())
        print(len(german_sentences))

        # Checking if the german and english files have the same number of lines
        if len(english_sentences) != len(german_sentences):
            raise ValueError(
                f"Mismatch in number of sentences between English and German for files {src_file} and {tgt_file}.")


    data = {'en': english_sentences, 'de': german_sentences}
    df_corpus = pd.DataFrame(data)
    df_corpus_normalized = normalize_corpus(df_corpus)

    # store corpus in directory using the constructed name
    df_corpus.to_csv(os.path.join(corpus_path, f"{corpus_name}.txt"), index=False, header=False)
    df_corpus_normalized.to_csv(os.path.join(corpus_path, f"{corpus_name}_normalized.txt"), index=False, header=False)

    return df_corpus_normalized, os.path.join(corpus_path, f"{corpus_name}_normalized.txt")


def prepare_bpe_model(input_path: str = 'corpus/corpus.txt', bpe_model_config: Dict = None):
    logger.info('Preparing the bpe model from the corpus file.')

    if input_path is None:
        raise ValueError(
            f"Cannot create a BPE encoding without a source corpus.")

    # check if proprietary config was passed
    if bpe_model_config is None:
        # Load the config file
        with open(BPE_MODEL_CONFIG_PATH, "r") as f:
            bpe_model_config = json.load(f)

    # define the location to store the bpe model using the input file
    # Extract the filename from the path
    filename = os.path.basename(input_path)
    # Remove the specific endings
    if filename.endswith("_normalized.txt"):
        filename = filename.replace("_normalized.txt", "")
    elif filename.endswith(".txt"):
        filename = filename.replace(".txt", "")

    # check if bpe model exists already
    if os.path.exists(os.path.join(bpe_model_config['model_prefix'], f"{filename}_bpe_model.model")):
        bpe_model_path = os.path.join(bpe_model_config['model_prefix'], f"{filename}_bpe_model.model")

        logger.info(f'BPE model exists already at {bpe_model_path}. Existing model is used instead.')

        return bpe_model_path

    # train model for BPE encoding
    spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=os.path.join(bpe_model_config['model_prefix'], f"{filename}_bpe_model"),
        vocab_size=bpe_model_config['vocab_size'],
        character_coverage=bpe_model_config['character_coverage'],
        model_type=bpe_model_config['model_type'],

        pad_id=bpe_model_config['pad_id'],
        unk_id=bpe_model_config['unk_id'],
        bos_id=bpe_model_config['bos_id'],
        eos_id=bpe_model_config['eos_id'],
        user_defined_symbols=','.join(bpe_model_config['user_defined_symbols']),
        normalization_rule_name=bpe_model_config['normalization_rule_name']
    )

    return os.path.join(bpe_model_config['model_prefix'], f"{filename}_bpe_model.model")

# Creating the encoded dataframe
def encode_dataframe(sp, df_corpus: pd.DataFrame):
    # use lambda function to reduce number of parameters
    encode_with_sp = lambda text: sp.encode_as_pieces(text) if isinstance(text, str) else ""

    df_corpus = df_corpus.map(encode_with_sp).copy(deep=True)

    #filter the empty lines
    return df_corpus[df_corpus != ""]

def run_pre_processing(corpus_config: Dict = None, bpe_model_config: Dict = None):
    logger.info('Process the text corpus.')

    # create the corpus
    df_normalized_corpus, corpus_path = prepare_corpus(corpus_config)

    logger.info('Train the BPE model.')
    # prepare the bpe model
    sp_path = prepare_bpe_model(corpus_path, bpe_model_config)

    logger.info('Load the BPE model.')
    # get the bpe model for encoding
    sp = spm.SentencePieceProcessor()
    sp.load(sp_path)

    logger.info('Encode the corpus with our BPE model.')
    # encode the dataframe
    df_encoded = encode_dataframe(sp,df_normalized_corpus)

    # store the encoded dataframe
    # Extract the filename from the path
    encoded_corpus_path = corpus_path.replace(".txt", "_encoded.pkl")
    df_encoded.to_pickle(encoded_corpus_path)

    logger.info('Preprocessing was successful.')

if __name__ == '__main__':
    run_pre_processing()
