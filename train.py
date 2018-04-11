from neural_machine_translation import NeuralMachineTranslation
from data_importer import SentenceTranslationDataset
from context_enhanced_gru_cell_a import ContextEnhancedGRUCellA
from context_enhanced_gru_cell_b import ContextEnhancedGRUCellB

if __name__ == '__main__':

    vocab_size = 3e3
    target_language = 'es'
    num_epochs = 10
    epoch_size = 10
    learning_rate = 1e-5

    # old_encoder_weights = 'weights/encoder_weights'
    # old_decoder_weights = 'weights/decoder_weights'

    new_encoder_weights = 'weights/encoder_weights'
    new_decoder_weights = 'weights/decoder_weights'

    targ_lang_vocab_path="./processed_data/en-es/vocab.{}".format(target_language)
    targ_lang_embedding_path="./processed_data/en-es/embedding.vocab.{}".format(target_language)
    targ_lang_text_path="./data/en-es/train.{}".format(target_language)
    data_loader = SentenceTranslationDataset(
        targ_lang_vocab_path=targ_lang_vocab_path,
        targ_lang_embedding_path=targ_lang_embedding_path,
        targ_lang_text_path=targ_lang_text_path,
        max_vocab_size=vocab_size,
        max_n_sentences=1e6,
        max_src_sentence_len=30,
        max_targ_sentence_len=30
    )
    print "data loaded"

    nmt = NeuralMachineTranslation(data_loader, vocab_size)
    # nmt = NeuralMachineTranslation(data_loader, vocab_size, encoder_weights=old_encoder_weights, decoder_weights=old_decoder_weights)
    # nmt = NeuralMachineTranslation(data_loader, vocab_size, use_attention_mechanism=True, bottom_time_cell=ContextEnhancedGRUCellA)
    # nmt = NeuralMachineTranslation(data_loader, vocab_size, use_attention_mechanism=True, bottom_time_cell=ContextEnhancedGRUCellB)
    print "nmt initialized"

    nmt.train(num_epochs, epoch_size, learning_rate)