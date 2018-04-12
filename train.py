from neural_machine_translation import NeuralMachineTranslation
from data_importer import SentenceTranslationDataset
from context_enhanced_gru_cell_a import ContextEnhancedGRUCellA
from context_enhanced_gru_cell_b import ContextEnhancedGRUCellB

if __name__ == '__main__':

    vocab_size = 3e3
    target_language = 'es'
    num_epochs = 70
    train_epoch_size = 1000
    test_epoch_size = 100
    learning_rate = 1e-5

    identifier = 'es_attention_b_stacked_large'

    # old_encoder_weights = 'weights/encoder_weights'
    # old_decoder_weights = 'weights/decoder_weights'

    targ_lang_vocab_path="./processed_data/en-es/vocab.{}".format(target_language)
    targ_lang_embedding_path="./processed_data/en-es/embedding.vocab.{}".format(target_language)
    targ_lang_text_train_path="./processed_data/en-es/text.{}.train".format(target_language)
    targ_lang_text_test_path="./processed_data/en-es/text.{}.test".format(target_language)
    
    train_data_loader = SentenceTranslationDataset(
        targ_lang_vocab_path=targ_lang_vocab_path,
        targ_lang_embedding_path=targ_lang_embedding_path,
        src_lang_text_path="./processed_data/en-es/text.en.train",
        targ_lang_text_path=targ_lang_text_train_path,
        max_vocab_size=vocab_size,
        max_n_sentences=1e6,
        max_src_sentence_len=30,
        max_targ_sentence_len=30
    )

    test_data_loader = SentenceTranslationDataset(
        targ_lang_vocab_path=targ_lang_vocab_path,
        targ_lang_embedding_path=targ_lang_embedding_path,
        src_lang_text_path="./processed_data/en-es/text.en.test",
        targ_lang_text_path=targ_lang_text_test_path,
        max_vocab_size=vocab_size,
        max_n_sentences=1e6,
        max_src_sentence_len=30,
        max_targ_sentence_len=30
    )
    print "data loaded"

    # nmt = NeuralMachineTranslation(train_data_loader, test_data_loader, vocab_size)
    # nmt = NeuralMachineTranslation(train_data_loader, test_data_loader, vocab_size, use_attention_mechanism=True, bottom_time_cell=ContextEnhancedGRUCellA)
    # nmt = NeuralMachineTranslation(train_data_loader, test_data_loader, vocab_size, use_attention_mechanism=True, bottom_time_cell=ContextEnhancedGRUCellA, stacked_time_cell=ContextEnhancedGRUCellA)
    # nmt = NeuralMachineTranslation(train_data_loader, test_data_loader, vocab_size, use_attention_mechanism=True, bottom_time_cell=ContextEnhancedGRUCellB)
    # nmt = NeuralMachineTranslation(train_data_loader, test_data_loader, vocab_size, use_attention_mechanism=True, bottom_time_cell=ContextEnhancedGRUCellB, stacked_time_cell=ContextEnhancedGRUCellB)

    n_encoder_layers=3
    enc_hidden_dimension_size=512
    n_decoder_layers=3
    dec_hidden_dimension_size=1024
    nmt = NeuralMachineTranslation(train_data_loader, test_data_loader, vocab_size, \
            n_encoder_layers=n_encoder_layers, enc_hidden_dimension_size=enc_hidden_dimension_size, \
            n_decoder_layers=n_decoder_layers, dec_hidden_dimension_size=dec_hidden_dimension_size, \
            use_attention_mechanism=True, \
            bottom_time_cell=ContextEnhancedGRUCellB, stacked_time_cell=ContextEnhancedGRUCellB)

    print "nmt initialized"

    print "training {}\n".format(identifier)

    nmt.train(num_epochs, train_epoch_size, test_epoch_size, learning_rate, identifier)
    # sample_src_text, sample_targ_text, sample_pred_text = nmt.sample_train_translation()
    # print sample_src_text[0]
    # print sample_targ_text[0]
    # print sample_pred_text[0]
