
from text import cleaners
from text.ch_en_mix_symbols_v2 import mix_symbols_v2


# _symbol_to_id_mix = {s: i for i, s in enumerate(mix_symbols)}
# _id_to_symbol_mix = {i: s for i, s in enumerate(mix_symbols)}


def cleaned_text_to_sequence_mix_v2(cleaned_text):
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
      Args:
        text: string to convert to a sequence
      Returns:
        List of integers corresponding to the symbols in the text
    '''
    symbol_to_id = {s: i for i, s in enumerate(mix_symbols_v2)}
    sequence = [symbol_to_id[symbol] for symbol in cleaned_text.split('#') if symbol in symbol_to_id.keys()]
    return sequence


def _clean_text_mix(text, cleaner_names):
    # print('--------------clean--------------')
    # print('-------------cleaner_names',cleaner_names)
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text

################zh_en_mix-v2################################
