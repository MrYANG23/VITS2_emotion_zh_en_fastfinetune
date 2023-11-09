""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
from g2p_en import G2p
# _pause = ["#"]
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '

en_backend=G2p()
letters=en_backend.phonemes

'''
['_', ';', ':', ',', '.', '!', '?', '¡', '¿', '—', '…', '"', '«', '»', '“', '”', ' ', '<pad>', '<unk>', '<s>', '</s>', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 
'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 
'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 
'UH1', 'UH2', 'UW', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

'''

#
# # Export all symbols:
symbols_en = [_pad] + list(_punctuation) + letters
#print('symbols_en)',symbols_en)
#
# # Special symbol ids
SPACE_ID = symbols_en.index(" ")
#print('SPACE_ID',SPACE_ID)


CN_PUNCT = ["、", "，", "；", "：", "。", "？", "！"]
_pause = ["sil", "eos", "sp", "#0", "#1", "#2", "#3"]


_initials = [
    "^",
    "b",
    "c",
    "ch",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "x",
    "z",
    "zh",
]

_tones = ["1", "2", "3", "4", "5"]
_finals = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "ii",
    "iii",
    "in",
    "ing",
    "iong",
    "iou",
    "o",
    "ong",
    "ou",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "uei",
    "uen",
    "ueng",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
]

symbols =CN_PUNCT+_pause + _initials + [i + j for i in _finals for j in _tones]


final_set=set()
for per_en_symbols in symbols_en:
    final_set.add(per_en_symbols)

for per_zh_symbols in symbols:
    final_set.add(per_zh_symbols)



final_list=symbols_en+symbols
#print('(final_list):',final_list)
#print(len(final_list))
save_symbols=list(set(final_list))
#print('save_symbols)',save_symbols)
#print(len(save_symbols))

mix_symbols_v2=final_list

