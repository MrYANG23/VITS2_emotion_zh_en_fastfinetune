import re
from text.english import english_cleaners2, english_cleaners3
from text.mandarin import chinese_to_phonemes, chinese_to_phonemes_v2
from tn.chinese.normalizer import Normalizer
normalizer = Normalizer()
from text.mix_symbols import language_id_map

def is_chinese(char):
    if char >= '\u4e00' and char <= '\u9fa5':
        return True
    else:
        return False


def is_alphabet( char):
    if (char >= '\u0041' and char <= '\u005a') or (char >= '\u0061' and
                                                   char <= '\u007a'):
        return True
    else:
        return False

def is_other(char):
    if not (is_chinese(char) or is_alphabet(char)):
        return True
    else:
        return False

def split_by_lang( text):
    # sentence --> [ch_part, en_part, ch_part, ...]
    segments = []
    types = []

    # Determine the type of each character. type: chinese, alphabet, other.
    for ch in text:
        if is_chinese(ch):
            types.append("zh")
        elif is_alphabet(ch):
            types.append("en")
        else:
            types.append("other")

    assert len(types) == len(text)

    flag = 0
    temp_seg = ""
    temp_lang = ""

    for i in range(len(text)):
        # find the first char of the seg
        if flag == 0:
            temp_seg += text[i]
            temp_lang = types[i]
            flag = 1
        else:
            if temp_lang == "other":
                # text start is not lang.
                temp_seg += text[i]
                if types[i] != temp_lang:
                    temp_lang = types[i]
            else:
                if types[i] == temp_lang or types[i] == "other":
                    # merge same lang or other
                    temp_seg += text[i]
                else:
                    # change lang
                    segments.append((temp_seg, temp_lang))
                    temp_seg = text[i]
                    temp_lang = types[i]  # new lang

    segments.append((temp_seg, temp_lang))
    return segments



# def mix_cleaners(text):
#     text = re.sub(r'\[zh\](.*?)\[zh\]',
#                   lambda x: chinese_to_phonemes_v2(text=x.group(1)) + ' ', text)
#     print('text-zh',text)
#     text = re.sub(r'\[EN\](.*?)\[EN\]',
#                   lambda x: english_cleaners3(x.group(1)) + ' ', text)
#     print('text-en',text)
#
#
#     return text



def mix_cleaners(text):
    mix_phonemes=['sil']
    mix_lang=[1]
    segments=split_by_lang(text=text)
    print('segments',segments)

    for per_text in segments:
        text=per_text[0]
        lang=per_text[1]
        if lang=='zh':
            new_text=normalizer.normalize(text)
            print('new_text',new_text)
            zh_phonemes=chinese_to_phonemes_v2(new_text)
            print('zh_phonemes',zh_phonemes)
            zh_lang=[language_id_map[lang.upper()] for i in zh_phonemes ]
            mix_phonemes.extend(zh_phonemes)
            mix_lang.extend(zh_lang)
        if lang=='en':
            en_phonemes=english_cleaners3(text)
            en_lang=[language_id_map[lang.upper()] for i in en_phonemes]
            mix_phonemes.extend(en_phonemes)
            mix_lang.extend(en_lang)
    mix_phonemes.extend(['eos'])
    mix_lang.extend([1])
    assert len(mix_phonemes)==len(mix_lang)
    return mix_phonemes,mix_lang

#     pass
if __name__ == '__main__':
    mix_cleaners('⼤家好，我是数字人主播榕榕，我由cyber智能团队创造，很高兴与大家见面')
