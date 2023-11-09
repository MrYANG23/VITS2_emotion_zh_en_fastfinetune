import re
from text.english import english_cleaners2, english_cleaners3
from text.mandarin import chinese_to_phonemes, chinese_to_phonemes_v2



def zh_en_mix_cleaners_v2(text):
    text = re.sub(r'\[zh\](.*?)\[zh\]',
                  lambda x: chinese_to_phonemes_v2(text=x.group(1)) + ' ', text)

    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_cleaners3(x.group(1)) + ' ', text)
    return text

#     pass
# if __name__ == '__main__':
#     text=zh_en_mix_cleaners(text='[zh]⼤家好，我是数字人主播榕榕，我由[zh][EN]cyber[EN][zh]智能团队创造，很高兴与大家见面[zh]')