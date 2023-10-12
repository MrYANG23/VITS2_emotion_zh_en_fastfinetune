from typing import List

import jieba
import pypinyin

from .pinyinToPhonemes import PINYIN_DICT


def _chinese_character_to_pinyin(text: str) -> List[str]:
    pinyins = pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)
    #print('pinyins:',pinyins)
    pinyins_flat_list = [item for sublist in pinyins for item in sublist]
    return pinyins_flat_list


def _chinese_pinyin_to_phoneme(pinyin: str) -> str:
    segment = pinyin[:-1]
    tone = pinyin[-1]
    phoneme = PINYIN_DICT.get(segment, [""])[0]
    return phoneme + tone


def chinese_text_to_phonemes(text: str, seperator: str = "|") -> str:
    tokenized_text = jieba.cut(text, HMM=False)
    #print('tokenized_text:',tokenized_text)

    tokenized_text = " ".join(tokenized_text)
    #print('tokenized_text:',tokenized_text)
    pinyined_text: List[str] = _chinese_character_to_pinyin(tokenized_text)
    #print('pinyined_text:',pinyined_text)
   # exit()
    results: List[str] = []

    for token in pinyined_text:
        if token[-1] in "12345":  # TODO transform to is_pinyin()
            pinyin_phonemes = _chinese_pinyin_to_phoneme(token)

            results += list(pinyin_phonemes)
        else:  # is ponctuation or other
            results += list(token)

    return seperator.join(results)

if __name__ == '__main__':
    text='心理疏导很重要，我认为全国公众应当参与进来，帮着上海一些市民开展这种疏导，增加他们的耐心和信心。一些上海人有难处有情绪，我们应就事论事帮着问题得到关注，带来解决。但要注意别拱火，别助推上纲上线，更不能幸灾乐祸。还是要多给上海一点时间。我相信，病毒淹没不了动态清零，相反，动态清零终将在在上海踩出一条更宽的路。'
    print(chinese_text_to_phonemes(text=text))