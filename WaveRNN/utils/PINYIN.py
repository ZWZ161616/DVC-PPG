# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:11:49 2019

@author: CCC
"""

# =============================================================================
# CH 2 pinyin
# =============================================================================
import pinyin as py
from pypinyin import pinyin, lazy_pinyin, Style

def CH2PY(word):
#    PY_result = py.get(word)
#    PY_result = py.get(word, format="strip", delimiter=" ")
#    PY_result = py.get(word, format="numerical")

    PY_result = ' '.join(lazy_pinyin(word))
#    PY_result = lazy_pinyin(word)
    return PY_result
# =============================================================================
# wade_giles pinyin
# =============================================================================
#from pypinyin import lazy_pinyin
from pypinyin.style import register
from pypinyin.style._utils import get_initials, get_finals
from pypinyin.style.others import converter


#def to_wade_giles(pinyin_in, **kwargs):
initials_convert_map = {
        'ba':'pa',
        'bei':'pei',
        'ben':'pen',
        'bi':'pi',
        'bo':'po',
        'bu':'pu',
        'ca':'tsa',
        'ce':'tse',
        'chi':'chih',
        'chong':'chung',
        'chuo':'cho',
        'ci':'tzu',
        'cong':'tsung',
        'cou':'tsou',
        'cu':'tsu',
        'cuan':'tsuan',
        'cui':'tsui',
        'cun':'tsun',
        'cuo':'tso',
        'da':'ta',
        'de':'te',
        'di':'ti',
        'dong':'tung',
        'dou':'tou',
        'du':'tu',
        'duo':'to',
        'e':'eh',
        'en':'en',
        'er':'erh',
        'ga':'ka',
        'ge':'ko',
        'gong':'kung',
        'gou':'kou',
        'gu':'ku',
        'he':'ho',
        'ji':'chi',
        'ju':'chu',
        'ke':'ko',
        'kong':'kung ',
        'kui':'kuei',
        'lian':'lien',
        'lie':'lieh',
        'long':'lung',
        'lue':'lueh',
        'luo':'lo',
        'nie':'nieh',
        'nue':'nueh',
        'nuo':'no',
        'qi':'chi',
        'qu':'chu',
        'qu':'chu',
        'ran':'jan',
        'rao':'jao',
        'ren':'jen',
        'ri':'jih',
        'rong':'jung',
        'rou':'jou',
        'ru':'ju',
        'ruo':'jo',
        'shi':'shih',
        'si':'ssu',
        'song':'sung',
        'suo':'so',
        'tian':'tien',
        'tie':'tieh',
        'tong':'tung',
        'tuo':'to',
        'xi':'hsi',
        'xu':'hsu',
        'ye':'yeh',
        'yi':'i',
        'you':'yu',
        'yue':'yueh',
        'za':'tsa',
        'ze':'tse',
        'zha':'cha',
        'zhe':'che',
        'zhi':'chih',
        'zhong':'chung',
        'zhou':'chou',
        'zhu':'chu',
        'zhuo':'cho',
        'zi':'tzu',
        'zong':'tsung',
        'zou':'tsou',
        'zu':'tsu',
        'zuo':'tso',
        }
#    finals_convert_map = {}
##    print('orgin: ', pinyin_in)  # 原始有音标的拼音
#    pinyin_in = converter.to_normal(pinyin_in)
##    print('to: ', pinyin_in)    # 去掉音标
#    initials = get_initials(pinyin_in, False)  # 获取声母
#    finals = get_finals(pinyin_in, False)    # 获取韵母
#    wade_giles = '{}{}'.format(initials_convert_map.get(initials, initials),
#                               finals_convert_map.get(finals, finals))
##    print('Wade-Giles: ', wade_giles)  # 按规则转换的韦氏拼音
#    return wade_giles

initials_convert_map2 = {
    'b': 'p',
    'j': 'ch',
}
#finals_convert_map = {
#}

def to_wade_giles(pinyin, **kwargs):
#    print('orgin: ', pinyin)  # 原始有音标的拼音
    pinyin = converter.to_normal(pinyin)
#    print('to: ', pinyin)    # 去掉音标
#    initials = get_initials(pinyin, False)  # 获取声母
#    finals = get_finals(pinyin, False)    # 获取韵母
#    wade_giles = '{}{}'.format(initials_convert_map.get(initials, initials), finals_convert_map.get(finals, finals))
#    wade_giles = '{}'.format(initials_convert_map.get(pinyin, pinyin))
#    print('Wade-Giles: ', wade_giles)  # 按规则转换的韦氏拼音
    for key, value in initials_convert_map.items():
        pinyin = pinyin.replace(key, value)
    return pinyin

if __name__ == "__main__":
    word = '京北'
    word_ = CH2PY(word)
    word__ = to_wade_giles(word_)
    print('orgin      :', word)
    print('pinyin     :', word_)
    print('wade giles :', word__)
