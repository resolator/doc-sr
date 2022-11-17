#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Documents generator."""
import cv2
import fpdf
import numpy as np
from xml.etree import ElementTree as ET
from pdf2image import convert_from_bytes


ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'v', 'w', 'x', 'y', 'z']


def gen_word(word_len=5, word_chars=False):
    """Generate word from random characters with length word_len."""
    if word_chars:
        char_range = np.concatenate([np.arange(65, 91), np.arange(97, 123)])
    else:
        char_range = np.arange(33, 127)

    return ''.join([chr(x) for x in np.random.choice(char_range, word_len)])


def read_text_from_corpus(corpus_xml_path):
    """Read page from "British National Corpus" xml.

    Parameters
    ----------
    corpus_xml_path : pathlib.Path or str
        Path to xml from "British National Corpus".

    Returns
    -------
    iterator
        Iterator of words from xml.

    """
    root = ET.parse(corpus_xml_path).getroot()

    for page in root[1]:
        for line in page:
            for sub_line in line:
                for word in sub_line:
                    if isinstance(word.text, str):
                        yield word.text
                    else:
                        for sub_word in word:
                            if isinstance(sub_word.text, str):
                                yield sub_word.text
                if sub_line.tag == 'head':
                    yield '\n'
            if line.tag == 'head':
                yield '\n'


def init_pdf(font_size=12, page_format='A5'):
    """Initialize pdf."""
    assert page_format in ['A3', 'A4', 'A5'], \
        'page format should be A3, A4 or A5.'

    pdf = fpdf.FPDF(format=page_format)
    pdf.add_page()

    # select and set font
    pdf.set_font('Courier', size=font_size)

    return pdf


def gen_text_from_xml(pdf, corpus_xml_path):
    """Generate text from xml file and print on pdf."""
    text = read_text_from_corpus(corpus_xml_path)
    generated_text = []

    for word in text:
        # decode word to latin-1
        try:
            word.encode('latin-1')
        except UnicodeEncodeError:
            replacers = {'\u2013': '-',
                         '—': '-',
                         '‘': '\'',
                         '’': '\'',
                         '\u2026': '...',
                         '\u0394': '',
                         '\u215b': '1/8'}
            for k, v in replacers.items():
                word = word.replace(k, v)

        if word[-1] == '.':
            word = word + ' '

        pdf.write(pdf.font_size, word)
        if pdf.page != 1:
            break

        generated_text.append(word)

    return generated_text


def gen_page(dpi=250,
             corpus_xml_path=None,
             mean_word_len=5,
             font_size=12,
             word_chars=False,
             page_format='A5'):
    """Generate page with random text.

    Parameters
    ----------
    dpi : int
        DPI for generated image.
    corpus_xml_path : pathlib.Path or str or None
        Path to xml from "British National Corpus".
    mean_word_len : int
        Mean length of generated words
        (randint from [1; mean_word_len * 2 + 1]).
    font_size : int
        Font size for generated text.
    word_chars : bool
        Generate only in-word characters.
    page_format : str
        Page format (A3, A4, A5) for generation.

    Returns
    -------
    (numpy.array, list)
        An image with generated text and an array of generated words.

    """
    pdf = init_pdf(font_size, page_format)

    # generate text
    generated_text = []
    if corpus_xml_path is not None:
        while len(generated_text) < 30:
            generated_text = gen_text_from_xml(pdf, corpus_xml_path)

    else:
        while True:
            word = gen_word(np.random.randint(1, mean_word_len * 2 + 1),
                            word_chars)
            pdf.write(pdf.font_size, word + ' ')
            if pdf.page != 1:
                break

            generated_text.append(word)

    # get image
    pdf.close()
    pdf_s = pdf.output(dest='S').encode('raw_unicode_escape')
    first_page = convert_from_bytes(pdf_s, dpi=dpi)[0]
    img = np.array(first_page)

    return img, generated_text


def gen_line(corpus_xml_path,
             size=(64, 900),
             font_scale=1,
             mean_word_len=5,
             word_chars=False):
    valid = False
    img, sentence = None, None

    while not valid:
        img = (np.ones(size) * 255.0).astype(np.uint8)

        total_words = np.random.randint(4, 9)
        sentence = []
        border_size = 30

        if corpus_xml_path is not None:
            text = read_text_from_corpus(corpus_xml_path)
        else:
            text = [''.join(np.random.choice(ALPHABET, np.random.randint(1, mean_word_len * 2 + 1)))
                    for _ in range(total_words)]
            # text = [gen_word(np.random.randint(1, mean_word_len * 2 + 1),
            #                  word_chars) for _ in range(total_words)]

        for word in text:
            if word == '\n':
                sentence[-1] = sentence[-1] + '.'
            else:
                sentence.append(word.replace(' ', ''))

            if len(sentence) == total_words:
                break

        cv2.putText(
            img,
            ' '.join(sentence),
            (border_size, 40),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        # check
        img_border = img[:, -border_size:]
        needed_sum = (img_border.shape[0] * img_border.shape[1] * 255)
        if img_border.sum() == needed_sum:
            valid = True
        else:
            print('Not valid img, regenerating')

    return img, sentence
