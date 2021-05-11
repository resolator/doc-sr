#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Documents generator."""
import fpdf
import numpy as np
from pdf2image import convert_from_bytes


def gen_word(word_len=5, word_chars=False):
    """Generate word from random characters with length word_len."""
    if word_chars:
        char_range = np.concatenate([np.arange(65, 91), np.arange(97, 123)])
    else:
        char_range = np.arange(33, 127)

    return ''.join([chr(x) for x in np.random.choice(char_range, word_len)])


def gen_page(dpi=250, mean_word_len=5, font_size=14, word_chars=False,
             page_format='A5'):
    """Generate page with random text.

    Parameters
    ----------
    dpi : int
        DPI for generated image.
    mean_word_len : int
        Mean length of generated words
        (randint from [1; mean_word_len * 2 + 1]).
    font_size : int
        Font size for generated text.
    word_chars : bool
        Generate only in-word characters.

    Returns
    -------
    (numpy.array, list)
        An image with generated text and an array of generated words.

    """
    pdf = fpdf.FPDF(format=page_format)
    pdf.add_page()

    # select and set font
    pdf.set_font('Courier', size=font_size)

    # generate text
    generated_text = []
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
