#!/usr/bin/env python3
# -*- coding: utf-8 -*
"""Documents generator."""
import fpdf
import numpy as np
from pdf2image import convert_from_bytes


def gen_word(word_len=5):
    """Generate word from random characters with length word_len."""
    return ''.join([chr(x) for x in np.random.choice(np.arange(33, 127),
                                                     word_len)])


def gen_page(dpi=250, mean_word_len=5, font=None):
    """Generate page with random text.

    Parameters
    ----------
    dpi : int
        DPI for generated image.
    mean_word_len : int
        Mean length of generated words
        (randint from [1; mean_word_len * 2 + 1]).
    font : str
        Name of font for fdpf.

    Returns
    -------
    array-like
        Array of printed words.

    """
    pdf = fpdf.FPDF()
    pdf.add_page()

    # select and set font
    fonts = list(fpdf.fonts.fpdf_charwidths.keys())
    if font is None:
        font = np.random.choice(fonts)
    pdf.set_font(font, size=14)

    # generate text
    generated_text = []
    while True:
        word = gen_word(np.random.randint(1, mean_word_len * 2 + 1))
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
