// Copyright 2023 The Mozilla Foundation. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::__seal_text_source;
use super::TextSource;

use sealed::sealed;

/// Implementation of TextSource for UTF-16 text in a [u16] array.
/// Note that there could be unpaired surrogates present!

// Convenience functions to check whether a UTF16 code unit is a surrogate.
#[inline]
fn is_high_surrogate(code: u16) -> bool {
    (code & 0xFC00) == 0xD800
}
#[inline]
fn is_low_surrogate(code: u16) -> bool {
    (code & 0xFC00) == 0xDC00
}

#[sealed]
impl<'text> TextSource<'text> for [u16] {
    type CharIter = Utf16CharIter<'text>;
    type CharIndexIter = Utf16CharIndexIter<'text>;
    type IndexLenIter = Utf16IndexLenIter<'text>;

    #[inline]
    fn len(&self) -> usize {
        (self as &[u16]).len()
    }
    fn char_at(&self, index: usize) -> Option<(char, usize)> {
        if index >= self.len() {
            return None;
        }
        // Get the indicated code unit and try simply converting it to a char;
        // this will fail if it is half of a surrogate pair.
        let c = self[index];
        if let Some(ch) = char::from_u32(c.into()) {
            return Some((ch, 1));
        }
        // If it's a low surrogate, and was immediately preceded by a high surrogate,
        // then we're in the middle of a (valid) character, and should return None.
        if is_low_surrogate(c) && index > 0 && is_high_surrogate(self[index - 1]) {
            return None;
        }
        // Otherwise, try to decode, returning REPLACEMENT_CHARACTER for errors.
        if let Some(ch) = char::decode_utf16(self[index..].iter().cloned()).next() {
            if let Ok(ch) = ch {
                // This must be a surrogate pair, otherwise char::from_u32() above should
                // have succeeded!
                debug_assert!(ch.len_utf16() == 2, "BMP should have already been handled");
                return Some((ch, ch.len_utf16()));
            }
        } else {
            debug_assert!(
                false,
                "Why did decode_utf16 return None when we're not at the end?"
            );
            return None;
        }
        // Failed to decode UTF-16: we must have encountered an unpaired surrogate.
        // Return REPLACEMENT_CHARACTER (not None), to continue processing the following text
        // and keep indexing correct.
        Some((char::REPLACEMENT_CHARACTER, 1))
    }
    #[inline]
    fn chars(&'text self) -> Self::CharIter {
        Utf16CharIter::new(&self)
    }
    #[inline]
    fn char_indices(&'text self) -> Self::CharIndexIter {
        Utf16CharIndexIter::new(&self)
    }
    #[inline]
    fn indices_lengths(&'text self) -> Self::IndexLenIter {
        Utf16IndexLenIter::new(&self)
    }
    #[inline]
    fn char_len(ch: char) -> usize {
        ch.len_utf16()
    }
}

/// Iterator over UTF-16 text in a [u16] slice, returning (index, char_len) tuple.
#[derive(Debug)]
pub struct Utf16IndexLenIter<'text> {
    text: &'text [u16],
    cur_pos: usize,
}

impl<'text> Utf16IndexLenIter<'text> {
    #[inline]
    pub fn new(text: &'text [u16]) -> Self {
        Utf16IndexLenIter { text, cur_pos: 0 }
    }
}

impl Iterator for Utf16IndexLenIter<'_> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((_, char_len)) = self.text.char_at(self.cur_pos) {
            let result = (self.cur_pos, char_len);
            self.cur_pos += char_len;
            return Some(result);
        }
        None
    }
}

/// Iterator over UTF-16 text in a [u16] slice, returning (index, char) tuple.
#[derive(Debug)]
pub struct Utf16CharIndexIter<'text> {
    text: &'text [u16],
    cur_pos: usize,
}

impl<'text> Utf16CharIndexIter<'text> {
    pub fn new(text: &'text [u16]) -> Self {
        Utf16CharIndexIter { text, cur_pos: 0 }
    }
}

impl Iterator for Utf16CharIndexIter<'_> {
    type Item = (usize, char);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((ch, char_len)) = self.text.char_at(self.cur_pos) {
            let result = (self.cur_pos, ch);
            self.cur_pos += char_len;
            return Some(result);
        }
        None
    }
}

/// Iterator over UTF-16 text in a [u16] slice, returning Unicode chars.
/// (Unlike the other iterators above, this also supports reverse iteration.)
#[derive(Debug)]
pub struct Utf16CharIter<'text> {
    text: &'text [u16],
    cur_pos: usize,
    end_pos: usize,
}

impl<'text> Utf16CharIter<'text> {
    pub fn new(text: &'text [u16]) -> Self {
        Utf16CharIter {
            text,
            cur_pos: 0,
            end_pos: text.len(),
        }
    }
}

impl Iterator for Utf16CharIter<'_> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((ch, char_len)) = self.text.char_at(self.cur_pos) {
            self.cur_pos += char_len;
            return Some(ch);
        }
        None
    }
}

impl DoubleEndedIterator for Utf16CharIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end_pos <= self.cur_pos {
            return None;
        }
        self.end_pos -= 1;
        if let Some(ch) = char::from_u32(self.text[self.end_pos] as u32) {
            return Some(ch);
        }
        if self.end_pos > self.cur_pos {
            if let Some((ch, char_len)) = self.text.char_at(self.end_pos - 1) {
                if char_len == 2 {
                    self.end_pos -= 1;
                    return Some(ch);
                }
            }
        }
        Some(char::REPLACEMENT_CHARACTER)
    }
}
