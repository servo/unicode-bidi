// Copyright 2017 The Servo Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg(all(test, feature = "bench_it"))]
#![feature(test)]

extern crate test;
extern crate unicode_bidi;

use test::Bencher;

use unicode_bidi::utf16::BidiInfo as BidiInfoU16;
use unicode_bidi::BidiInfo;

fn to_utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

const LTR_TEXTS: &[&str] = &["abc\ndef\nghi", "abc 123\ndef 456\nghi 789"];

const BIDI_TEXTS: &[&str] = &["ابجد\nهوز\nحتی", "ابجد ۱۲۳\nهوز ۴۵۶\nحتی ۷۸۹"];

fn bench_bidi_info_new(b: &mut Bencher, texts: &[&str]) {
    for text in texts {
        b.iter(|| {
            BidiInfo::new(text, None);
        });
    }
}

fn bench_bidi_info_new_u16(b: &mut Bencher, texts: &Vec<Vec<u16>>) {
    for text in texts {
        b.iter(|| {
            BidiInfoU16::new(&text, None);
        });
    }
}

fn bench_reorder_line(b: &mut Bencher, texts: &[&str]) {
    for text in texts {
        let bidi_info = BidiInfo::new(text, None);
        b.iter(|| {
            for para in &bidi_info.paragraphs {
                let line = para.range.clone();
                bidi_info.reorder_line(para, line);
            }
        });
    }
}

fn bench_reorder_line_u16(b: &mut Bencher, texts: &Vec<Vec<u16>>) {
    for text in texts {
        let bidi_info = BidiInfoU16::new(text, None);
        b.iter(|| {
            for para in &bidi_info.paragraphs {
                let line = para.range.clone();
                bidi_info.reorder_line(para, line);
            }
        });
    }
}

#[bench]
fn bench_1_bidi_info_new_for_ltr_texts(b: &mut Bencher) {
    bench_bidi_info_new(b, LTR_TEXTS);
}

#[bench]
fn bench_2_bidi_info_new_for_bidi_texts(b: &mut Bencher) {
    bench_bidi_info_new(b, BIDI_TEXTS);
}

#[bench]
fn bench_3_reorder_line_for_ltr_texts(b: &mut Bencher) {
    bench_reorder_line(b, LTR_TEXTS);
}

#[bench]
fn bench_4_reorder_line_for_bidi_texts(b: &mut Bencher) {
    bench_reorder_line(b, BIDI_TEXTS);
}

#[bench]
fn bench_5_bidi_info_new_for_ltr_texts_u16(b: &mut Bencher) {
    let texts_u16: Vec<_> = LTR_TEXTS.iter().map(|t| to_utf16(t)).collect();
    bench_bidi_info_new_u16(b, &texts_u16);
}

#[bench]
fn bench_6_bidi_info_new_for_bidi_texts_u16(b: &mut Bencher) {
    let texts_u16: Vec<_> = BIDI_TEXTS.iter().map(|t| to_utf16(t)).collect();
    bench_bidi_info_new_u16(b, &texts_u16);
}

#[bench]
fn bench_7_reorder_line_for_ltr_texts_u16(b: &mut Bencher) {
    let texts_u16: Vec<_> = LTR_TEXTS.iter().map(|t| to_utf16(t)).collect();
    bench_reorder_line_u16(b, &texts_u16);
}

#[bench]
fn bench_8_reorder_line_for_bidi_texts_u16(b: &mut Bencher) {
    let texts_u16: Vec<_> = BIDI_TEXTS.iter().map(|t| to_utf16(t)).collect();
    bench_reorder_line_u16(b, &texts_u16);
}
