// Copyright 2017 The Servo Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


//! Profiling example

#![allow(unused_imports)]

#![cfg_attr(feature="flame_it", feature(plugin, custom_attribute))]
#![cfg_attr(feature="flame_it", plugin(flamer))]


#[cfg(feature = "flame_it")]
extern crate flame;

extern crate unicode_bidi;


use std::fs::File;

use unicode_bidi::BidiInfo;


#[cfg(feature = "flame_it")]
fn main() {
    const BIDI_TEXT: str = include_str!("../data/udhr/bidi/udhr_pes_1.txt");

    flame::start("main(): BidiInfo::new()");
    let bidi_info = BidiInfo::new(BIDI_TEXT, None);
    flame::end("main(): BidiInfo::new()");

    flame::start("main(): iter bidi_info.paragraphs");
    for para in &bidi_info.paragraphs {
        let line = para.range.clone();
        bidi_info.reorder_line(para, line);
    }
    flame::end("main(): iter bidi_info.paragraphs");

    flame::dump_html(&mut File::create("flame-udhr-graph.html").unwrap()).unwrap();
}

#[cfg(not(feature = "flame_it"))]
// Allow example to compile
fn main() {}
