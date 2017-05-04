// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![forbid(unsafe_code)]

//! TODO

pub use char_data_tables::{BidiClass, bidi_class_table, UNICODE_VERSION};

use std::cmp::Ordering::{Equal, Less, Greater};

use BidiClass::*;

/// Find the BidiClass of a single char.
pub fn bidi_class(c: char) -> BidiClass {
    bsearch_range_value_table(c, bidi_class_table)
}

fn bsearch_range_value_table(c: char, r: &'static [(char, char, BidiClass)]) -> BidiClass {
    match r.binary_search_by(
        |&(lo, hi, _)| if lo <= c && c <= hi {
            Equal
        } else if hi < c {
            Less
        } else {
            Greater
        },
    ) {
        Ok(idx) => {
            let (_, _, cat) = r[idx];
            cat
        }
        // UCD/extracted/DerivedBidiClass.txt: "All code points not explicitly listed
        // for Bidi_Class have the value Left_To_Right (L)."
        Err(_) => L,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bidi_class() {
        assert_eq!(bidi_class('c'), L);
        assert_eq!(bidi_class('\u{05D1}'), R);
        assert_eq!(bidi_class('\u{0627}'), AL);
    }

}
