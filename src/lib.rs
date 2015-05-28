// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod tables;
pub use tables::UNICODE_VERSION;
pub use tables::bidi::{BidiClass, bidi_class};

use BidiClass::*;

#[inline]
/// Even levels are left-to-right, and odd levels are right-to-left.
///
/// http://www.unicode.org/reports/tr9/#BD2
pub fn is_rtl(level: u8) -> bool { level % 2 == 1 }

/// The default embedding level for a paragraph.
///
/// http://www.unicode.org/reports/tr9/#The_Paragraph_Level
fn paragraph_level(classes: &[BidiClass]) -> u8 {
    // P2. Find the first character of type L, AL, or R, skipping characters between an isolate
    // initiator and its matching PDI.
    let mut isolate_level = 0u32;
    for &class in classes {
        match (isolate_level, class) {
            (0, L) => return 0,
            (0, R) => return 1,
            (0, AL) => return 1,
            // Push a directional isolate:
            (_, LRI) | (_, RLI) | (_, FSI) => isolate_level += 1,
            // Ignore an unmatched PDI:
            (0, PDI) => continue,
            // Pop a directional isolate:
            (_, PDI) => isolate_level -= 1,
            _ => continue
        }
    }
    // P3. If no character is found in P2, set the embedding level to zero.
    0
}

#[cfg(test)]
mod test {
    use super::{bidi_class, paragraph_level};
    use super::BidiClass::*;

    #[test]
    fn test_paragraph_level() {
        assert_eq!(paragraph_level(&[]), 0);
        assert_eq!(paragraph_level(&[WS]), 0);
        assert_eq!(paragraph_level(&[L, L, L]), 0);
        assert_eq!(paragraph_level(&[EN, EN]), 0);

        assert_eq!(paragraph_level(&[R, L]), 1);
        assert_eq!(paragraph_level(&[EN, EN, R, EN]), 1);
        assert_eq!(paragraph_level(&[AL]), 1);
        assert_eq!(paragraph_level(&[WS, WS, AL, L]), 1);

        // Ignore characters between directional isolates:
        assert_eq!(paragraph_level(&[LRI, L, PDI, R]), 1);
        assert_eq!(paragraph_level(&[LRI, AL, R, PDI]), 0);
    }

    #[test]
    fn test_bidi_class() {
        assert_eq!(bidi_class('c'), L);
        assert_eq!(bidi_class('\u{05D1}'), R);
        assert_eq!(bidi_class('\u{0627}'), AL);
    }
}
