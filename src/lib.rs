// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_use] extern crate matches;

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

/// 3.3.2 Explicit Levels and Directions
///
/// http://www.unicode.org/reports/tr9/#Explicit_Levels_and_Directions
mod explicit {
    use super::{BidiClass, is_rtl, paragraph_level};
    use super::BidiClass::*;

    /// Output of the explicit levels algorithm.
    pub struct Result {
        pub levels: Vec<u8>,
        pub classes: Vec<BidiClass>,
    }

    /// Compute explicit embedding levels for one paragraph of text (X1-X8).
    ///
    /// `classes[i]` must contain the BidiClass of the char at byte index `i`,
    /// for each char in `text`.
    pub fn compute(text: &str, para_level: u8, classes: &[BidiClass]) -> Result {
        assert!(text.len() == classes.len());

        let mut result = Result {
            levels: vec![para_level; text.len()],
            classes: Vec::from(classes),
        };

        // http://www.unicode.org/reports/tr9/#X1
        let mut stack = DirectionalStatusStack::new();
        stack.push(para_level, OverrideStatus::Neutral);

        let mut overflow_isolate_count = 0u32;
        let mut overflow_embedding_count = 0u32;
        let mut valid_isolate_count = 0u32;

        for (i, c) in text.char_indices() {
            match classes[i] {
                // Rules X2-X5c
                RLE | LRE | RLO | LRO | RLI | LRI | FSI => {
                    let is_rtl = match classes[i] {
                        RLE | RLO | RLI => true,
                        FSI => {
                            // TODO: Find the matching PDI.
                            is_rtl(paragraph_level(&classes[i..]))
                        }
                        _ => false
                    };

                    let last_level = stack.last().level;
                    let new_level = match is_rtl {
                        true  => next_rtl_level(last_level),
                        false => next_ltr_level(last_level)
                    };

                    // X5a-X5c: Isolate initiators get the level of the last entry on the stack.
                    let is_isolate = matches!(classes[i], RLI | LRI | FSI);
                    if is_isolate {
                        result.levels[i] = last_level;
                    }

                    if valid(new_level) && overflow_isolate_count == 0 && overflow_embedding_count == 0 {
                        stack.push(new_level, match classes[i] {
                            RLO => OverrideStatus::RTL,
                            LRO => OverrideStatus::LTR,
                            RLI | LRI | FSI => OverrideStatus::Isolate,
                            _ => OverrideStatus::Neutral
                        });
                        if is_isolate {
                            valid_isolate_count += 1;
                        } else {
                            result.levels[i] = new_level;
                        }
                    } else if is_isolate {
                        overflow_isolate_count += 1;
                    } else if overflow_isolate_count == 0 {
                        overflow_embedding_count += 1;
                    }
                }
                // http://www.unicode.org/reports/tr9/#X6a
                PDI => {
                    if overflow_isolate_count > 0 {
                        overflow_isolate_count -= 1;
                        continue
                    }
                    if valid_isolate_count == 0 {
                        continue
                    }
                    overflow_embedding_count = 0;
                    loop {
                        // Pop everything up to and including the last Isolate status.
                        match stack.vec.pop() {
                            Some(Status { status: OverrideStatus::Isolate, .. }) => break,
                            None => break,
                            _ => continue
                        }
                    }
                    valid_isolate_count -= 1;
                    result.levels[i] = stack.last().level;
                }
                // http://www.unicode.org/reports/tr9/#X7
                PDF => {
                    if overflow_isolate_count > 0 {
                        continue
                    }
                    if overflow_embedding_count > 0 {
                        overflow_embedding_count -= 1;
                        continue
                    }
                    if stack.last().status != OverrideStatus::Isolate && stack.vec.len() >= 2 {
                        stack.vec.pop();
                    }
                    result.levels[i] = stack.last().level;
                }
                // http://www.unicode.org/reports/tr9/#X6
                B | BN => {}
                _ => {
                    let last = stack.last();
                    result.levels[i] = last.level;
                    match last.status {
                        OverrideStatus::RTL => result.classes[i] = R,
                        OverrideStatus::LTR => result.classes[i] = L,
                        _ => {}
                    }
                }
            }
            // Handle multi-byte characters.
            for j in 1..c.len_utf8() {
                result.levels[i+j] = result.levels[i];
                // TODO: Only do this if result.classes changed?
                result.classes[i+j] = result.classes[i];
            }
        }
        result
    }

    /// Maximum depth of the directional status stack.
    pub const MAX_DEPTH: u8 = 125;

    /// Levels from 0 through max_depth are valid at this stage.
    /// http://www.unicode.org/reports/tr9/#X1
    fn valid(level: u8) -> bool { level <= MAX_DEPTH }

    /// The next odd level greater than `level`.
    fn next_rtl_level(level: u8) -> u8 { (level + 1) |  1 }

    /// The next odd level greater than `level`.
    fn next_ltr_level(level: u8) -> u8 { (level + 2) & !1 }

    /// Entries in the directional status stack:
    struct Status {
        level: u8,
        status: OverrideStatus,
    }

    #[derive(PartialEq)]
    enum OverrideStatus { Neutral, RTL, LTR, Isolate }

    struct DirectionalStatusStack {
        vec: Vec<Status>,
    }

    impl DirectionalStatusStack {
        fn new() -> Self {
            DirectionalStatusStack {
                vec: Vec::with_capacity(MAX_DEPTH as usize + 2)
            }
        }
        fn push(&mut self, level: u8, status: OverrideStatus) {
            self.vec.push(Status { level: level, status: status });
        }
        fn last(&self) -> &Status {
            self.vec.last().unwrap()
        }
    }
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
