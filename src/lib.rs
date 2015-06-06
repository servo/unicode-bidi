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

use std::iter::repeat;

/// Run the Unicode Bidirectional Algorithm.
pub fn process(text: &str, mut para_level: u8) {
    let initial_classes = classes(text);
    if para_level == IMPLICIT_LEVEL {
        para_level = paragraph_level(&initial_classes);
    }
    assert!(para_level <= 1);
    let explicit = explicit::compute(text, para_level, &initial_classes);
    let _sequences = prepare::isolating_run_sequences(para_level, &initial_classes,
                                                     &explicit.levels);
}

/// Pass this to make `process` determine the paragraph level implicitly.
pub const IMPLICIT_LEVEL: u8 = 2;

#[inline]
/// Even levels are left-to-right, and odd levels are right-to-left.
///
/// http://www.unicode.org/reports/tr9/#BD2
pub fn is_rtl(level: u8) -> bool { level % 2 == 1 }

/// Generate a character type based on a level (as specified in steps X10 and N2).
fn class_for_level(level: u8) -> BidiClass {
    if is_rtl(level) { R } else { L }
}

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

/// Returns a vector containing the BidiClass for each byte in the input text.
///
/// A multi-byte input char will have its BidiClass repeated multiple times in the output.
fn classes(text: &str) -> Vec<BidiClass> {
    let mut classes = Vec::with_capacity(text.len());
    for c in text.chars() {
        let class = bidi_class(c);
        classes.extend(repeat(class).take(c.len_utf8()));
    }
    assert!(classes.len() == text.len());
    classes
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

/// 3.3.3 Preparations for Implicit Processing
///
/// http://www.unicode.org/reports/tr9/#Preparations_for_Implicit_Processing
mod prepare {
    use super::{BidiClass, class_for_level};
    use super::BidiClass::*;
    use std::cmp::max;
    use std::ops::Range;

    /// Output of `isolating_run_sequences` (steps X9-X10)
    pub struct IsolatingRunSequence {
        pub runs: Vec<LevelRun>,
        pub sos: BidiClass, // Start-of-sequence type.
        pub eos: BidiClass, // End-of-sequence type.
    }

    /// A maximal substring of characters with the same embedding level.
    ///
    /// Represented as a range of byte indices within a paragraph.
    pub type LevelRun = Range<usize>;

    /// Compute the set of isolating run sequences.
    ///
    /// An isolating run sequence is a maximal sequence of level runs such that for all level runs
    /// except the last one in the sequence, the last character of the run is an isolate initiator
    /// whose matching PDI is the first character of the next level run in the sequence.
    pub fn isolating_run_sequences(para_level: u8, initial_classes: &[BidiClass], levels: &[u8])
        -> Vec<IsolatingRunSequence>
    {
        let runs = level_runs(levels, initial_classes);

        // Compute the set of isolating run sequences.
        // http://www.unicode.org/reports/tr9/#BD13
        let mut sequences = Vec::with_capacity(runs.len());
        for run in runs {
            // TODO: Actually check for isolate initiators and matching PDIs.
            // For now this is just a stub that puts each level run in a separate sequence.
            sequences.push(vec![run.clone()]);
        }

        // Determine the `sos` and `eos` class for each sequence.
        // http://www.unicode.org/reports/tr9/#X10
        return sequences.into_iter().map(|sequence| {
            assert!(sequence.len() > 0);
            let start = sequence[0].start;
            let end = sequence[sequence.len() - 1].end;

            // Get the level inside these level runs.
            let level = levels[start];

            // Get the level of the last non-removed char before the runs.
            let pred_level = match initial_classes[..start].iter().rposition(not_removed_by_x9) {
                Some(idx) => levels[idx],
                None => para_level
            };

            // Get the level of the next non-removed char after the runs.
            let succ_level = if matches!(initial_classes[end - 1], RLI|LRI|FSI) {
                para_level
            } else {
                match initial_classes[end..].iter().position(not_removed_by_x9) {
                    Some(idx) => levels[idx],
                    None => para_level
                }
            };

            IsolatingRunSequence {
                runs: sequence,
                sos: class_for_level(max(level, pred_level)),
                eos: class_for_level(max(level, succ_level)),
            }
        }).collect()
    }

    /// Finds the level runs in a paragraph.
    ///
    /// `levels[i]` and `classes[i]` must contain the explicit embedding level and the BidiClass,
    /// respectively, of the char at byte index `i`.
    ///
    /// http://www.unicode.org/reports/tr9/#BD7
    fn level_runs(levels: &[u8], classes: &[BidiClass]) -> Vec<LevelRun> {
        assert!(levels.len() == classes.len());

        let mut runs = Vec::new();
        if levels.len() == 0 {
            return runs
        }

        let mut current_run_level = levels[0];
        let mut current_run_start = 0;

        for i in 1..levels.len() {
            if !removed_by_x9(classes[i]) {
                if levels[i] != current_run_level {
                    // End the last run and start a new one.
                    runs.push(current_run_start..i);
                    current_run_level = levels[i];
                    current_run_start = i;
                }
            }
        }
        runs.push(current_run_start..levels.len());
        runs
    }

    /// Should this character be ignored in steps after X9?
    ///
    /// http://www.unicode.org/reports/tr9/#X9
    pub fn removed_by_x9(class: BidiClass) -> bool {
        matches!(class, RLE | LRE | RLO | LRO | PDF | BN)
    }

    // For use as a predicate for `position` / `rposition`
    fn not_removed_by_x9(class: &BidiClass) -> bool {
        !removed_by_x9(*class)
    }

    #[cfg(test)] #[test]
    fn test_level_runs() {
        use super::prepare::level_runs;
        assert_eq!(level_runs(&[0,0,0,1,1,2,0,0], &[L; 8]), &[0..3, 3..5, 5..6, 6..8]);
    }
}

#[cfg(test)]
mod test {
    use super::{bidi_class, classes, paragraph_level};
    use super::BidiClass::*;

    #[test]
    fn test_classes() {
        assert_eq!(classes(""), &[]);
        assert_eq!(classes("a1"), &[L, EN]);

        // multi-byte characters
        let s = "\u{05D1} \u{0627}";
        assert_eq!(classes(s), &[R, R, WS, AL, AL]);
        assert_eq!(classes(s).len(), s.len());
    }

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
