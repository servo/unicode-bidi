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

use std::borrow::Cow;
use std::cmp::{max, min};
use std::iter::repeat;
use std::ops::Range;

/// Output of `process_paragraph`
#[derive(Debug, PartialEq)]
pub struct ParagraphInfo {
    pub classes: Vec<BidiClass>,
    pub levels: Vec<u8>,
    pub para_level: u8,
    pub max_level: u8,
}

/// Determine the bidirectional embedding levels for a single paragraph.
///
/// TODO: In early steps, check for special cases that allow later steps to be skipped. like text
/// that is entirely LTR.  See the `nsBidi` class from Gecko for comparison.
pub fn process_paragraph(text: &str, mut para_level: u8) -> ParagraphInfo {
    let initial_classes = classes(text);
    if para_level == IMPLICIT_LEVEL {
        para_level = paragraph_level(&initial_classes);
    }
    assert!(para_level <= 1);

    let explicit::Result { mut classes, mut levels } =
        explicit::compute(text, para_level, &initial_classes);

    let sequences = prepare::isolating_run_sequences(para_level, &initial_classes, &levels);
    for sequence in &sequences {
        implicit::resolve_weak(sequence, &mut classes);
        implicit::resolve_neutral(sequence, &levels, &mut classes);
    }
    let max_level = implicit::resolve_levels(&classes, &mut levels);

    ParagraphInfo {
        levels: levels,
        classes: initial_classes,
        para_level: para_level,
        max_level: max_level,
    }
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

pub fn reorder_line<'a>(paragraph: &'a str, line: Range<usize>, info: &ParagraphInfo) -> Cow<'a, str> {
    let runs = visual_runs(line.clone(), info);
    if runs.len() == 1 && !is_rtl(info.levels[runs[0].start]) {
        return paragraph.into()
    }
    let mut result = String::with_capacity(line.len());
    for run in runs {
        if is_rtl(info.levels[run.start]) {
            result.extend(paragraph[run].chars().rev());
        } else {
            result.push_str(&paragraph[run]);
        }
    }
    result.into()
}

/// A maximal substring of characters with the same embedding level.
///
/// Represented as a range of byte indices within a paragraph.
pub type LevelRun = Range<usize>;

/// Find the level runs within a line and return them in visual order.
///
/// `line` is a range of bytes indices with in a paragraph.
pub fn visual_runs(line: Range<usize>, info: &ParagraphInfo) -> Vec<LevelRun> {
    assert!(line.start <= info.levels.len());
    assert!(line.end <= info.levels.len());

    // TODO: Whitespace handling.
    // http://www.unicode.org/reports/tr9/#L1

    assert!(info.max_level >= info.para_level);
    let mut runs = Vec::with_capacity((info.max_level - info.para_level) as usize + 1);

    // Optimization: If there's only one level, just return a single run for the whole line.
    if info.max_level == info.para_level || line.len() == 0 {
        runs.push(line.clone());
        return runs
    }

    // Find consecutive level runs.
    let mut start = line.start;
    let mut level = info.levels[start];
    let mut min_level = level;
    let mut max_level = level;

    for i in (start + 1)..line.end {
        let new_level = info.levels[i];
        if new_level != level {
            // End of the previous run, start of a new one.
            runs.push(start..i);
            start = i;
            level = new_level;

            min_level = min(level, min_level);
            max_level = max(level, max_level);
        }
    }
    runs.push(start..line.end);

    let run_count = runs.len();

    // Re-order the odd runs.
    // http://www.unicode.org/reports/tr9/#L2

    // Stop at the lowest *odd* level.
    min_level |= 1;

    while max_level >= min_level {
        // Look for the start of a sequence of consecutive runs of max_level or higher.
        let mut seq_start = 0;
        while seq_start < run_count {
            if info.levels[runs[seq_start].start] < max_level {
                seq_start += 1;
            }
            if seq_start >= run_count {
                break // No more runs found at this level.
            }

            // Found the start of a sequence. Now find the end.
            let mut seq_end = seq_start + 1;
            while seq_end < run_count {
                if info.levels[runs[seq_end].start] < max_level {
                    break
                }
                seq_end += 1;
            }

            // Reverse the runs within this sequence.
            runs[seq_start..seq_end].reverse();

            seq_start = seq_end;
        }
        max_level -= 1;
    }

    runs
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
    use super::{BidiClass, class_for_level, LevelRun};
    use super::BidiClass::*;
    use std::cmp::max;

    /// Output of `isolating_run_sequences` (steps X9-X10)
    pub struct IsolatingRunSequence {
        pub runs: Vec<LevelRun>,
        pub sos: BidiClass, // Start-of-sequence type.
        pub eos: BidiClass, // End-of-sequence type.
    }

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

/// 3.3.4 - 3.3.6. Resolve implicit levels and types.
mod implicit {
    use super::{BidiClass, class_for_level, is_rtl};
    use super::BidiClass::*;
    use super::prepare::IsolatingRunSequence;
    use std::cmp::max;

    /// 3.3.4 Resolving Weak Types
    ///
    /// http://www.unicode.org/reports/tr9/#Resolving_Weak_Types
    pub fn resolve_weak(sequence: &IsolatingRunSequence, classes: &mut [BidiClass]) {
        let mut prev_class = sequence.sos;
        let mut last_strong_is_al = false;
        let mut last_strong_is_l = false;
        let mut et_run_indices = Vec::new(); // for W5

        let mut indices = sequence.runs.iter().flat_map(Clone::clone).peekable();
        while let Some(i) = indices.next() {
            match classes[i] {
                // http://www.unicode.org/reports/tr9/#W1
                NSM => {
                    classes[i] = match prev_class {
                        RLI | LRI | FSI | PDI => ON,
                        _ => prev_class
                    };
                }
                EN => {
                    if last_strong_is_al {
                        // W2. If previous strong char was AL, change EN to AL.
                        classes[i] = AN;
                    } else {
                        // W5. If a run of ETs is adjacent to an EN, change the ETs to EN.
                        // W7. If the previous strong char was L, change all the ENs to L.
                        if last_strong_is_l {
                            classes[i] = L;
                        }
                        for j in &et_run_indices {
                            classes[*j] = classes[i];
                        }
                        et_run_indices.clear();
                    }
                }
                // http://www.unicode.org/reports/tr9/#W3
                AL => classes[i] = R,

                // http://www.unicode.org/reports/tr9/#W4
                ES | CS => {
                    let next_class = indices.peek().map(|j| classes[*j]);
                    classes[i] = match (prev_class, classes[i], next_class) {
                        (EN, ES, Some(EN)) |
                        (EN, CS, Some(EN)) => EN,
                        (AN, CS, Some(AN)) => AN,
                        (_,  _,  _       ) => ON,
                    }
                }
                // http://www.unicode.org/reports/tr9/#W5
                ET => {
                    match prev_class {
                        EN => classes[i] = EN,
                        _ => et_run_indices.push(i) // In case this is followed by an EN.
                    }
                }
                _ => {}
            }

            prev_class = classes[i];
            match prev_class {
                L =>  { last_strong_is_al = false; last_strong_is_l = true;  }
                R =>  { last_strong_is_al = false; last_strong_is_l = false; }
                AL => { last_strong_is_al = true;  last_strong_is_l = false; }
                _ => {}
            }
            if prev_class != ET {
                // W6. If we didn't find an adjacent EN, turn any ETs into ON instead.
                for j in &et_run_indices {
                    classes[*j] = ON;
                }
                et_run_indices.clear();
            }
        }
    }

    /// 3.3.5 Resolving Neutral Types
    ///
    /// http://www.unicode.org/reports/tr9/#Resolving_Neutral_Types
    pub fn resolve_neutral(sequence: &IsolatingRunSequence, levels: &[u8],
                           classes: &mut [BidiClass])
    {
        let mut indices = sequence.runs.iter().flat_map(Clone::clone).peekable();
        let mut prev_class = sequence.sos;

        // http://www.unicode.org/reports/tr9/#NI
        fn ni(class: BidiClass) -> bool {
            matches!(class, B | S | WS | ON | FSI | LRI | RLI | PDI)
        }

        while let Some(i) = indices.next() {
            // N0. Process bracket pairs.
            // TODO

            // Process sequences of NI characters.
            let mut ni_run = Vec::new();
            if ni(classes[i]) {
                // Consume a run of consecutive NI characters.
                let mut next_class;
                loop {
                    ni_run.push(i);
                    next_class = match indices.peek() {
                        Some(&j) => classes[j],
                        None => sequence.eos
                    };
                    if !ni(next_class) {
                        break
                    }
                    indices.next();
                }

                // N1-N2.
                let new_class = match (prev_class, next_class) {
                    (L,  L ) => L,
                    (R,  R ) |
                    (R,  AN) |
                    (R,  EN) |
                    (AN, R ) |
                    (AN, AN) |
                    (AN, EN) |
                    (EN, R ) |
                    (EN, AN) |
                    (EN, EN) => R,
                    (_,  _ ) => class_for_level(levels[i]),
                };
                for j in &ni_run {
                    classes[*j] = new_class;
                }
                ni_run.clear();
            }
            prev_class = classes[i];
        }
    }

    /// 3.3.6 Resolving Implicit Levels
    ///
    /// Returns the minimum and maximum level in the paragraph.
    ///
    /// http://www.unicode.org/reports/tr9/#Resolving_Implicit_Levels
    pub fn resolve_levels(classes: &[BidiClass], levels: &mut [u8]) -> u8 {
        let mut max_level = 0;

        assert!(classes.len() == levels.len());
        for i in 0..levels.len() {
            match (is_rtl(levels[i]), classes[i]) {
                // http://www.unicode.org/reports/tr9/#I1
                (false, R)  => levels[i] += 1,
                (false, AN) |
                (false, EN) => levels[i] += 2,
                // http://www.unicode.org/reports/tr9/#I2
                (true, L)  |
                (true, EN) |
                (true, AN) => levels[i] += 1,
                (_, _) => {}
            }
            max_level = max(max_level, levels[i]);
        }
        max_level
    }
}

#[cfg(test)]
mod test {
    use super::BidiClass::*;

    #[test]
    fn test_classes() {
        use super::classes;

        assert_eq!(classes(""), &[]);
        assert_eq!(classes("a1"), &[L, EN]);

        // multi-byte characters
        let s = "\u{05D1} \u{0627}";
        assert_eq!(classes(s), &[R, R, WS, AL, AL]);
        assert_eq!(classes(s).len(), s.len());
    }

    #[test]
    fn test_paragraph_level() {
        use super::paragraph_level;

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
        use super::bidi_class;

        assert_eq!(bidi_class('c'), L);
        assert_eq!(bidi_class('\u{05D1}'), R);
        assert_eq!(bidi_class('\u{0627}'), AL);
    }

    #[test]
    fn test_paragraph_info() {
        use super::{IMPLICIT_LEVEL, ParagraphInfo, process_paragraph};

        assert_eq!(process_paragraph("abc123", 0), ParagraphInfo {
            levels:  vec![0, 0, 0, 0,  0,  0],
            classes: vec![L, L, L, EN, EN, EN],
            para_level: 0,
            max_level: 0,
        });
        assert_eq!(process_paragraph("abc אבג", 0), ParagraphInfo {
            levels:  vec![0, 0, 0, 0,  1,1, 1,1, 1,1],
            classes: vec![L, L, L, WS, R,R, R,R, R,R],
            para_level: 0,
            max_level: 1,
        });
        assert_eq!(process_paragraph("abc אבג", 1), ParagraphInfo {
            levels:  vec![2, 2, 2, 1,  1,1, 1,1, 1,1],
            classes: vec![L, L, L, WS, R,R, R,R, R,R],
            para_level: 1,
            max_level: 2,
        });
        assert_eq!(process_paragraph("אבג abc", 0), ParagraphInfo {
            levels:  vec![1,1, 1,1, 1,1, 0,  0, 0, 0],
            classes: vec![R,R, R,R, R,R, WS, L, L, L],
            para_level: 0,
            max_level: 1,
        });
        assert_eq!(process_paragraph("אבג abc", IMPLICIT_LEVEL), ParagraphInfo {
            levels:  vec![1,1, 1,1, 1,1, 1,  2, 2, 2],
            classes: vec![R,R, R,R, R,R, WS, L, L, L],
            para_level: 1,
            max_level: 2,
        });
        assert_eq!(process_paragraph("غ2ظ א2ג", 0), ParagraphInfo {
            levels:  vec![1, 1,  2,  1, 1,  1,  1,1, 2,  1,1],
            classes: vec![AL,AL, EN, AL,AL, WS, R,R, EN, R,R],
            para_level: 0,
            max_level: 2,
        });
    }

    #[test]
    fn test_reorder_line() {
        use super::{IMPLICIT_LEVEL, process_paragraph, reorder_line};
        use std::borrow::Cow;

        fn reorder(s: &str) -> Cow<str> {
            reorder_line(s, 0..s.len(), &process_paragraph(s, IMPLICIT_LEVEL))
        }

        assert_eq!(reorder("abc123"), "abc123");
        assert_eq!(reorder("abc אבג"), "abc גבא");
        assert_eq!(reorder("אבג abc"), "abc גבא");
    }
}
