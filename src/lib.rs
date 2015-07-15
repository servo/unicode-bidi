// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This crate implements the [Unicode Bidirectional Algorithm][tr9] for display of mixed
//! right-to-left and left-to-right text.  It is written in safe Rust, compatible with Rust 1.0 and
//! later.
//!
//! ## Example
//!
//! ```rust
//! use unicode_bidi::{process_paragraph, reorder_line};
//!
//! // This example text is defined using `concat!` because some browsers
//! // and text editors have trouble displaying bidi strings.
//! let paragraph = concat!["א",
//!                         "ב",
//!                         "ג",
//!                         "a",
//!                         "b",
//!                         "c"];
//!
//! // Resolve embedding levels within a paragraph.  Pass `None` to detect the
//! // paragraph level automatically.
//! let info = process_paragraph(&paragraph, None);
//!
//! // This paragraph has embedding level 1 because its first strong character is RTL.
//! assert_eq!(info.para_level, 1);
//!
//! // Re-ordering is done after wrapping the paragraph into a sequence of
//! // lines. For this example, I'll just use a single line that spans the
//! // entire paragraph.
//! let line = 0..paragraph.len();
//!
//! let display = reorder_line(&paragraph, line, &info);
//! assert_eq!(display, concat!["a",
//!                             "b",
//!                             "c",
//!                             "ג",
//!                             "ב",
//!                             "א"]);
//! ```
//!
//! [tr9]: http://www.unicode.org/reports/tr9/

#![forbid(unsafe_code)]

#[macro_use] extern crate matches;
#[macro_use] extern crate bitflags;

pub mod tables;

pub use tables::{BidiClass, bidi_class, UNICODE_VERSION};
use BidiClass::*;

use std::borrow::Cow;
use std::cmp::{max, min};
use std::iter::repeat;
use std::ops::Range;

/// Output of `process_paragraph`
///
/// The `classes` and `levels` vectors are indexed by byte offsets into the paragraph text.  If a
/// character is multiple bytes wide, then its class and level will appear multiple times in these
/// vectors.
#[derive(Debug, PartialEq)]
pub struct ParagraphInfo {
    /// The BidiClass of the character at each byte in the paragraph.
    pub classes: Vec<BidiClass>,

    /// The directional embedding level of each byte in the paragraph.
    pub levels: Vec<u8>,

    /// The paragraph embedding level.
    ///
    /// http://www.unicode.org/reports/tr9/#BD4
    pub para_level: u8,

    /// The highest embedding level in the paragraph. (Can be used for optimizations.)
    pub max_level: u8,
}

/// Determine the bidirectional embedding levels for a single paragraph.
///
/// `level` should contain the paragraph embedding level if it is set
/// explicitly, or `None` otherwise.
pub fn process_paragraph(text: &str, level: Option<u8>) -> ParagraphInfo {
    let InitialProperties { para_level, initial_classes, flags } = initial_scan(text, level);

    let explicit::Result { mut classes, mut levels } =
        explicit::compute(text, para_level, &initial_classes, flags);

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

#[inline]
/// Even embedding levels are left-to-right.
///
/// http://www.unicode.org/reports/tr9/#BD2
pub fn is_ltr(level: u8) -> bool { level % 2 == 0 }

/// Odd levels are right-to-left.
///
/// http://www.unicode.org/reports/tr9/#BD2
pub fn is_rtl(level: u8) -> bool { level % 2 == 1 }

/// Generate a character type based on a level (as specified in steps X10 and N2).
fn class_for_level(level: u8) -> BidiClass {
    if is_rtl(level) { R } else { L }
}

/// Re-order a line based on resolved levels.
///
/// `info` is the result of calling `process_paragraph` on `paragraph`.
/// `line` is a range of bytes indices within `paragraph`.
///
/// Returns the line in display order.
pub fn reorder_line<'a>(paragraph: &'a str, line: Range<usize>, info: &ParagraphInfo)
    -> Cow<'a, str>
{
    let runs = visual_runs(line.clone(), info.para_level, info.max_level, &info.levels);
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
/// `line` is a range of bytes indices within `paragraph`.
///
/// http://www.unicode.org/reports/tr9/#Reordering_Resolved_Levels
pub fn visual_runs(line: Range<usize>,
                   para_level: u8,
                   max_level: u8,
                   levels: &[u8]) -> Vec<LevelRun> {
    assert!(line.start <= levels.len());
    assert!(line.end <= levels.len());

    // TODO: Whitespace handling.
    // http://www.unicode.org/reports/tr9/#L1

    assert!(max_level >= para_level);
    let mut runs = Vec::with_capacity((max_level - para_level) as usize + 1);

    // Optimization: If there's only one level, just return a single run for the whole line.
    if max_level == para_level || line.len() == 0 {
        runs.push(line.clone());
        return runs
    }

    // Find consecutive level runs.
    let mut start = line.start;
    let mut level = levels[start];
    let mut min_level = level;
    let mut max_level = level;

    for i in (start + 1)..line.end {
        let new_level = levels[i];
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
            if levels[runs[seq_start].start] < max_level {
                seq_start += 1;
            }
            if seq_start >= run_count {
                break // No more runs found at this level.
            }

            // Found the start of a sequence. Now find the end.
            let mut seq_end = seq_start + 1;
            while seq_end < run_count {
                if levels[runs[seq_end].start] < max_level {
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

/// Output of `initial_scan`
#[derive(PartialEq, Debug)]
pub struct InitialProperties {
    /// The BidiClass of the character at each byte in the paragraph.
    /// If a character is multiple bytes, its class will appear multiple times in the vector.
    pub initial_classes: Vec<BidiClass>,

    /// The paragraph embedding level.
    pub para_level: u8,

    /// The paragraph direction; used for optimizations in later stages.
    pub flags: Flags,
}

/// Find the paragraph embedding level, and the BidiClass for each character.
///
/// http://www.unicode.org/reports/tr9/#The_Paragraph_Level
///
/// Also sets the class for each First Strong Isolate initiator (FSI) to LRI or RLI if a strong
/// character is found before the matching PDI.  If no strong character is found, the class will
/// remain FSI, and it's up to later stages to treat these as LRI when needed.
pub fn initial_scan(paragraph: &str, mut para_level: Option<u8>) -> InitialProperties {
    let mut classes = Vec::with_capacity(paragraph.len());

    let mut flags = Flags::empty();

    // The stack contains the starting byte index for each nested isolate we're inside.
    let mut isolate_stack = Vec::new();

    const FSI_CHAR: char = '\u{2069}';

    for (i, c) in paragraph.char_indices() {
        let class = bidi_class(c);
        classes.extend(repeat(class).take(c.len_utf8()));
        match class {
            L | R | AL => match isolate_stack.last() {
                Some(&start) => if classes[start] == FSI {
                    // X5c. If the first strong character between FSI and its matching PDI is R
                    // or AL, treat it as RLI. Otherwise, treat it as LRI.
                    for j in 0..FSI_CHAR.len_utf8() {
                        classes[start+j] = if class == L { LRI } else { RLI };
                    }
                },
                None => if para_level.is_none() {
                    // P2. Find the first character of type L, AL, or R, while skipping any
                    // characters between an isolate initiator and its matching PDI.
                    para_level = Some(if class == L { 0 } else { 1 });
                }
            },
            RLI | LRI | FSI => {
                isolate_stack.push(i);
            }
            PDI => {
                isolate_stack.pop();
            }
            _ => {}
        }
        flags.insert(Flags::from_class(class));
    }
    assert!(classes.len() == paragraph.len());

    InitialProperties {
        // P3. If no character is found in p2, set the paragraph level to zero.
        para_level: para_level.unwrap_or(0),
        initial_classes: classes,
        flags: flags,
    }
}

/// Extra information about a paragraph, used to optimize later steps.
bitflags! {
    flags Flags: u8 {
        const HAS_LTR              = 0x0001,
        const HAS_RTL              = 0x0002,
        const HAS_AN               = 0x0004,
        const HAS_POSSIBLE_NEUTRAL = 0x0008,
        const HAS_EXPLICIT         = 0x0010,
    }
}

impl Flags {
    #[inline]
    fn from_class(class: BidiClass) -> Flags {
        match class {
            L | EN =>
                HAS_LTR,
            LRE | LRO | LRI =>
                HAS_LTR | HAS_POSSIBLE_NEUTRAL | HAS_EXPLICIT,
            R | AL =>
                HAS_RTL,
            RLE | RLO | RLI =>
                HAS_RTL | HAS_POSSIBLE_NEUTRAL | HAS_EXPLICIT,
            AN =>
                HAS_AN | HAS_LTR,
            ON | CS | ES | ET | B | S | WS | BN | FSI | PDI | PDF =>
                HAS_POSSIBLE_NEUTRAL,
            NSM => Flags::empty()
        }
    }

    fn to_direction(self) -> Direction {
        // If the text has both AN and neutrals, then some neutrals may become RTL.
        let has_rtl = self.contains(HAS_RTL) || self.contains(HAS_AN | HAS_POSSIBLE_NEUTRAL);
        let has_ltr = self.contains(HAS_LTR);

        match (has_ltr, has_rtl) {
            (true, true) => Direction::Mixed,
            (false, true) => Direction::RTL,
            (_, false) => Direction::LTR
        }
    }
}


/// The direction of a paragraph based on the classes of its characters.
#[derive(PartialEq, Debug)]
pub enum Direction {
    /// All characters are RTL.
    RTL,
    /// All characters are LTR.
    LTR,
    /// The paragraph may contain both LTR and RTL characters.
    Mixed
}

/// 3.3.2 Explicit Levels and Directions
///
/// http://www.unicode.org/reports/tr9/#Explicit_Levels_and_Directions
mod explicit {
    use super::{BidiClass, Direction, Flags, HAS_EXPLICIT, is_rtl};
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
    pub fn compute(text: &str, para_level: u8, classes: &[BidiClass], flags: Flags)
        -> Result
    {
        assert!(text.len() == classes.len());

        let direction = flags.to_direction();
        let default_level = match direction {
            Direction::RTL => para_level | 1, // make sure level is odd.
            Direction::LTR => para_level + 1 & !1, // make sure level is even.
            Direction::Mixed => para_level
        };

        let mut result = Result {
            levels: vec![default_level; text.len()],
            classes: Vec::from(classes),
        };

        if direction != Direction::Mixed || !flags.contains(HAS_EXPLICIT) {
            // Optimization: Skip the rest of this step and just return initial
            // levels and classes.
            return result
        }

        // TODO: Recompute Flags based on the classes and levels computed below,
        // for use in optimization of later steps.  Add a flag for whether the
        // levels vec contains multiple runs.

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
    ///
    /// Note: This function does *not* return the sequences in order by their first characters.
    pub fn isolating_run_sequences(para_level: u8, initial_classes: &[BidiClass], levels: &[u8])
        -> Vec<IsolatingRunSequence>
    {
        // TODO: If earlier steps detected that the text is unidirectional (has only one level),
        // just return a single run of the entire paragraph.

        let runs = level_runs(levels, initial_classes);

        // Compute the set of isolating run sequences.
        // http://www.unicode.org/reports/tr9/#BD13

        let mut sequences = Vec::with_capacity(runs.len());

        // When we encounter an isolate initiator, we push the current sequence onto the
        // stack so we can resume it after the matching PDI.
        let mut stack = vec![Vec::new()];

        for run in runs {
            assert!(run.len() > 0);
            assert!(stack.len() > 0);

            let start_class = initial_classes[run.start];
            let end_class = initial_classes[run.end - 1];

            let mut sequence = if start_class == PDI && stack.len() > 1 {
                // Continue a previous sequence interrupted by an isolate.
                stack.pop().unwrap()
            } else {
                // Start a new sequence.
                Vec::new()
            };

            sequence.push(run);

            if matches!(end_class, RLI | LRI | FSI) {
                // Resume this sequence after the isolate.
                stack.push(sequence);
            } else {
                // This sequence is finished.
                sequences.push(sequence);
            }
        }
        // Pop any remaning sequences off the stack.
        sequences.extend(stack.into_iter().rev().filter(|seq| seq.len() > 0));

        // Determine the `sos` and `eos` class for each sequence.
        // http://www.unicode.org/reports/tr9/#X10
        return sequences.into_iter().map(|sequence| {
            assert!(!sequence.len() > 0);
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
        assert_eq!(level_runs(&[0,0,0,1,1,2,0,0], &[L; 8]), &[0..3, 3..5, 5..6, 6..8]);
    }

    #[cfg(test)] #[test]
    fn test_isolating_run_sequences() {
        // Example 3 from http://www.unicode.org/reports/tr9/#BD13:

        //              0  1    2   3    4  5  6  7    8   9   10
        let classes = &[L, RLI, AL, LRI, L, R, L, PDI, AL, PDI, L];
        let levels =  &[0, 0,   1,  1,   2, 3, 2, 1,   1,  0,   0];
        let para_level = 0;

        let sequences = isolating_run_sequences(para_level, classes, levels);
        let runs: Vec<Vec<LevelRun>> = sequences.iter().map(|s| s.runs.clone()).collect();
        assert_eq!(runs, vec![vec![4..5], vec![5..6], vec![6..7], vec![2..4, 7..9], vec![0..2, 9..11]]);
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
        let mut indices = sequence.runs.iter().flat_map(Clone::clone);
        let mut prev_class = sequence.sos;

        // Neutral or Isolate formatting characters (NI).
        // http://www.unicode.org/reports/tr9/#NI
        fn ni(class: BidiClass) -> bool {
            matches!(class, B | S | WS | ON | FSI | LRI | RLI | PDI)
        }

        while let Some(mut i) = indices.next() {
            // N0. Process bracket pairs.
            // TODO

            // Process sequences of NI characters.
            let mut ni_run = Vec::new();
            if ni(classes[i]) {
                // Consume a run of consecutive NI characters.
                ni_run.push(i);
                let mut next_class;
                loop {
                    match indices.next() {
                        Some(j) => {
                            i = j;
                            if ::prepare::removed_by_x9(classes[i]) {
                                continue
                            }
                            next_class = classes[j];
                            if ni(next_class) {
                                ni_run.push(i);
                            } else {
                                break
                            }
                        }
                        None => {
                            next_class = sequence.eos;
                            break
                        }
                    };
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
    /// Returns the maximum embedding level in the paragraph.
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
    fn test_initial_scan() {
        use super::{Direction, initial_scan};

        let props = initial_scan("a1", None);
        assert_eq!(props.para_level, 0);
        assert_eq!(props.initial_classes, &[L, EN]);
        assert_eq!(props.flags.to_direction(), Direction::LTR);

        let props = initial_scan("غ א", None);
        assert_eq!(props.para_level, 1);
        assert_eq!(props.initial_classes, &[AL, AL, WS, R, R]);
        assert_eq!(props.flags.to_direction(), Direction::RTL);

        let fsi = '\u{2068}';
        let pdi = '\u{2069}';

        let s = format!("{}א{}a", fsi, pdi);
        let props = initial_scan(&s, None);
        assert_eq!(props.para_level, 0);
        assert_eq!(props.initial_classes, &[RLI, RLI, RLI, R, R, PDI, PDI, PDI, L]);
        assert_eq!(props.flags.to_direction(), Direction::Mixed);
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
        use super::{ParagraphInfo, process_paragraph};

        assert_eq!(process_paragraph("abc123", Some(0)), ParagraphInfo {
            levels:  vec![0, 0, 0, 0,  0,  0],
            classes: vec![L, L, L, EN, EN, EN],
            para_level: 0,
            max_level: 0,
        });
        assert_eq!(process_paragraph("abc אבג", Some(0)), ParagraphInfo {
            levels:  vec![0, 0, 0, 0,  1,1, 1,1, 1,1],
            classes: vec![L, L, L, WS, R,R, R,R, R,R],
            para_level: 0,
            max_level: 1,
        });
        assert_eq!(process_paragraph("abc אבג", Some(1)), ParagraphInfo {
            levels:  vec![2, 2, 2, 1,  1,1, 1,1, 1,1],
            classes: vec![L, L, L, WS, R,R, R,R, R,R],
            para_level: 1,
            max_level: 2,
        });
        assert_eq!(process_paragraph("אבג abc", Some(0)), ParagraphInfo {
            levels:  vec![1,1, 1,1, 1,1, 0,  0, 0, 0],
            classes: vec![R,R, R,R, R,R, WS, L, L, L],
            para_level: 0,
            max_level: 1,
        });
        assert_eq!(process_paragraph("אבג abc", None), ParagraphInfo {
            levels:  vec![1,1, 1,1, 1,1, 1,  2, 2, 2],
            classes: vec![R,R, R,R, R,R, WS, L, L, L],
            para_level: 1,
            max_level: 2,
        });
        assert_eq!(process_paragraph("غ2ظ א2ג", Some(0)), ParagraphInfo {
            levels:  vec![1, 1,  2,  1, 1,  1,  1,1, 2,  1,1],
            classes: vec![AL,AL, EN, AL,AL, WS, R,R, EN, R,R],
            para_level: 0,
            max_level: 2,
        });
    }

    #[test]
    fn test_reorder_line() {
        use super::{process_paragraph, reorder_line};
        use std::borrow::Cow;

        fn reorder(s: &str) -> Cow<str> {
            reorder_line(s, 0..s.len(), &process_paragraph(s, None))
        }

        assert_eq!(reorder("abc123"), "abc123");
        assert_eq!(reorder("אבג"), "גבא");
        assert_eq!(reorder("abc אבג"), "abc גבא");
        assert_eq!(reorder("אבג abc"), "abc גבא");
        assert_eq!(reorder("abc\u{2067}.-\u{2069}ghi"),
                           "abc\u{2067}-.\u{2069}ghi");
        assert_eq!(reorder("Hello, \u{2068}\u{202E}world\u{202C}\u{2069}!"),
                           "Hello, \u{2068}dlrow\u{202E}\u{202C}\u{2069}!");
    }
}
