// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This crate implements the [Unicode Bidirectional Algorithm][tr9] for display of mixed
//! right-to-left and left-to-right text.  It is written in safe Rust, compatible with the
//! current stable release.
//!
//! ## Example
//!
//! ```rust
//! use unicode_bidi::{process_text, reorder_line};
//!
//! // This example text is defined using `concat!` because some browsers
//! // and text editors have trouble displaying bidi strings.
//! let text = concat!["א",
//!                    "ב",
//!                    "ג",
//!                    "a",
//!                    "b",
//!                    "c"];
//!
//! // Resolve embedding levels within the text.  Pass `None` to detect the
//! // paragraph level automatically.
//! let info = process_text(&text, None);
//!
//! // This paragraph has embedding level 1 because its first strong character is RTL.
//! assert_eq!(info.paragraphs.len(), 1);
//! let paragraph_info = &info.paragraphs[0];
//! assert_eq!(paragraph_info.level, 1);
//!
//! // Re-ordering is done after wrapping each paragraph into a sequence of
//! // lines. For this example, I'll just use a single line that spans the
//! // entire paragraph.
//! let line = paragraph_info.range.clone();
//!
//! let display = reorder_line(&text, line, &info.levels);
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

pub mod tables;
pub mod brackets;


pub use tables::{BidiClass, bidi_class, UNICODE_VERSION};
use BidiClass::*;

use std::borrow::Cow;
use std::cmp::{max, min};
use std::iter::repeat;
use std::ops::Range;

/// Output of `process_text`
///
/// The `classes` and `levels` vectors are indexed by byte offsets into the text.  If a character
/// is multiple bytes wide, then its class and level will appear multiple times in these vectors.
#[derive(Debug, PartialEq)]
pub struct BidiInfo {
    /// The BidiClass of the character at each byte in the text.
    pub classes: Vec<BidiClass>,

    /// The directional embedding level of each byte in the text.
    pub levels: Vec<u8>,

    /// The boundaries and paragraph embedding level of each paragraph within the text.
    ///
    /// TODO: Use SmallVec or similar to avoid overhead when there are only one or two paragraphs?
    /// Or just don't include the first paragraph, which always starts at 0?
    pub paragraphs: Vec<ParagraphInfo>,
}

/// Info about a single paragraph 
#[derive(Debug, PartialEq)]
pub struct ParagraphInfo {
    /// The paragraphs boundaries within the text, as byte indices.
    ///
    /// TODO: Shrink this to only include the starting index?
    pub range: Range<usize>,

    /// The paragraph embedding level. http://www.unicode.org/reports/tr9/#BD4
    pub level: u8,
}

/// Determine the bidirectional embedding levels for a single paragraph.
///
/// TODO: In early steps, check for special cases that allow later steps to be skipped. like text
/// that is entirely LTR.  See the `nsBidi` class from Gecko for comparison.
pub fn process_text(text: &str, level: Option<u8>) -> BidiInfo {
    let InitialProperties { initial_classes, paragraphs } = initial_scan(text, level);

    let mut levels = Vec::with_capacity(text.len());
    let mut classes = initial_classes.clone();

    for para in &paragraphs {
        let text = &text[para.range.clone()];
        let classes = &mut classes[para.range.clone()];
        let initial_classes = &initial_classes[para.range.clone()];

        // FIXME: Use `levels.resize(...)` when it becomes stable.
        levels.extend(repeat(para.level).take(para.range.len()));
        let levels = &mut levels[para.range.clone()];

        explicit::compute(text, para.level, &initial_classes, levels, classes);

        let sequences = prepare::isolating_run_sequences(para.level, &initial_classes, levels);
        for sequence in &sequences {
            implicit::resolve_weak(sequence, classes);

            //Step N0: 
            bracket_pair_resolver::resolve_n0(&text, &sequence.sos, classes, &levels[0]);

            implicit::resolve_neutral(sequence, levels, classes);
        }
        implicit::resolve_levels(classes, levels);
        resolve_white_space(para.level, &initial_classes, levels);
        //assign_levels_to_removed_chars(para.level, &initial_classes, levels);
    }

    BidiInfo {
        levels: levels,
        classes: initial_classes,
        paragraphs: paragraphs,
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
/// `levels` are the embedding levels returned by `process_text`.
/// `line` is a range of bytes indices within `text`.
///
/// Returns the line in display order.
pub fn reorder_line<'a>(text: &'a str, line: Range<usize>, levels: &[u8])
    -> Cow<'a, str>
{
    let runs = visual_runs(line.clone(), &levels);
    if runs.len() == 1 && !is_rtl(levels[runs[0].start]) {
        return text.into()
    }
    let mut result = String::with_capacity(line.len());
    for run in runs {
        if is_rtl(levels[run.start]) {
            result.extend(text[run].chars().rev());
        } else {
            result.push_str(&text[run]);
        }
    }
    result.into()
}

/// A maximal substring of characters with the same embedding level.
///
/// Represented as a range of byte indices.
pub type LevelRun = Range<usize>;

/// Find the level runs within a line and return them in visual order.
///
/// `line` is a range of bytes indices within `levels`.
///
/// http://www.unicode.org/reports/tr9/#Reordering_Resolved_Levels
pub fn visual_runs(line: Range<usize>, levels: &[u8]) -> Vec<LevelRun> {
    assert!(line.start <= levels.len());
    assert!(line.end <= levels.len());

    // TODO: Whitespace handling.
    // http://www.unicode.org/reports/tr9/#L1

    let mut runs = Vec::new();

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
                continue
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
    /// The BidiClass of the character at each byte in the text.
    /// If a character is multiple bytes, its class will appear multiple times in the vector.
    pub initial_classes: Vec<BidiClass>,

    /// The boundaries and level of each paragraph within the text.
    pub paragraphs: Vec<ParagraphInfo>,
}

/// Find the paragraphs and BidiClasses in a string of text.
///
/// http://www.unicode.org/reports/tr9/#The_Paragraph_Level
///
/// Also sets the class for each First Strong Isolate initiator (FSI) to LRI or RLI if a strong
/// character is found before the matching PDI.  If no strong character is found, the class will
/// remain FSI, and it's up to later stages to treat these as LRI when needed.
pub fn initial_scan(text: &str, default_para_level: Option<u8>) -> InitialProperties {
    let mut classes = Vec::with_capacity(text.len());

    // The stack contains the starting byte index for each nested isolate we're inside.
    let mut isolate_stack = Vec::new();
    let mut paragraphs = Vec::new();

    let mut para_start = 0;
    let mut para_level = default_para_level;

    const FSI_CHAR: char = '\u{2069}';

    for (i, c) in text.char_indices() {
        let class = bidi_class(c);
        classes.extend(repeat(class).take(c.len_utf8()));
        match class {
            B => {
                // P1. Split the text into separate paragraphs. The paragraph separator is kept
                // with the previous paragraph.
                let para_end = i + c.len_utf8();
                paragraphs.push(ParagraphInfo {
                    range: para_start..para_end,
                    // P3. If no character is found in p2, set the paragraph level to zero.
                    level: para_level.unwrap_or(0)
                });
                // Reset state for the start of the next paragraph.
                para_start = para_end;
                para_level = default_para_level;
                isolate_stack.clear();
            }
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
    }
    if para_start < text.len() {
        paragraphs.push(ParagraphInfo {
            range: para_start..text.len(),
            level: para_level.unwrap_or(0)
        });
    }
    assert!(classes.len() == text.len());

    InitialProperties {
        initial_classes: classes,
        paragraphs: paragraphs,
    }
}

fn resolve_white_space(para_level: u8, classes: &[BidiClass], levels: &mut [u8])
{   
    let mut start = 0;
    let line_end = levels.len();
    let mut level = levels[start];

     for i in start..line_end {

        let new_level = levels[i];

        if new_level != level {
            let prev_start = start;
            start = i;
            level = new_level;
            //Rule L1, clause 4
            for k in (prev_start..start).rev()
            {
                if prepare::white_space(classes[k]) {
                   levels[k] = para_level;
                }
                else if prepare::removed_by_x9(classes[k]) {
                            levels[k] = if k > 0 { levels[k-1] } else { para_level };
                            
                        }
                else {
                    break;
                }
            } 
        }
        else {
            //Rule L1, clauses 1 and 2
            if prepare::is_segement_or_paragraph_separator(classes[i]) {
                    levels[i] = para_level;

                    //Rule L1, clause  3
                    for j in (0..i).rev(){

                        if prepare::white_space(classes[j]) {
                           levels[j] = para_level;
                        }
                        else if prepare::removed_by_x9(classes[j]) {
                            levels[j] = if j > 0 { levels[j-1] } else { para_level };
                        }
                        else {
                             break;
                        }
                    }
                    
                } 
        }
    }

}

/// Assign levels to characters removed by rule X9.
///
/// The levels assigned to these characters are not specified by the algorithm.  This function
/// assigns each one the level of the previous character, to avoid breaking level runs.

//fn assign_levels_to_removed_chars(para_level: u8, classes: &[BidiClass], levels: &mut [u8]) {
    //for i in 0..levels.len() {
        //if prepare::removed_by_x9(classes[i]) {
            //levels[i] = if i > 0 { levels[i-1] } else { para_level };
        //}
    //}
//}

/// 3.3.2 Explicit Levels and Directions
///
/// http://www.unicode.org/reports/tr9/#Explicit_Levels_and_Directions
mod explicit {
    use super::{BidiClass, is_rtl};
    use super::BidiClass::*;

    /// Compute explicit embedding levels for one paragraph of text (X1-X8).
    ///
    /// `classes[i]` must contain the BidiClass of the char at byte index `i`,
    /// for each char in `text`.
    pub fn compute(text: &str, para_level: u8, initial_classes: &[BidiClass],
                   levels: &mut [u8], classes: &mut [BidiClass]) {
        assert!(text.len() == initial_classes.len());

        // http://www.unicode.org/reports/tr9/#X1
        let mut stack = DirectionalStatusStack::new();
        stack.push(para_level, OverrideStatus::Neutral);

        let mut overflow_isolate_count = 0u32;
        let mut overflow_embedding_count = 0u32;
        let mut valid_isolate_count = 0u32;

        for (i, c) in text.char_indices() {
            match initial_classes[i] {
                // Rules X2-X5c
                RLE | LRE | RLO | LRO | RLI | LRI | FSI => {
                    let is_rtl = match initial_classes[i] {
                        RLE | RLO | RLI => true,
                        _ => false
                    };

                    let last_level = stack.last().level;
                    let new_level = match is_rtl {
                        true  => next_rtl_level(last_level),
                        false => next_ltr_level(last_level)
                    };

                    // X5a-X5c: Isolate initiators get the level of the last entry on the stack.
                    let is_isolate = matches!(initial_classes[i], RLI | LRI | FSI);
                    if is_isolate {
                        levels[i] = last_level;
                        match stack.last().status {
                            OverrideStatus::RTL => classes[i] = R,
                            OverrideStatus::LTR => classes[i] = L,
                            _ => {}
                        }
                    }

                    if valid(new_level) && overflow_isolate_count == 0 && overflow_embedding_count == 0 {
                        stack.push(new_level, match initial_classes[i] {
                            RLO => OverrideStatus::RTL,
                            LRO => OverrideStatus::LTR,
                            RLI | LRI | FSI => OverrideStatus::Isolate,
                            _ => OverrideStatus::Neutral
                        });
                        if is_isolate {
                            valid_isolate_count += 1;
                        } else {
                            // The spec doesn't explicitly mention this step, but it is necessary.
                            // See the reference implementations for comparison.
                            levels[i] = new_level;
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
                    } else if valid_isolate_count > 0 {
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
                    }
                    let last = stack.last();
                    levels[i] = last.level;
                    match last.status {
                        OverrideStatus::RTL => classes[i] = R,
                        OverrideStatus::LTR => classes[i] = L,
                        _ => {}
                    }
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
                    // The spec doesn't explicitly mention this step, but it is necessary.
                    // See the reference implementations for comparison.
                    levels[i] = stack.last().level;
                }
                // http://www.unicode.org/reports/tr9/#X6
                B | BN => {}
                _ => {
                    let last = stack.last();
                    levels[i] = last.level;
                    match last.status {
                        OverrideStatus::RTL => classes[i] = R,
                        OverrideStatus::LTR => classes[i] = L,
                        _ => {}
                    }
                }
            }
            // Handle multi-byte characters.
            for j in 1..c.len_utf8() {
                levels[i+j] = levels[i];
                classes[i+j] = classes[i];
            }
        }
    }

    /// Maximum depth of the directional status stack.
    pub const MAX_DEPTH: u8 = 125;

    /// Levels from 0 through max_depth are valid at this stage.
    /// http://www.unicode.org/reports/tr9/#X1
    fn valid(level: u8) -> bool { level <= MAX_DEPTH }

    /// The next odd level greater than `level`.
    fn next_rtl_level(level: u8) -> u8 { (level + 1) |  1 }

    /// The next even level greater than `level`.
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

    pub fn is_segement_or_paragraph_separator (class: BidiClass) -> bool {   
    match class {
         B | S => true,
        _ => false,
    }}

    pub fn white_space(class: BidiClass) -> bool
    {    match class {
        WS  => true,
        _ => false,
    }}

    // For use as a predicate for `position` / `rposition`
    pub fn not_removed_by_x9(class: &BidiClass) -> bool {
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
    use super::{BidiClass, class_for_level, is_rtl, LevelRun};
    use super::BidiClass::*;
    use super::prepare::{IsolatingRunSequence, not_removed_by_x9, removed_by_x9};
    use std::cmp::max;

    /// 3.3.4 Resolving Weak Types
    ///
    /// http://www.unicode.org/reports/tr9/#Resolving_Weak_Types
    pub fn resolve_weak(sequence: &IsolatingRunSequence, classes: &mut [BidiClass]) {
        // FIXME (#8): This function applies steps W1-W6 in a single pass.  This can produce
        // incorrect results in cases where a "later" rule changes the value of `prev_class` seen
        // by an "earlier" rule.  We should either split this into separate passes, or preserve
        // extra state so each rule can see the correct previous class.

        let mut prev_class = sequence.sos;
        let mut last_strong_is_al = false;
        let mut et_run_indices = Vec::new(); // for W5
        // Like sequence.runs.iter().flat_map(Clone::clone), but make indices itself clonable.
        fn id(x: LevelRun) -> LevelRun { x }
        let mut indices = sequence.runs.iter().cloned().flat_map(id as fn(LevelRun) -> LevelRun);
    
        // Creating new variables to store the previous state of prev class for each rule
        let prev_class_w1 = prev_class;
        let prev_class_w4 = prev_class;
        let mut prev_class_w5 = prev_class;
        let prev_class_w6 = prev_class;

        while let Some(i) = indices.next() {
            match classes[i] {
                // http://www.unicode.org/reports/tr9/#W1
                NSM => {
                    classes[i] = match prev_class_w1 {
                        RLI | LRI | FSI | PDI => ON,
                        _ => prev_class_w1
                     };
                }
                EN => {
                    if last_strong_is_al {
                        // W2. If previous strong char was AL, change EN to AN.
                        classes[i] = AN;
                    } else {
                        // W5. If a run of ETs is adjacent to an EN, change the ETs to EN.
                        for j in &et_run_indices {
                            classes[*j] = EN;
                        }
                        et_run_indices.clear();
                    }
                }
                // http://www.unicode.org/reports/tr9/#W3
                AL => classes[i] = R,

                // http://www.unicode.org/reports/tr9/#W4
                ES | CS => {
                    let next_class = indices.clone().map(|j| classes[j]).filter(not_removed_by_x9)
                        .next().unwrap_or(sequence.eos);
                    classes[i] = match (prev_class_w4, classes[i], next_class) {
                        (EN, ES, EN) |
                        (EN, CS, EN) => EN,
                        (AN, CS, AN) => AN,
                        (_,  _,  _ ) => ON,
                    }
                }
                // http://www.unicode.org/reports/tr9/#W5
                ET => {
                    match prev_class_w5 {
                       EN => classes[i] = EN,
                        _ => et_run_indices.push(i) // In case this is followed by an EN.
                    }
                }
                class => if removed_by_x9(class) {
                    continue
                }
            }

            prev_class_w5 = classes[i];
            match prev_class_w5 {
              L | R => { last_strong_is_al = false; }
                AL => { last_strong_is_al = true;  }
                _ => {}
            }
            if prev_class_w6 != ET {
             // W6. If we didn't find an adjacent EN, turn any ETs into ON instead.
                for j in &et_run_indices {
                    classes[*j] = ON;
                }
                et_run_indices.clear();
            }
        }

        // W7. If the previous strong char was L, change EN to L.
        let mut last_strong_is_l = sequence.sos == L;
        for run in &sequence.runs {
            for i in run.clone() {
                match classes[i] {
                    EN if last_strong_is_l => { classes[i] = L; }
                    L => { last_strong_is_l = true; }
                    R | AL => { last_strong_is_l = false; }
                    _ => {}
                }
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
                            if removed_by_x9(classes[i]) {
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

mod bracket_pair_resolver{

    pub use std::vec::Vec;
    pub use tables::{BidiClass, bidi_class, UNICODE_VERSION};
    pub use brackets::{BracketType, bracket_type, pair_id};
    pub use tables::BidiClass::*;
    pub use brackets::BracketType::*;

    mod bracket_pair {
        use std::cmp::Ordering;
        #[derive(Debug, Copy, Clone)]
        pub struct BracketPair {
            pub ich_opener: u8,
            pub ich_closer: u8
        }
        impl Ord for BracketPair {
            fn cmp(&self, other: &Self) -> Ordering {
                self.ich_opener.cmp(&other.ich_opener)
            }
        }
        impl PartialOrd for BracketPair {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl PartialEq for BracketPair {
            fn eq(&self, other: &Self) -> bool {
                (self.ich_opener, self.ich_closer) == (other.ich_opener, other.ich_closer)
            }
        }
        impl Eq for BracketPair { }
    }

    fn locate_brackets(text: &str) -> Vec<bracket_pair::BracketPair> {
        let mut bracket_pair_list = Vec::<bracket_pair::BracketPair>::new();
        let mut stack_index = Vec::<u8>::new();
        let mut stack_char = Vec::<char>::new();
        //let mut next_position_on_stack:i32 = -1;
        //next_position_on_stack = next_position_on_stack - 1;

        let char_vec: Vec<_> = text.chars().collect();

        //println!("{},{}, {}", next_position_on_stack, stack_char.len(), stack_index.len());
        'outer_loop: for index in 0..char_vec.len() {
            let case = bracket_type(char_vec[index]);
            //print!("\nFor index:{} indexes[index]:, {}: pair_values[indexes[index]:{}",index, indexes[index], case);
            if case == Open {
                //println!("{}", next_position_on_stack);
                //next_position_on_stack = next_position_on_stack + 1;
                stack_index.push(index as u8);
                stack_char.push(char_vec[index]);
                //print!("-->Opening Bracket, {}", indexes[index]);
                }
            else if case == Close {
                    //print!("-->Closing Bracket, {}", indexes[index]);
                    if stack_index.len()==0 { // no opening bracket exists
                        continue 'outer_loop;
                    }//search the stack
                    for rev_index in (0..stack_index.len()).rev(){ //iterate down the stack
                        if pair_id(char_vec[index])==pair_id(stack_char[rev_index as usize]) { // match found
                            let new_br_pair: bracket_pair::BracketPair = 
                                bracket_pair::BracketPair {ich_opener:stack_index[rev_index as usize], 
                                ich_closer:index as u8}; //make a new BracketPair
                            stack_index.remove(rev_index);
                            stack_char.remove(rev_index);
                            bracket_pair_list.push(new_br_pair);
                            println!("!!Added to list[{}, {}]", new_br_pair.ich_opener, new_br_pair.ich_closer);
                            continue 'outer_loop;
                        }
                        else{// to take care of ([)] --> () and not [] 
                            stack_index.remove(rev_index); 
                            stack_char.remove(rev_index);
                        }
                    }
                }
            else if case == None {
                //print!("-->Not a Bracket, {}", indexes[index]);
            }
        }
        bracket_pair_list.sort();
        bracket_pair_list//return
    }

    fn is_strong_type_by_n0(class: BidiClass) -> bool {
        class == R || class == L
    }

    fn return_strong_type_by_n0(character: char) -> BidiClass {
        let class_of_character=bidi_class(character);
        match class_of_character {
            EN => R,
            AN => R,
            AL => R,
            R  => R,
            L  => L,
            _  => ON
        }
    }

    fn classify_pair_content(text: &str, curr_pair: 
        bracket_pair::BracketPair, dir_embed: BidiClass) -> BidiClass {
        let mut dir_opposite = ON;
        let char_vec: Vec<_> = text.chars().collect();
        for pos in curr_pair.ich_opener+1..curr_pair.ich_closer{
            //println!("return_strong_type_by_n0({}) is ON? {}", indexes[pos as usize], return_strong_type_by_n0(pos, indexes)==ON)
            let dir_found = return_strong_type_by_n0(char_vec[pos as usize]);
            if is_strong_type_by_n0(dir_found){
                if dir_found == dir_embed{
                    return dir_embed;
                } 
                dir_opposite = dir_found;
            }
        }
        //Return Opposite direction, unless no opposite direction found
        dir_opposite
    }

    fn first_strong_class_before_pair(text: &str, sos: &BidiClass,
        curr_pair: bracket_pair::BracketPair) -> BidiClass {
        let mut dir_before = ON;
        let char_vec: Vec<_> = text.chars().collect();
        'for_loop: for index in (0..curr_pair.ich_closer).rev() {
            let dir_found = return_strong_type_by_n0(char_vec[index as usize]);
            if dir_found != ON{
                dir_before = dir_found;
                break 'for_loop;
            }
        }
        *sos
    }

    fn resolve_bracket_pair(text: &str, classes: &mut [BidiClass], dir_embed: BidiClass,
                 sos: &BidiClass, curr_pair: bracket_pair::BracketPair) {
        println!("Trying to resolve {}--{}", curr_pair.ich_opener, curr_pair.ich_closer);
        let mut set_direction = true;
        let mut dir_of_pair = classify_pair_content(text, curr_pair, dir_embed);
        println!("classify_pair_content:-->IS ON:{}, IS L:{}, IS_R:{}",(dir_of_pair==ON), (dir_of_pair==L), (dir_of_pair==R));
        if  dir_of_pair == ON {
            set_direction = false;
        }
        else if dir_of_pair != dir_embed {
            let dir_before = first_strong_class_before_pair(text, sos, curr_pair);
            if dir_before == dir_embed || dir_before == ON {
                dir_of_pair = dir_embed
            }
        }
        if set_direction == true{
            set_dir_of_br_pair(classes, curr_pair, dir_of_pair);
        }
        else {
            println!("not setting direction for {}--{}", curr_pair.ich_opener, curr_pair.ich_closer);
        }
    }

    fn set_dir_of_br_pair( classes: &mut [BidiClass], br_pair: bracket_pair::BracketPair, 
                    dir_to_be_set: BidiClass) {
        println!("setting direction of pair {}--{}, as IS ON:{}, IS L:{}, IS_R:{}", br_pair.ich_opener, br_pair.ich_closer, (dir_to_be_set==ON), (dir_to_be_set==L), (dir_to_be_set==R));
        classes[br_pair.ich_opener as usize] = dir_to_be_set;
        classes[br_pair.ich_closer as usize] = dir_to_be_set;
    }

    fn resolve_all_paired_brackets(text: &str, classes: &mut [BidiClass], 
                    sos: &BidiClass, level: &u8) {
        let bracket_pair_list= locate_brackets(text);
        let dir_embed:BidiClass = 
        if 1 == (level & 1) {
            R
        } else{
            L
        };
        println!("dir_embed direction is IS ON:{}, IS L:{}, IS_R:{}", (dir_embed==ON), (dir_embed==L), (dir_embed==R));
        for br_pair in bracket_pair_list {
            println!("resolving pair {}--{}",br_pair.ich_opener, br_pair.ich_closer);
            resolve_bracket_pair(text, classes, dir_embed, sos, br_pair);
        }
    }

    pub fn resolve_n0(text: &str, sos: &BidiClass, classes: &mut [BidiClass]
            ,level: &u8) {
        resolve_all_paired_brackets(&text, classes,
            &sos, level);
    }
}

#[cfg(test)]
mod test {
    use super::BidiClass::*;

    #[test]
    fn test_initial_scan() {
        use super::{InitialProperties, initial_scan, ParagraphInfo};

        assert_eq!(initial_scan("a1", None), InitialProperties {
            initial_classes: vec![L, EN],
            paragraphs: vec![ParagraphInfo { range: 0..2, level: 0 }],
        });
        assert_eq!(initial_scan("غ א", None), InitialProperties {
            initial_classes: vec![AL, AL, WS, R, R],
            paragraphs: vec![ParagraphInfo { range: 0..5, level: 1 }],
        });
        {
            let para1 = ParagraphInfo { range: 0..4, level: 0 };
            let para2 = ParagraphInfo { range: 4..5, level: 0 };
            assert_eq!(initial_scan("a\u{2029}b", None), InitialProperties {
                initial_classes: vec![L, B, B, B, L],
                paragraphs: vec![para1, para2],
            });
        }

        let fsi = '\u{2068}';
        let pdi = '\u{2069}';

        let s = format!("{}א{}a", fsi, pdi);
        assert_eq!(initial_scan(&s, None), InitialProperties {
            initial_classes: vec![RLI, RLI, RLI, R, R, PDI, PDI, PDI, L],
            paragraphs: vec![ParagraphInfo { range: 0..9, level: 0 }],
        });
    }

    #[test]
    fn test_bidi_class() {
        use super::bidi_class;

        assert_eq!(bidi_class('c'), L);
        assert_eq!(bidi_class('\u{05D1}'), R);
        assert_eq!(bidi_class('\u{0627}'), AL);
    }

    #[test]
    fn test_process_text() {
        use super::{BidiInfo, ParagraphInfo, process_text};

        assert_eq!(process_text("abc123", Some(0)), BidiInfo {
            levels:  vec![0, 0, 0, 0,  0,  0],
            classes: vec![L, L, L, EN, EN, EN],
            paragraphs: vec![ParagraphInfo { range: 0..6, level: 0 }],
        });
        assert_eq!(process_text("abc אבג", Some(0)), BidiInfo {
            levels:  vec![0, 0, 0, 0,  1,1, 1,1, 1,1],
            classes: vec![L, L, L, WS, R,R, R,R, R,R],
            paragraphs: vec![ParagraphInfo { range: 0..10, level: 0 }],
        });
        assert_eq!(process_text("abc אבג", Some(1)), BidiInfo {
            levels:  vec![2, 2, 2, 1,  1,1, 1,1, 1,1],
            classes: vec![L, L, L, WS, R,R, R,R, R,R],
            paragraphs: vec![ParagraphInfo { range: 0..10, level: 1 }],
        });
        assert_eq!(process_text("אבג abc", Some(0)), BidiInfo {
            levels:  vec![1,1, 1,1, 1,1, 0,  0, 0, 0],
            classes: vec![R,R, R,R, R,R, WS, L, L, L],
            paragraphs: vec![ParagraphInfo { range: 0..10, level: 0 }],
        });
        assert_eq!(process_text("אבג abc", None), BidiInfo {
            levels:  vec![1,1, 1,1, 1,1, 1,  2, 2, 2],
            classes: vec![R,R, R,R, R,R, WS, L, L, L],
            paragraphs: vec![ParagraphInfo { range: 0..10, level: 1 }],
        });
        assert_eq!(process_text("غ2ظ א2ג", Some(0)), BidiInfo {
            levels:  vec![1, 1,  2,  1, 1,  1,  1,1, 2,  1,1],
            classes: vec![AL,AL, EN, AL,AL, WS, R,R, EN, R,R],
            paragraphs: vec![ParagraphInfo { range: 0..11, level: 0 }],
        });
        assert_eq!(process_text("a א.\nג", None), BidiInfo {
            classes: vec![L, WS, R,R, CS, B, R,R],
            levels:  vec![0, 0,  1,1, 0,  0, 1,1],
            paragraphs: vec![ParagraphInfo { range: 0..6, level: 0 },
                             ParagraphInfo { range: 6..8, level: 1 }],
        });
// * 
// * The comments below help locate where to push Automated Test Cases. Do not remove or change indentation.
// * 
//BeginInsertedTestCases: Test cases from BidiTest.txt go here
//EndInsertedTestCases: Test cases from BidiTest.txt go here

//assert_eq!(process_text("\u{14606}\u{1D7E9}\u{10E7E}", None), BidiInfo { levels: vec![ 4,  4,  7], classes: vec![L, EN, AN], paragraphs: vec![ParagraphInfo { range: 0..3, level: 2 } ], });

//assert_eq!(process_text("\u{0B07}\u{102E1}\u{FB29}\u{09FB}\u{FF0E}\u{11234}\u{001D}\u{000B}\u{3000}\u{26C5}\u{2066}\u{2067}\u{2068}\u{2069}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI], paragraphs: vec![ParagraphInfo { range: 0..14, level: 0 } ], });
//assert_eq!(process_text("\u{1E82B}\u{FB5D}\u{2212}\u{0E3F}\u{002C}\u{0747}\u{2029}\u{000B}\u{3000}\u{1F86D}\u{2066}\u{2067}\u{2068}\u{2069}", Some(0)), BidiInfo { levels: vec![ 7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI], paragraphs: vec![ParagraphInfo { range: 0..14, level: 1 } ], });
//assert_eq!(process_text("\u{2F96C}\u{102E6}\u{10E77}", Some(0)), BidiInfo { levels: vec![ 4,  4,  7], classes: vec![L, EN, AN], paragraphs: vec![ParagraphInfo { range: 0..3, level: 2 } ], });
// assert_eq!(process_text("\u{168E6}\u{AAA5}\u{1D780}\u{1D7C6}\u{A268}\u{1D90F}\u{1D7F9}\u{2074}\u{1F100}\u{2495}\u{0034}\u{2488}\u{207A}\u{208B}\u{207A}\u{2212}\u{FE62}\u{2212}\u{00A2}\u{20A9}\u{060A}\u{20B4}\u{09FB}\u{0609}\u{FE55}\u{060C}\u{002F}\u{002E}\u{FE55}\u{FF1A}\u{A678}\u{E01E7}\u{1C33}\u{059E}\u{1DA1B}\u{0C81}\u{0009}\u{001F}\u{0009}\u{001F}\u{0009}\u{000B}\u{2006}\u{2003}\u{2008}\u{2001}\u{1680}\u{2004}\u{2268}\u{29AE}\u{1F916}\u{1F351}\u{1F40E}\u{2A51}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..78, level: 0 } ], });
// assert_eq!(process_text("\u{10B63}\u{10AE0}\u{1E86D}\u{10B50}\u{109D4}\u{10B8B}\u{079F}\u{08B2}\u{FD14}\u{FCA6}\u{FDC0}\u{076F}\u{FE62}\u{208B}\u{FB29}\u{002B}\u{FE63}\u{FE62}\u{2032}\u{20B9}\u{00A2}\u{FF04}\u{FFE1}\u{A838}\u{FE50}\u{060C}\u{003A}\u{FE55}\u{FE55}\u{202F}\u{1D17F}\u{E0108}\u{E01ED}\u{2DEE}\u{FE05}\u{1BAC}\u{001F}\u{000B}\u{000B}\u{001F}\u{0009}\u{0009}\u{2007}\u{2006}\u{2008}\u{2003}\u{1680}\u{2008}\u{1F6B8}\u{2A78}\u{1016F}\u{27C1}\u{2EAA}\u{2054}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}", Some(0)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..78, level: 1 } ], });
// assert_eq!(process_text("\u{A6E4}\u{A979}\u{1D123}\u{1134C}\u{130C3}\u{0C26}\u{1D7FC}\u{102F6}\u{2076}\u{2496}\u{1D7F1}\u{1D7DD}\u{10E68}\u{10E7E}\u{10E75}\u{10E67}\u{10E78}\u{0604}", Some(0)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  7,  7,  7,  7], classes: vec![L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, AN, AN, AN, AN, AN, AN], paragraphs: vec![ParagraphInfo { range: 0..18, level: 2 } ], });
// assert_eq!(process_text("\u{000A}\u{001F}\u{1680}\u{2066}\u{2067}\u{2068}\u{2069}\u{0085}\u{001F}\u{1680}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{001F}\u{2002}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{0009}\u{000C}\u{2066}\u{2067}\u{2068}\u{2069}\u{A44C}\u{102E3}\u{FF0B}\u{20AA}\u{00A0}\u{05A6}\u{000A}\u{000B}\u{3000}\u{1D329}\u{2066}\u{2067}\u{2068}\u{2069}\u{A699}\u{0035}\u{FE63}\u{0BF9}\u{FE55}\u{E01D7}\u{001D}\u{000B}\u{2003}\u{FE56}\u{2066}\u{2067}\u{2068}\u{2069}", Some(1)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI], paragraphs: vec![ParagraphInfo { range: 0..56, level: 0 } ], });
// assert_eq!(process_text("\u{000A}\u{001F}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{000D}\u{001F}\u{2009}\u{2066}\u{2067}\u{2068}\u{2069}\u{10B2F}\u{FCED}\u{002D}\u{20BB}\u{FF0E}\u{0FB8}\u{001D}\u{001F}\u{2006}\u{25AD}\u{2066}\u{2067}\u{2068}\u{2069}\u{01A6}\u{07CC}\u{FCEC}\u{1F100}\u{FE63}\u{0609}\u{10E75}\u{FF0F}\u{1D189}\u{001D}\u{0009}\u{0020}\u{25AB}\u{2066}\u{2067}\u{2068}\u{2069}\u{10916}\u{FC70}\u{208B}\u{00A3}\u{FE50}\u{17D3}\u{2029}\u{001F}\u{2007}\u{2535}\u{2066}\u{2067}\u{2068}\u{2069}\u{10A9B}\u{FC1F}\u{208A}\u{20A7}\u{FF1A}\u{1CDC}\u{001D}\u{0009}\u{0020}\u{1F870}\u{2066}\u{2067}\u{2068}\u{2069}", Some(1)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  3,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, L, R, AL, EN, ES, ET, AN, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI], paragraphs: vec![ParagraphInfo { range: 0..73, level: 1 } ], });
// assert_eq!(process_text("\u{0C0A}\u{1D7EF}\u{207A}\u{FF03}\u{FE52}\u{1B6C}\u{2E20}\u{A040}\u{07CC}\u{075A}\u{102EB}\u{208B}\u{2030}\u{10E6F}\u{FE55}\u{07ED}\u{1D222}\u{1D815}\u{1D7D7}\u{0601}\u{120AA}\u{2497}\u{10E61}\u{1D12E}\u{1D7F0}\u{10E78}", Some(1)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  3,  3,  3,  4,  4,  7,  4,  4,  7], classes: vec![L, EN, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, L, EN, AN, L, EN, AN], paragraphs: vec![ParagraphInfo { range: 0..26, level: 2 } ], });
// assert_eq!(process_text("\u{10B43}\u{FB57}\u{0804}\u{1EE64}\u{FE63}\u{20BD}\u{002F}\u{05B0}\u{2ED7}\u{1D6B6}\u{10A9D}\u{0640}\u{2087}\u{002B}\u{09FB}\u{10E7C}\u{002C}\u{16AF3}\u{1F311}", Some(1)), BidiInfo { levels: vec![ 7,  7,  5,  5,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON], paragraphs: vec![ParagraphInfo { range: 0..19, level: 3 } ], });
// assert_eq!(process_text("\u{10E67}\u{1D682}\u{1D7FB}\u{10E69}", Some(1)), BidiInfo { levels: vec![ 7,  4,  4,  4], classes: vec![AN, L, EN, AN], paragraphs: vec![ParagraphInfo { range: 0..4, level: 4 } ], });
// assert_eq!(process_text("\u{1504}\u{04FB}\u{3325}\u{2088}\u{13156}\u{208A}\u{1301B}\u{20B8}\u{A817}\u{FF1A}\u{03EF}\u{E01EF}\u{191D}\u{001C}\u{100E1}\u{000B}\u{13265}\u{2005}\u{13364}\u{1F35C}\u{A898}\u{2066}\u{A0D8}\u{2067}\u{194D}\u{2068}\u{112A4}\u{2069}\u{249A}\u{013B}\u{248F}\u{102E2}\u{1D7D1}\u{002D}\u{0038}\u{20AD}\u{102ED}\u{003A}\u{FF14}\u{135D}\u{102F7}\u{001C}\u{2494}\u{0009}\u{102EC}\u{2005}\u{1D7F5}\u{2B85}\u{102FA}\u{2066}\u{1D7F6}\u{2067}\u{2074}\u{2068}\u{1D7FD}\u{2069}\u{002B}\u{121E9}\u{207A}\u{102E6}\u{FE62}\u{FB29}\u{2212}\u{20A7}\u{FF0D}\u{060C}\u{FF0B}\u{1DAA1}\u{FE62}\u{0085}\u{207B}\u{0009}\u{FF0B}\u{3000}\u{208B}\u{3014}\u{FB29}\u{2066}\u{FB29}\u{2067}\u{208B}\u{2068}\u{FF0D}\u{2069}\u{20BC}\u{2F8E0}\u{17DB}\u{FF14}\u{20B7}\u{207B}\u{2034}\u{2031}\u{FFE6}\u{FF1A}\u{00A4}\u{AAC1}\u{2031}\u{000A}\u{0E3F}\u{0009}\u{066A}\u{2028}\u{20B3}\u{19FB}\u{2213}\u{2066}\u{A839}\u{2067}\u{00A2}\u{2068}\u{09F3}\u{2069}\u{FF0C}\u{A72F}\u{FE52}\u{06F2}\u{002F}\u{207A}\u{FF0F}\u{2213}\u{FF1A}\u{2044}\u{00A0}\u{033D}\u{003A}\u{000A}\u{2044}\u{001F}\u{060C}\u{200A}\u{FE55}\u{2476}\u{FF0E}\u{2066}\u{202F}\u{2067}\u{002C}\u{2068}\u{202F}\u{2069}\u{1D243}\u{04CD}\u{0318}\u{102F1}\u{E0165}\u{FB29}\u{081D}\u{A839}\u{E0151}\u{060C}\u{1A17}\u{1DA29}\u{0B44}\u{001E}\u{05A6}\u{0009}\u{05A7}\u{2028}\u{1A73}\u{2956}\u{E0157}\u{2066}\u{A8E6}\u{2067}\u{E01A1}\u{2068}\u{05A3}\u{2069}\u{000B}\u{A14B}\u{000B}\u{102ED}\u{001F}\u{FE62}\u{001F}\u{FE5F}\u{000B}\u{FF0E}\u{001F}\u{0C3E}\u{001F}\u{001C}\u{000B}\u{000B}\u{001F}\u{2003}\u{0009}\u{1F491}\u{001F}\u{2066}\u{0009}\u{2067}\u{001F}\u{2068}\u{0009}\u{2069}\u{2028}\u{2836}\u{1680}\u{102E7}\u{2000}\u{207B}\u{2003}\u{20B9}\u{2005}\u{FF0E}\u{0020}\u{0F9B}\u{2009}\u{001C}\u{3000}\u{001F}\u{2009}\u{2001}\u{205F}\u{2575}\u{1680}\u{2066}\u{2001}\u{2067}\u{2009}\u{2068}\u{2005}\u{2069}\u{1D224}\u{17E3}\u{2AB0}\u{06F5}\u{27FA}\u{207B}\u{2E39}\u{FE69}\u{250F}\u{002F}\u{2B36}\u{0322}\u{321D}\u{000D}\u{205C}\u{000B}\u{1F5CB}\u{2006}\u{2E81}\u{2465}\u{1F318}\u{2066}\u{1F4CD}\u{2067}\u{1F3A9}\u{2068}\u{02F0}\u{2069}\u{2066}\u{2029}\u{2066}\u{0009}\u{2066}\u{205F}\u{2066}\u{2066}\u{2066}\u{2067}\u{2066}\u{2068}\u{2066}\u{2069}\u{2067}\u{000A}\u{2067}\u{000B}\u{2067}\u{2005}\u{2067}\u{2066}\u{2067}\u{2067}\u{2067}\u{2068}\u{2067}\u{2069}\u{2068}\u{000D}\u{2068}\u{0009}\u{2068}\u{2003}\u{2068}\u{2066}\u{2068}\u{2067}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{1000F}\u{2069}\u{1D7D8}\u{2069}\u{FE63}\u{2069}\u{20B5}\u{2069}\u{FE55}\u{2069}\u{0300}\u{2069}\u{2029}\u{2069}\u{0009}\u{2069}\u{0020}\u{2069}\u{3016}\u{2069}\u{2066}\u{2069}\u{2067}\u{2069}\u{2068}\u{2069}\u{2069}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, L, L, EN, L, ES, L, ET, L, CS, L, NSM, L, B, L, S, L, WS, L, ON, L, LRI, L, RLI, L, FSI, L, PDI, EN, L, EN, EN, EN, ES, EN, ET, EN, CS, EN, NSM, EN, B, EN, S, EN, WS, EN, ON, EN, LRI, EN, RLI, EN, FSI, EN, PDI, ES, L, ES, EN, ES, ES, ES, ET, ES, CS, ES, NSM, ES, B, ES, S, ES, WS, ES, ON, ES, LRI, ES, RLI, ES, FSI, ES, PDI, ET, L, ET, EN, ET, ES, ET, ET, ET, CS, ET, NSM, ET, B, ET, S, ET, WS, ET, ON, ET, LRI, ET, RLI, ET, FSI, ET, PDI, CS, L, CS, EN, CS, ES, CS, ET, CS, CS, CS, NSM, CS, B, CS, S, CS, WS, CS, ON, CS, LRI, CS, RLI, CS, FSI, CS, PDI, NSM, L, NSM, EN, NSM, ES, NSM, ET, NSM, CS, NSM, NSM, NSM, B, NSM, S, NSM, WS, NSM, ON, NSM, LRI, NSM, RLI, NSM, FSI, NSM, PDI, S, L, S, EN, S, ES, S, ET, S, CS, S, NSM, S, B, S, S, S, WS, S, ON, S, LRI, S, RLI, S, FSI, S, PDI, WS, L, WS, EN, WS, ES, WS, ET, WS, CS, WS, NSM, WS, B, WS, S, WS, WS, WS, ON, WS, LRI, WS, RLI, WS, FSI, WS, PDI, ON, L, ON, EN, ON, ES, ON, ET, ON, CS, ON, NSM, ON, B, ON, S, ON, WS, ON, ON, ON, LRI, ON, RLI, ON, FSI, ON, PDI, LRI, B, LRI, S, LRI, WS, LRI, LRI, LRI, RLI, LRI, FSI, LRI, PDI, RLI, B, RLI, S, RLI, WS, RLI, LRI, RLI, RLI, RLI, FSI, RLI, PDI, FSI, B, FSI, S, FSI, WS, FSI, LRI, FSI, RLI, FSI, FSI, FSI, PDI, PDI, L, PDI, EN, PDI, ES, PDI, ET, PDI, CS, PDI, NSM, PDI, B, PDI, S, PDI, WS, PDI, ON, PDI, LRI, PDI, RLI, PDI, FSI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..161, level: 0 } ], });
// assert_eq!(process_text("\u{0522}\u{10C14}\u{3337}\u{FC01}\u{2087}\u{108E1}\u{1F101}\u{FC2E}\u{FB29}\u{07E7}\u{FF0D}\u{06A5}\u{20B9}\u{FB28}\u{00A4}\u{FD2C}\u{060C}\u{1099D}\u{FF0C}\u{FB8F}\u{1CED}\u{1081A}\u{1DCC}\u{FCD6}\u{000B}\u{10923}\u{001F}\u{06FC}\u{2028}\u{FB4A}\u{1680}\u{06C5}\u{266A}\u{10CAC}\u{1D213}\u{FCE5}\u{2067}\u{1082E}\u{2067}\u{FD6A}\u{2067}\u{002B}\u{2067}\u{060A}\u{2067}\u{00A0}\u{2067}\u{FE04}\u{2067}\u{2FA1}\u{2068}\u{10A2C}\u{2068}\u{FD75}\u{2069}\u{10C97}\u{2069}\u{1EE30}", Some(0)), BidiInfo { levels: vec![ 3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2], classes: vec![L, R, L, AL, EN, R, EN, AL, ES, R, ES, AL, ET, R, ET, AL, CS, R, CS, AL, NSM, R, NSM, AL, S, R, S, AL, WS, R, WS, AL, ON, R, ON, AL, RLI, R, RLI, AL, RLI, ES, RLI, ET, RLI, CS, RLI, NSM, RLI, ON, FSI, R, FSI, AL, PDI, R, PDI, AL], paragraphs: vec![ParagraphInfo { range: 0..29, level: 0 } ], });
// assert_eq!(process_text("\u{114D8}\u{0666}\u{0030}\u{0664}\u{FE62}\u{0664}\u{20A8}\u{10E65}\u{FF0E}\u{10E77}\u{1DA6C}\u{0667}\u{0009}\u{0664}\u{2005}\u{10E7E}\u{058A}\u{10E78}\u{2066}\u{A420}\u{2066}\u{248A}\u{2066}\u{002B}\u{2066}\u{17DB}\u{2066}\u{FF1A}\u{2066}\u{0FA5}\u{2066}\u{2047}\u{2067}\u{A340}\u{2067}\u{FF10}\u{2067}\u{10E72}\u{2068}\u{1681D}\u{2068}\u{0037}\u{2068}\u{002B}\u{2068}\u{2034}\u{2068}\u{FF1A}\u{2068}\u{11724}\u{2068}\u{276D}\u{2069}\u{10E69}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, AN, EN, AN, ES, AN, ET, AN, CS, AN, NSM, AN, S, AN, WS, AN, ON, AN, LRI, L, LRI, EN, LRI, ES, LRI, ET, LRI, CS, LRI, NSM, LRI, ON, RLI, L, RLI, EN, RLI, AN, FSI, L, FSI, EN, FSI, ES, FSI, ET, FSI, CS, FSI, NSM, FSI, ON, PDI, AN], paragraphs: vec![ParagraphInfo { range: 0..27, level: 0 } ], });
// assert_eq!(process_text("\u{2066}\u{1087E}\u{2066}\u{FBF6}", Some(0)), BidiInfo { levels: vec![ 3,  3], classes: vec![LRI, R, LRI, AL], paragraphs: vec![ParagraphInfo { range: 0..2, level: 0 } ], });
// assert_eq!(process_text("\u{2066}\u{10E63}\u{2068}\u{10E65}", Some(0)), BidiInfo { levels: vec![ 3,  3], classes: vec![LRI, AN, FSI, AN], paragraphs: vec![ParagraphInfo { range: 0..2, level: 0 } ], });
// assert_eq!(process_text("\u{011C}\u{16957}\u{A37C}\u{2F8E4}\u{1D517}\u{1D0D3}\u{A1D2}\u{1BF3}\u{1D577}\u{1403}\u{1FE9}\u{1206A}\u{F9DE}\u{1555}\u{0AEF}\u{168F3}\u{1D8AD}\u{1016}\u{055C}\u{1440C}\u{3228}\u{D7DE}\u{A4EC}\u{12458}\u{1205C}\u{044E}\u{172A}\u{118AF}\u{16F7E}\u{1F174}\u{1C59}\u{1561}\u{306F}\u{131FF}\u{0E51}\u{0F26}\u{06F0}\u{2490}\u{06F0}\u{102E8}\u{1D7D9}\u{2489}\u{1D7FF}\u{1D7FF}\u{2077}\u{2075}\u{102F5}\u{1D7F8}\u{102E3}\u{FF10}\u{2499}\u{1D7E7}\u{1D7F1}\u{2088}\u{1D7F9}\u{102EE}\u{2081}\u{248E}\u{248C}\u{102F7}\u{1D7E1}\u{102FB}\u{2080}\u{2089}\u{2490}\u{FF14}\u{1D7D4}\u{1D7F6}\u{1D7F2}\u{102E8}\u{00B9}\u{1D7ED}\u{208B}\u{FE62}\u{207B}\u{FF0D}\u{207B}\u{208B}\u{207B}\u{002B}\u{207B}\u{FF0B}\u{207A}\u{FF0D}\u{FE63}\u{207B}\u{FE63}\u{FF0D}\u{208A}\u{FF0D}\u{2212}\u{207A}\u{208A}\u{FE62}\u{002B}\u{208B}\u{FF0B}\u{002B}\u{208B}\u{002D}\u{FE63}\u{FE62}\u{002D}\u{002B}\u{207A}\u{207A}\u{208B}\u{208A}\u{20A8}\u{20BD}\u{20AC}\u{FE69}\u{20AC}\u{0E3F}\u{A839}\u{09F2}\u{20B0}\u{20B9}\u{2034}\u{2034}\u{FFE5}\u{20B1}\u{20A1}\u{20B5}\u{2030}\u{20B9}\u{20B6}\u{060A}\u{2031}\u{066A}\u{20A0}\u{20BE}\u{20A2}\u{2034}\u{2030}\u{0025}\u{20BA}\u{20A2}\u{0E3F}\u{FFE6}\u{20B9}\u{2032}\u{20AA}\u{2033}\u{002F}\u{002E}\u{FE52}\u{002F}\u{202F}\u{002F}\u{060C}\u{FF0C}\u{060C}\u{FE50}\u{2044}\u{FE55}\u{FF0F}\u{060C}\u{FF0C}\u{002C}\u{FF0E}\u{060C}\u{002C}\u{FF0E}\u{FF1A}\u{002C}\u{202F}\u{060C}\u{FF0F}\u{FE52}\u{FF0C}\u{00A0}\u{002E}\u{003A}\u{FF0C}\u{FF0C}\u{2044}\u{FE52}\u{FE50}\u{002E}\u{111BB}\u{0734}\u{20E5}\u{E0189}\u{0354}\u{0953}\u{AABE}\u{E0113}\u{AAB3}\u{111B9}\u{116B2}\u{0F79}\u{AA2C}\u{E012F}\u{1E8D0}\u{036F}\u{E0166}\u{031C}\u{0FA5}\u{FE07}\u{1DD1}\u{112DF}\u{111CB}\u{E01B8}\u{E01C7}\u{E01B2}\u{1D18B}\u{0FBA}\u{030C}\u{114B3}\u{11101}\u{A9BC}\u{0C4B}\u{1DA69}\u{1D17D}\u{0FBC}\u{0009}\u{001F}\u{0009}\u{000B}\u{001F}\u{0009}\u{000B}\u{001F}\u{000B}\u{0009}\u{001F}\u{001F}\u{0009}\u{001F}\u{001F}\u{0009}\u{000B}\u{000B}\u{001F}\u{001F}\u{001F}\u{001F}\u{001F}\u{001F}\u{001F}\u{000B}\u{000B}\u{0009}\u{000B}\u{000B}\u{001F}\u{001F}\u{000B}\u{001F}\u{0009}\u{001F}\u{2009}\u{2006}\u{2004}\u{3000}\u{2002}\u{2007}\u{0020}\u{2002}\u{2007}\u{000C}\u{2009}\u{2006}\u{2002}\u{2009}\u{2002}\u{2008}\u{2009}\u{205F}\u{2007}\u{2009}\u{000C}\u{000C}\u{2000}\u{2001}\u{2001}\u{3000}\u{2008}\u{2006}\u{2028}\u{205F}\u{2008}\u{200A}\u{2003}\u{2008}\u{1680}\u{200A}\u{226B}\u{0375}\u{240C}\u{294A}\u{2B26}\u{21DA}\u{1F61D}\u{26C3}\u{2A3F}\u{2923}\u{29FA}\u{2F8A}\u{23EF}\u{2CFD}\u{1F873}\u{02EF}\u{1F5AF}\u{295A}\u{2931}\u{290B}\u{219E}\u{10B3E}\u{2027}\u{1D30F}\u{27CB}\u{275B}\u{27BC}\u{2660}\u{27C0}\u{2748}\u{FE17}\u{A4B2}\u{1F4E4}\u{02CD}\u{22C5}\u{31D6}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..468, level: 0 } ], });
// assert_eq!(process_text("\u{10A98}\u{11704}\u{083B}\u{208A}\u{10B55}\u{20AF}\u{109AC}\u{FF0E}\u{1E893}\u{001C}\u{10CCC}\u{000B}\u{10829}\u{0020}\u{1E8AA}\u{25F6}\u{108FC}\u{2066}\u{10AEF}\u{2067}\u{1E83F}\u{2068}\u{10CAC}\u{2069}\u{FBF7}\u{122DD}\u{0752}\u{208B}\u{FEFB}\u{20AA}\u{FB8D}\u{002C}\u{FD27}\u{2029}\u{FD19}\u{000B}\u{FD27}\u{0020}\u{FEC8}\u{0F3C}\u{FC78}\u{2066}\u{FB62}\u{2067}\u{1EE07}\u{2068}\u{FB77}\u{2069}", Some(0)), BidiInfo { levels: vec![ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2], classes: vec![R, L, R, ES, R, ET, R, CS, R, B, R, S, R, WS, R, ON, R, LRI, R, RLI, R, FSI, R, PDI, AL, L, AL, ES, AL, ET, AL, CS, AL, B, AL, S, AL, WS, AL, ON, AL, LRI, AL, RLI, AL, FSI, AL, PDI], paragraphs: vec![ParagraphInfo { range: 0..24, level: 1 } ], });
// assert_eq!(process_text("\u{108AB}\u{108FB}\u{10C8E}\u{FBDC}\u{10C18}\u{FF0D}\u{10CC9}\u{FFE0}\u{10CEB}\u{FE52}\u{1E866}\u{0945}\u{10853}\u{2029}\u{05E5}\u{000B}\u{FB31}\u{1680}\u{10B09}\u{1F71B}\u{109C8}\u{2066}\u{07CC}\u{2067}\u{1E8CE}\u{2068}\u{109AF}\u{2069}\u{0791}\u{10CC5}\u{FDC5}\u{FBE2}\u{1EE42}\u{208B}\u{FE95}\u{20BB}\u{1EE22}\u{FF0C}\u{FDA0}\u{0741}\u{1EE71}\u{2029}\u{FEEE}\u{0009}\u{FB76}\u{2008}\u{FD5E}\u{1014E}\u{FE8E}\u{2066}\u{FD77}\u{2067}\u{FBBC}\u{2068}\u{FCDF}\u{2069}\u{208A}\u{10ACD}\u{002D}\u{06C5}\u{FE63}\u{002B}\u{FE62}\u{2033}\u{FE63}\u{FE55}\u{FE63}\u{E0137}\u{FF0D}\u{001E}\u{2212}\u{0009}\u{208A}\u{000C}\u{FF0D}\u{2997}\u{FB29}\u{2066}\u{207B}\u{2067}\u{207A}\u{2068}\u{FE63}\u{2069}\u{20A4}\u{10C34}\u{00B0}\u{06A5}\u{20AE}\u{FF0B}\u{FF03}\u{20A8}\u{2031}\u{FF0F}\u{20A6}\u{0A75}\u{00B1}\u{000A}\u{09F3}\u{001F}\u{20B4}\u{2006}\u{0609}\u{26E6}\u{A839}\u{2066}\u{20B4}\u{2067}\u{17DB}\u{2068}\u{20AE}\u{2069}\u{00A0}\u{10845}\u{FE55}\u{FE7B}\u{00A0}\u{207B}\u{002E}\u{20AA}\u{FE50}\u{FF1A}\u{FE50}\u{2DE1}\u{060C}\u{0085}\u{FF0C}\u{000B}\u{00A0}\u{2007}\u{FE55}\u{FFEA}\u{002F}\u{2066}\u{002F}\u{2067}\u{FF0F}\u{2068}\u{00A0}\u{2069}\u{031D}\u{10AEC}\u{05AC}\u{071F}\u{E0110}\u{208B}\u{E015E}\u{A839}\u{1DC9}\u{00A0}\u{0597}\u{E01B0}\u{E01E8}\u{2029}\u{20D0}\u{001F}\u{1DA66}\u{2001}\u{0A81}\u{1F60B}\u{A982}\u{2066}\u{11370}\u{2067}\u{09E2}\u{2068}\u{08F1}\u{2069}\u{001F}\u{05D9}\u{0009}\u{1EE00}\u{001F}\u{002D}\u{0009}\u{20BB}\u{001F}\u{FF0E}\u{000B}\u{E01E4}\u{001F}\u{2029}\u{000B}\u{0009}\u{0009}\u{2028}\u{0009}\u{260C}\u{001F}\u{2066}\u{0009}\u{2067}\u{0009}\u{2068}\u{001F}\u{2069}\u{2005}\u{10B61}\u{2001}\u{1EE29}\u{2001}\u{FB29}\u{2004}\u{20BE}\u{200A}\u{FF0C}\u{2008}\u{035A}\u{2004}\u{001C}\u{205F}\u{001F}\u{205F}\u{2009}\u{3000}\u{1F51B}\u{1680}\u{2066}\u{0020}\u{2067}\u{2006}\u{2068}\u{2008}\u{2069}\u{1F694}\u{1089D}\u{1F0D3}\u{06E6}\u{2255}\u{FB29}\u{FE19}\u{20B4}\u{1F360}\u{FF0C}\u{2581}\u{111BC}\u{1F084}\u{000D}\u{22A5}\u{0009}\u{FFFC}\u{205F}\u{19DF}\u{2666}\u{FE17}\u{2066}\u{2249}\u{2067}\u{2B2D}\u{2068}\u{227A}\u{2069}\u{2066}\u{001D}\u{2066}\u{0009}\u{2066}\u{2028}\u{2066}\u{2066}\u{2066}\u{2067}\u{2066}\u{2068}\u{2066}\u{2069}\u{2067}\u{2029}\u{2067}\u{001F}\u{2067}\u{2000}\u{2067}\u{2066}\u{2067}\u{2067}\u{2067}\u{2068}\u{2067}\u{2069}\u{2068}\u{0085}\u{2068}\u{0009}\u{2068}\u{200A}\u{2068}\u{2066}\u{2068}\u{2067}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{1E889}\u{2069}\u{FBBC}\u{2069}\u{FE62}\u{2069}\u{20A5}\u{2069}\u{2044}\u{2069}\u{05C2}\u{2069}\u{000A}\u{2069}\u{000B}\u{2069}\u{2005}\u{2069}\u{21D2}\u{2069}\u{2066}\u{2069}\u{2067}\u{2069}\u{2068}\u{2069}\u{2069}", Some(1)), BidiInfo { levels: vec![ 7,  7,  5,  5,  5,  7,  5,  5,  5,  5,  5,  5,  5,  5,  7,  7,  5,  5,  5,  7,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, R, R, AL, R, ES, R, ET, R, CS, R, NSM, R, B, R, S, R, WS, R, ON, R, LRI, R, RLI, R, FSI, R, PDI, AL, R, AL, AL, AL, ES, AL, ET, AL, CS, AL, NSM, AL, B, AL, S, AL, WS, AL, ON, AL, LRI, AL, RLI, AL, FSI, AL, PDI, ES, R, ES, AL, ES, ES, ES, ET, ES, CS, ES, NSM, ES, B, ES, S, ES, WS, ES, ON, ES, LRI, ES, RLI, ES, FSI, ES, PDI, ET, R, ET, AL, ET, ES, ET, ET, ET, CS, ET, NSM, ET, B, ET, S, ET, WS, ET, ON, ET, LRI, ET, RLI, ET, FSI, ET, PDI, CS, R, CS, AL, CS, ES, CS, ET, CS, CS, CS, NSM, CS, B, CS, S, CS, WS, CS, ON, CS, LRI, CS, RLI, CS, FSI, CS, PDI, NSM, R, NSM, AL, NSM, ES, NSM, ET, NSM, CS, NSM, NSM, NSM, B, NSM, S, NSM, WS, NSM, ON, NSM, LRI, NSM, RLI, NSM, FSI, NSM, PDI, S, R, S, AL, S, ES, S, ET, S, CS, S, NSM, S, B, S, S, S, WS, S, ON, S, LRI, S, RLI, S, FSI, S, PDI, WS, R, WS, AL, WS, ES, WS, ET, WS, CS, WS, NSM, WS, B, WS, S, WS, WS, WS, ON, WS, LRI, WS, RLI, WS, FSI, WS, PDI, ON, R, ON, AL, ON, ES, ON, ET, ON, CS, ON, NSM, ON, B, ON, S, ON, WS, ON, ON, ON, LRI, ON, RLI, ON, FSI, ON, PDI, LRI, B, LRI, S, LRI, WS, LRI, LRI, LRI, RLI, LRI, FSI, LRI, PDI, RLI, B, RLI, S, RLI, WS, RLI, LRI, RLI, RLI, RLI, FSI, RLI, PDI, FSI, B, FSI, S, FSI, WS, FSI, LRI, FSI, RLI, FSI, FSI, FSI, PDI, PDI, R, PDI, AL, PDI, ES, PDI, ET, PDI, CS, PDI, NSM, PDI, B, PDI, S, PDI, WS, PDI, ON, PDI, LRI, PDI, RLI, PDI, FSI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..161, level: 1 } ], });
// assert_eq!(process_text("\u{10928}\u{110E8}\u{10C3B}\u{FF13}\u{1E8B2}\u{10E61}\u{FCEE}\u{132C3}\u{FD1F}\u{2082}\u{FC94}\u{10E6D}\u{FF0D}\u{1131E}\u{FB29}\u{1F101}\u{207A}\u{0668}\u{060A}\u{14500}\u{20BE}\u{10E68}\u{FE50}\u{16AE3}\u{003A}\u{1D7F8}\u{002E}\u{10E71}\u{A8E1}\u{1442C}\u{20E7}\u{1D7CF}\u{2DEA}\u{10E60}\u{0009}\u{A477}\u{0009}\u{102F7}\u{0009}\u{10E72}\u{2002}\u{1C43}\u{2009}\u{248E}\u{2000}\u{10E64}\u{1F6C8}\u{1533}\u{2481}\u{0030}\u{A4A2}\u{0665}\u{2066}\u{16800}\u{2066}\u{102F0}\u{2066}\u{FF0D}\u{2066}\u{0BF9}\u{2066}\u{FF1A}\u{2066}\u{0652}\u{2066}\u{2A3A}\u{2068}\u{FF79}\u{2068}\u{248A}\u{2068}\u{207B}\u{2068}\u{FF04}\u{2068}\u{FF0F}\u{2068}\u{1DA32}\u{2068}\u{2E0B}\u{2069}\u{16857}\u{2069}\u{1D7DB}\u{2069}\u{0668}", Some(1)), BidiInfo { levels: vec![ 5,  7,  7,  5,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, L, R, EN, R, AN, AL, L, AL, EN, AL, AN, ES, L, ES, EN, ES, AN, ET, L, ET, AN, CS, L, CS, EN, CS, AN, NSM, L, NSM, EN, NSM, AN, S, L, S, EN, S, AN, WS, L, WS, EN, WS, AN, ON, L, ON, EN, ON, AN, LRI, L, LRI, EN, LRI, ES, LRI, ET, LRI, CS, LRI, NSM, LRI, ON, FSI, L, FSI, EN, FSI, ES, FSI, ET, FSI, CS, FSI, NSM, FSI, ON, PDI, L, PDI, EN, PDI, AN], paragraphs: vec![ParagraphInfo { range: 0..43, level: 1 } ], });
// assert_eq!(process_text("\u{2066}\u{1E85E}\u{2066}\u{0782}\u{2067}\u{FB31}\u{2067}\u{1EE4B}\u{2067}\u{FF0B}\u{2067}\u{0024}\u{2067}\u{060C}\u{2067}\u{1A18}\u{2067}\u{2BD1}\u{2068}\u{10AE4}\u{2068}\u{FEA6}", Some(1)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![LRI, R, LRI, AL, RLI, R, RLI, AL, RLI, ES, RLI, ET, RLI, CS, RLI, NSM, RLI, ON, FSI, R, FSI, AL], paragraphs: vec![ParagraphInfo { range: 0..11, level: 1 } ], });
// assert_eq!(process_text("\u{2066}\u{10E73}\u{2067}\u{2F874}\u{2067}\u{102E2}\u{2067}\u{0668}\u{2068}\u{10E6C}", Some(1)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4], classes: vec![LRI, AN, RLI, L, RLI, EN, RLI, AN, FSI, AN], paragraphs: vec![ParagraphInfo { range: 0..5, level: 1 } ], });
// assert_eq!(process_text("\u{10A24}\u{0814}\u{10B1E}\u{10C1C}\u{05D5}\u{0809}\u{109A1}\u{109A5}\u{0857}\u{10AC2}\u{109CA}\u{07E2}\u{1E866}\u{10905}\u{10B14}\u{10B45}\u{10811}\u{10A31}\u{10B54}\u{10810}\u{1E88E}\u{10C88}\u{1E84D}\u{0808}\u{10AD6}\u{10CEA}\u{1E8C7}\u{1E8AE}\u{10B49}\u{10B31}\u{05E9}\u{10B11}\u{1E8A9}\u{05BE}\u{10923}\u{10BAA}\u{060B}\u{FC8B}\u{FECE}\u{1EE92}\u{0722}\u{1EE06}\u{08A9}\u{FCA6}\u{0753}\u{FB51}\u{FCAA}\u{1EE79}\u{FCD6}\u{FD36}\u{1EEB8}\u{FB62}\u{FBE9}\u{0708}\u{0620}\u{1EE27}\u{1EE84}\u{1EE8D}\u{0728}\u{0726}\u{1EE27}\u{FD7F}\u{0684}\u{FCD6}\u{1EE5D}\u{0783}\u{FDBF}\u{FCA5}\u{FB6C}\u{FD0F}\u{FD02}\u{1EE89}\u{207B}\u{207B}\u{002D}\u{FE62}\u{002B}\u{FE63}\u{FE63}\u{002B}\u{FE62}\u{FF0D}\u{208A}\u{FE62}\u{2212}\u{FE63}\u{207A}\u{FB29}\u{207A}\u{002D}\u{207A}\u{FB29}\u{FB29}\u{FE63}\u{FF0D}\u{207A}\u{207A}\u{002B}\u{002D}\u{207B}\u{208A}\u{207A}\u{FE63}\u{207A}\u{FE63}\u{208B}\u{FE63}\u{207A}\u{20B5}\u{20A4}\u{212E}\u{FFE1}\u{20B1}\u{00A4}\u{00A4}\u{2034}\u{20BE}\u{FFE0}\u{09F2}\u{2032}\u{00B0}\u{20A0}\u{0AF1}\u{FE5F}\u{066A}\u{20B8}\u{0AF1}\u{20BC}\u{058F}\u{20A6}\u{0BF9}\u{20B8}\u{00B0}\u{20B3}\u{066A}\u{FFE1}\u{FFE5}\u{20B0}\u{20A0}\u{20A1}\u{FE69}\u{066A}\u{17DB}\u{2032}\u{003A}\u{002E}\u{FF0F}\u{FF0C}\u{FF0C}\u{FE55}\u{FF0E}\u{002C}\u{FF0C}\u{060C}\u{060C}\u{002F}\u{FF0F}\u{FF0E}\u{060C}\u{FF0F}\u{002F}\u{003A}\u{060C}\u{003A}\u{003A}\u{FE52}\u{FF0E}\u{002E}\u{FF0E}\u{FE50}\u{00A0}\u{FF0F}\u{002F}\u{002F}\u{002C}\u{060C}\u{FF1A}\u{FF0E}\u{00A0}\u{FF0E}\u{0320}\u{0336}\u{E0133}\u{E0139}\u{E010C}\u{115DD}\u{1A66}\u{06D8}\u{1DA00}\u{065F}\u{FB1E}\u{E01C7}\u{1DF0}\u{1D17F}\u{05BD}\u{A947}\u{0E38}\u{111BA}\u{1D182}\u{1921}\u{0361}\u{0F7A}\u{FE24}\u{1112D}\u{0900}\u{0F90}\u{0FA3}\u{09CD}\u{0AE2}\u{A825}\u{10A3F}\u{A8F0}\u{E016C}\u{E01DF}\u{E012C}\u{10A0E}\u{001F}\u{0009}\u{001F}\u{0009}\u{000B}\u{001F}\u{000B}\u{000B}\u{001F}\u{0009}\u{0009}\u{001F}\u{0009}\u{0009}\u{001F}\u{000B}\u{001F}\u{001F}\u{000B}\u{001F}\u{000B}\u{000B}\u{000B}\u{000B}\u{0009}\u{001F}\u{001F}\u{001F}\u{001F}\u{0009}\u{0009}\u{0009}\u{000B}\u{000B}\u{0009}\u{000B}\u{2008}\u{2008}\u{2006}\u{2009}\u{2005}\u{3000}\u{2008}\u{1680}\u{2005}\u{2006}\u{2003}\u{3000}\u{2005}\u{2008}\u{2004}\u{3000}\u{2009}\u{2009}\u{2009}\u{1680}\u{200A}\u{2008}\u{2005}\u{200A}\u{2028}\u{2009}\u{1680}\u{205F}\u{200A}\u{2004}\u{2004}\u{2006}\u{2000}\u{2004}\u{200A}\u{2028}\u{2F19}\u{1F6F2}\u{A716}\u{29A0}\u{23F4}\u{1D217}\u{0C7D}\u{29B2}\u{1F52C}\u{26D3}\u{02DF}\u{1F3BA}\u{2F82}\u{21CD}\u{1F690}\u{2BED}\u{1F666}\u{3254}\u{1D30E}\u{1F6AC}\u{1F473}\u{1F04E}\u{10179}\u{1F887}\u{2F44}\u{2237}\u{1F751}\u{1F6E5}\u{26BB}\u{1F5F3}\u{2FB1}\u{3377}\u{2F83}\u{27FE}\u{27EB}\u{2F58}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}", Some(0)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, AL, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..468, level: 1 } ], });
// assert_eq!(process_text("\u{10E67}\u{10391}\u{10E73}\u{102F8}\u{10E74}\u{FE63}\u{10E7E}\u{20AB}\u{10E64}\u{FF0F}\u{10E61}\u{0085}\u{06DD}\u{0009}\u{10E7B}\u{2007}\u{10E79}\u{21F8}\u{0605}\u{2066}\u{10E75}\u{2067}\u{10E69}\u{2068}\u{0660}\u{2069}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![AN, L, AN, EN, AN, ES, AN, ET, AN, CS, AN, B, AN, S, AN, WS, AN, ON, AN, LRI, AN, RLI, AN, FSI, AN, PDI], paragraphs: vec![ParagraphInfo { range: 0..13, level: 2 } ], });
// assert_eq!(process_text("\u{1EDA}\u{1099F}\u{115AE}\u{FB59}\u{0245}\u{207A}\u{A59D}\u{0609}\u{049B}\u{002E}\u{2CDF}\u{001E}\u{0A93}\u{000B}\u{18A4}\u{2002}\u{FA4C}\u{1F64C}\u{FF71}\u{2066}\u{1F80}\u{2067}\u{0A8F}\u{2068}\u{1D98F}\u{2069}\u{102ED}\u{1090C}\u{2080}\u{FC19}\u{2496}\u{FF0D}\u{1D7DA}\u{FE50}\u{FF11}\u{2029}\u{0035}\u{001F}\u{1D7D0}\u{2009}\u{102FB}\u{303D}\u{06F1}\u{2066}\u{1D7D9}\u{2067}\u{0033}\u{2068}\u{1D7F1}\u{2069}\u{0601}\u{1084B}\u{0669}\u{FD18}\u{10E66}\u{FF0B}\u{10E66}\u{0AF1}\u{10E6D}\u{FE50}\u{0666}\u{001D}\u{10E7D}\u{0009}\u{10E6D}\u{2004}\u{10E65}\u{241D}\u{0664}\u{2066}\u{10E6D}\u{2067}\u{066C}\u{2068}\u{10E72}\u{2069}", Some(1)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![L, R, L, AL, L, ES, L, ET, L, CS, L, B, L, S, L, WS, L, ON, L, LRI, L, RLI, L, FSI, L, PDI, EN, R, EN, AL, EN, ES, EN, CS, EN, B, EN, S, EN, WS, EN, ON, EN, LRI, EN, RLI, EN, FSI, EN, PDI, AN, R, AN, AL, AN, ES, AN, ET, AN, CS, AN, B, AN, S, AN, WS, AN, ON, AN, LRI, AN, RLI, AN, FSI, AN, PDI], paragraphs: vec![ParagraphInfo { range: 0..38, level: 2 } ], });
// assert_eq!(process_text("\u{0125}\u{12452}\u{10367}\u{2075}\u{0216}\u{0663}\u{1D01E}\u{135D}\u{1D7F8}\u{14563}\u{1D7D4}\u{102E7}\u{1D7F1}\u{20B9}\u{1F103}\u{10E72}\u{1D7D5}\u{0F77}\u{20A6}\u{1D7EE}\u{10E63}\u{339F}\u{10E64}\u{1D7DD}\u{0603}\u{10E7B}\u{10E63}\u{1DA4E}", Some(0)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7], classes: vec![L, L, L, EN, L, AN, L, NSM, EN, L, EN, EN, EN, ET, EN, AN, EN, NSM, ET, EN, AN, L, AN, EN, AN, AN, AN, NSM], paragraphs: vec![ParagraphInfo { range: 0..14, level: 2 } ], });
// assert_eq!(process_text("\u{100BE}\u{1008F}\u{1D7C0}\u{1F36}\u{0251}\u{144DC}\u{106E3}\u{16B74}\u{A337}\u{1D614}\u{14641}\u{03F8}\u{132D1}\u{1685A}\u{0569}\u{1F14}\u{0C6A}\u{14639}\u{145C9}\u{1202A}\u{0186}\u{1583}\u{168BA}\u{1D591}\u{1E07}\u{00C9}\u{FAD3}\u{1E03}\u{1D861}\u{3072}\u{14599}\u{2834}\u{0AA6}\u{1D154}\u{AA27}\u{1D85}\u{2077}\u{0035}\u{102F4}\u{2086}\u{2498}\u{2493}\u{102F8}\u{1D7CF}\u{1D7FB}\u{102E8}\u{102E9}\u{1F105}\u{1D7F6}\u{0033}\u{1D7D3}\u{06F2}\u{2089}\u{102EA}\u{1D7E7}\u{2489}\u{2494}\u{1D7DD}\u{102E2}\u{2088}\u{1D7E1}\u{00B3}\u{FF13}\u{2077}\u{06F1}\u{1D7EE}\u{1D7F5}\u{249A}\u{102EF}\u{1D7DB}\u{102F1}\u{1D7FE}\u{10E70}\u{0660}\u{0660}\u{10E76}\u{10E65}\u{10E6C}\u{10E69}\u{0664}\u{10E72}\u{10E6A}\u{10E74}\u{10E62}\u{10E7C}\u{0661}\u{10E6F}\u{10E73}\u{10E7E}\u{066B}\u{066B}\u{0660}\u{0600}\u{0666}\u{066B}\u{10E75}\u{10E75}\u{0601}\u{10E79}\u{10E68}\u{0604}\u{0665}\u{10E6F}\u{0601}\u{066B}\u{0602}\u{10E7A}\u{0603}", Some(0)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7], classes: vec![L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, EN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN, AN], paragraphs: vec![ParagraphInfo { range: 0..108, level: 2 } ], });
// assert_eq!(process_text("\u{0009}\u{0009}\u{0009}\u{001F}\u{0009}\u{000B}\u{2006}\u{3000}\u{2007}\u{2005}\u{2003}\u{2000}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{000B}\u{000B}\u{0009}\u{001F}\u{001F}\u{000B}\u{2002}\u{1680}\u{205F}\u{2028}\u{2004}\u{2008}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{0009}\u{0009}\u{001F}\u{0009}\u{001F}\u{0009}\u{0020}\u{2009}\u{3000}\u{1680}\u{2003}\u{2002}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{000B}\u{0009}\u{0009}\u{001F}\u{0009}\u{001F}\u{0020}\u{2007}\u{2003}\u{200A}\u{2005}\u{2002}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{1D197}\u{1D634}\u{10505}\u{120B1}\u{1600}\u{A5B9}\u{2087}\u{1D7DB}\u{1F10A}\u{1D7D5}\u{2079}\u{1D7DD}\u{FE62}\u{FB29}\u{208B}\u{FF0B}\u{FE63}\u{207A}\u{20B2}\u{09FB}\u{20AA}\u{20BA}\u{2033}\u{212E}\u{FF1A}\u{2044}\u{FF1A}\u{00A0}\u{060C}\u{FE50}\u{1DAAD}\u{E01B2}\u{E015E}\u{0591}\u{E01EC}\u{1085}\u{001F}\u{001F}\u{0009}\u{001F}\u{000B}\u{0009}\u{1680}\u{2028}\u{2000}\u{2004}\u{2008}\u{2009}\u{1F785}\u{2591}\u{29FC}\u{2F68}\u{2474}\u{0C79}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{1D83C}\u{3263}\u{A2F5}\u{1D6E}\u{1D368}\u{10346}\u{1D7F0}\u{1D7EE}\u{249A}\u{1D7F6}\u{1D7F0}\u{FF14}\u{208B}\u{FF0D}\u{207B}\u{002B}\u{FB29}\u{002B}\u{2213}\u{00B0}\u{0E3F}\u{FE6A}\u{20BC}\u{2034}\u{FE52}\u{003A}\u{002E}\u{FF0C}\u{FE52}\u{202F}\u{1CDF}\u{112E5}\u{1B6F}\u{10379}\u{20EF}\u{05AD}\u{001F}\u{001F}\u{000B}\u{000B}\u{000B}\u{000B}\u{2002}\u{2007}\u{2028}\u{0020}\u{2004}\u{2001}\u{1F72E}\u{02D9}\u{21FA}\u{A4B5}\u{1398}\u{2A1E}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}", Some(1)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..300, level: 0 } ], });
// assert_eq!(process_text("\u{001F}\u{0009}\u{001F}\u{000B}\u{000B}\u{0009}\u{200A}\u{205F}\u{200A}\u{2008}\u{2009}\u{2004}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{000B}\u{001F}\u{000B}\u{0009}\u{000B}\u{000B}\u{2007}\u{1680}\u{200A}\u{2000}\u{000C}\u{000C}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{07C1}\u{1E86E}\u{10A65}\u{1E844}\u{108F2}\u{1E885}\u{0683}\u{FEE9}\u{FC3A}\u{FC11}\u{1EE52}\u{06C3}\u{2212}\u{208A}\u{FE63}\u{207B}\u{FB29}\u{2212}\u{FFE0}\u{20B6}\u{20A6}\u{20AD}\u{058F}\u{20A5}\u{FE50}\u{002F}\u{2044}\u{FE55}\u{002C}\u{FE55}\u{193B}\u{A8E2}\u{05B5}\u{059E}\u{1DA64}\u{2D7F}\u{001F}\u{000B}\u{001F}\u{0009}\u{000B}\u{000B}\u{2003}\u{2008}\u{2006}\u{000C}\u{2005}\u{200A}\u{2325}\u{31D7}\u{1D20D}\u{2AD7}\u{1F5FD}\u{2EB9}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{050F}\u{118BA}\u{13262}\u{1E1C}\u{142B}\u{1D8AF}\u{0850}\u{10B8D}\u{10CEA}\u{1E824}\u{109E6}\u{10A2E}\u{FED8}\u{FB77}\u{FB8D}\u{FCAF}\u{079C}\u{FB5D}\u{2076}\u{1F102}\u{102E7}\u{1F106}\u{1D7FB}\u{FF18}\u{207B}\u{207A}\u{002D}\u{002D}\u{2212}\u{002D}\u{20BC}\u{20A7}\u{FE6A}\u{FE5F}\u{20BE}\u{FFE6}\u{0604}\u{10E6A}\u{10E64}\u{0604}\u{0600}\u{10E7E}\u{FF0F}\u{FF1A}\u{202F}\u{003A}\u{FF0C}\u{FE55}\u{0E4A}\u{11636}\u{0EB8}\u{1D168}\u{1928}\u{08E4}\u{001F}\u{001F}\u{000B}\u{000B}\u{000B}\u{001F}\u{3000}\u{2004}\u{2007}\u{2003}\u{2007}\u{000C}\u{21C4}\u{247E}\u{1F893}\u{2B7C}\u{301A}\u{2059}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{FB34}\u{10A6A}\u{10911}\u{10C41}\u{07E6}\u{080B}\u{FE96}\u{079A}\u{FDC1}\u{FCFB}\u{FB6F}\u{FBA1}\u{208A}\u{FE62}\u{FF0D}\u{2212}\u{FB29}\u{207B}\u{0024}\u{2032}\u{20BD}\u{20B5}\u{20BC}\u{2031}\u{202F}\u{00A0}\u{003A}\u{002E}\u{003A}\u{FF1A}\u{1DD6}\u{0307}\u{E01DC}\u{115B3}\u{1CE7}\u{16F91}\u{001F}\u{001F}\u{0009}\u{000B}\u{0009}\u{001F}\u{2001}\u{000C}\u{1680}\u{000C}\u{3000}\u{2009}\u{1F6EC}\u{1F6E4}\u{298B}\u{1F67B}\u{2AAE}\u{1F512}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{10B04}\u{1E817}\u{1099C}\u{10A1D}\u{10838}\u{1098F}\u{0712}\u{06C0}\u{1EE01}\u{FD75}\u{0682}\u{FC21}\u{FE63}\u{002D}\u{FB29}\u{FB29}\u{FB29}\u{002B}\u{FE6A}\u{20A2}\u{2030}\u{20BC}\u{20A4}\u{20A3}\u{FE52}\u{002C}\u{003A}\u{002F}\u{002F}\u{FF0E}\u{073D}\u{1A66}\u{0D42}\u{A802}\u{103A}\u{E012F}\u{0009}\u{001F}\u{0009}\u{001F}\u{001F}\u{001F}\u{2007}\u{2001}\u{2009}\u{2000}\u{2005}\u{2028}\u{1F54F}\u{1F0D1}\u{2AA2}\u{1F809}\u{261C}\u{1F493}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2066}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2067}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2068}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}\u{2069}", Some(1)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, L, L, L, L, L, L, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, AN, AN, AN, AN, AN, AN, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, S, S, S, S, S, S, WS, WS, WS, WS, WS, WS, ON, ON, ON, ON, ON, ON, LRI, LRI, LRI, LRI, LRI, LRI, RLI, RLI, RLI, RLI, RLI, RLI, FSI, FSI, FSI, FSI, FSI, FSI, PDI, PDI, PDI, PDI, PDI, PDI], paragraphs: vec![ParagraphInfo { range: 0..402, level: 1 } ], });
// assert_eq!(process_text("\u{1E11}\u{1B88}\u{10767}\u{13002}\u{1D047}\u{1219}\u{06F7}\u{00B9}\u{1F100}\u{102ED}\u{102EF}\u{1D7F0}\u{002D}\u{FB29}\u{FE62}\u{FB29}\u{208B}\u{208A}\u{20B7}\u{20B4}\u{20A2}\u{2030}\u{20B6}\u{20A2}\u{FF0F}\u{FF0C}\u{FE52}\u{202F}\u{002F}\u{FE52}\u{0EC9}\u{10A02}\u{10377}\u{06E7}\u{05AE}\u{1BF1}\u{4DCC}\u{2EAF}\u{2790}\u{2569}\u{2E05}\u{A4B2}\u{1447F}\u{0D33}\u{1F133}\u{0978}\u{A994}\u{144DA}\u{10907}\u{10A40}\u{1092D}\u{10813}\u{FB26}\u{10CF0}\u{1EE8C}\u{FC44}\u{FB58}\u{FE73}\u{1EEB5}\u{FB7B}\u{0032}\u{0030}\u{1D7D0}\u{248A}\u{2087}\u{1D7F4}\u{FF0B}\u{207A}\u{FE62}\u{207A}\u{2212}\u{2212}\u{20BC}\u{20BC}\u{058F}\u{00B1}\u{FF04}\u{20B1}\u{10E67}\u{10E6F}\u{0668}\u{0663}\u{066B}\u{10E6C}\u{FF1A}\u{2044}\u{FE50}\u{003A}\u{FF1A}\u{002C}\u{1C30}\u{11041}\u{0D43}\u{E01BA}\u{031A}\u{E01BA}\u{1F805}\u{1F702}\u{1F44B}\u{1017A}\u{1F59A}\u{2961}\u{1E10}\u{16923}\u{1528}\u{1D79E}\u{1F50}\u{12444}\u{1D7FE}\u{102E8}\u{FF16}\u{1F103}\u{1D7E8}\u{1D7DF}\u{10E73}\u{0661}\u{10E67}\u{10E61}\u{10E64}\u{0661}\u{2F901}\u{114E}\u{185B}\u{1D913}\u{1E5A}\u{1D814}\u{1D7E3}\u{1D7DA}\u{102E6}\u{2499}\u{1D7FB}\u{2078}\u{10E6C}\u{0664}\u{0660}\u{10E65}\u{10E7A}\u{10E7A}\u{13031}\u{12271}\u{19A9}\u{1D5BA}\u{10547}\u{0B60}\u{1D7EC}\u{249B}\u{06F2}\u{1D7EE}\u{1D7E2}\u{1D7FB}\u{0663}\u{10E60}\u{10E63}\u{10E6C}\u{10E6C}\u{10E74}", Some(1)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  7,  7,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  7,  7,  7,  7], classes: vec![L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, ON, ON, ON, ON, ON, ON, L, L, L, L, L, L, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, AN, AN, AN, AN, AN, AN, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, ON, ON, ON, ON, ON, ON, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, AN, AN, AN, AN, AN, AN, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, AN, AN, AN, AN, AN, AN, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, AN, AN, AN, AN, AN, AN], paragraphs: vec![ParagraphInfo { range: 0..156, level: 2 } ], });
// assert_eq!(process_text("\u{10CA5}\u{1E875}\u{10B6F}\u{10900}\u{10A71}\u{10B78}\u{FD27}\u{FD17}\u{0782}\u{FC99}\u{FBD7}\u{FC9C}\u{10A23}\u{1092B}\u{10AF1}\u{1E852}\u{10900}\u{10A19}\u{072A}\u{FCFE}\u{1EE35}\u{FB61}\u{FD6D}\u{FD15}\u{FE63}\u{2212}\u{208A}\u{FE62}\u{002D}\u{002B}\u{FF04}\u{20AB}\u{20B9}\u{20BB}\u{20B5}\u{20B7}\u{FF0F}\u{2044}\u{060C}\u{FF0C}\u{00A0}\u{FF0E}\u{111BA}\u{1DF1}\u{0330}\u{E0136}\u{A8E6}\u{09E3}\u{1F3C1}\u{22E5}\u{251B}\u{1F7CF}\u{22DB}\u{1F57B}\u{1E67}\u{1457E}\u{124F6}\u{03EF}\u{0D7E}\u{1D4A}\u{10C11}\u{10ADE}\u{0828}\u{1099B}\u{10A56}\u{1E84D}\u{FBBF}\u{FBBA}\u{FE9A}\u{FCC9}\u{FD26}\u{0727}\u{248B}\u{1D7FF}\u{00B3}\u{1D7ED}\u{1D7E8}\u{102E6}\u{FF0B}\u{FE63}\u{FE62}\u{FE63}\u{208A}\u{FE63}\u{20B3}\u{00A5}\u{20A8}\u{20B6}\u{20B0}\u{060A}\u{10E74}\u{10E6B}\u{0605}\u{10E68}\u{10E6D}\u{10E6E}\u{FE55}\u{060C}\u{FF0C}\u{060C}\u{FE55}\u{FF0F}\u{11132}\u{2DE9}\u{2DFA}\u{2DF2}\u{033C}\u{1136A}\u{1F647}\u{1F3AF}\u{2954}\u{2F67}\u{224A}\u{1F783}", Some(1)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, ON, ON, ON, ON, ON, ON, L, L, L, L, L, L, R, R, R, R, R, R, AL, AL, AL, AL, AL, AL, EN, EN, EN, EN, EN, EN, ES, ES, ES, ES, ES, ES, ET, ET, ET, ET, ET, ET, AN, AN, AN, AN, AN, AN, CS, CS, CS, CS, CS, CS, NSM, NSM, NSM, NSM, NSM, NSM, ON, ON, ON, ON, ON, ON], paragraphs: vec![ParagraphInfo { range: 0..114, level: 3 } ], });
// assert_eq!(process_text("\u{0665}\u{0669}\u{10E6F}\u{0604}\u{06DD}\u{10E69}\u{1D687}\u{13026}\u{144A1}\u{10721}\u{2D61}\u{D7B5}\u{1D7EE}\u{102F0}\u{2077}\u{1D7EF}\u{102EE}\u{FF14}\u{10E77}\u{10E63}\u{0666}\u{10E66}\u{10E6C}\u{10E65}", Some(1)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![AN, AN, AN, AN, AN, AN, L, L, L, L, L, L, EN, EN, EN, EN, EN, EN, AN, AN, AN, AN, AN, AN], paragraphs: vec![ParagraphInfo { range: 0..24, level: 4 } ], });
// assert_eq!(process_text("\u{000D}\u{0009}\u{2000}\u{2066}\u{2067}\u{2068}\u{2069}\u{000A}\u{001F}\u{3000}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{001F}\u{2002}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{001F}\u{2005}\u{2066}\u{2067}\u{2068}\u{2069}\u{A8A2}\u{1D7E2}\u{2212}\u{20A4}\u{002C}\u{0361}\u{001D}\u{001F}\u{2007}\u{2693}\u{2066}\u{2067}\u{2068}\u{2069}\u{0085}\u{001F}\u{2005}\u{2066}\u{2067}\u{2068}\u{2069}\u{000A}\u{0009}\u{3000}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{0009}\u{2005}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{000B}\u{2007}\u{2066}\u{2067}\u{2068}\u{2069}\u{000D}\u{001F}\u{1680}\u{2066}\u{2067}\u{2068}\u{2069}\u{1337A}\u{06F5}\u{208B}\u{20A8}\u{060C}\u{1DA0C}\u{0085}\u{001F}\u{2008}\u{1F6C1}\u{2066}\u{2067}\u{2068}\u{2069}\u{2029}\u{001F}\u{000C}\u{2066}\u{2067}\u{2068}\u{2069}\u{2029}\u{000B}\u{2009}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{0009}\u{2009}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{001F}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{0085}\u{001F}\u{200A}\u{2066}\u{2067}\u{2068}\u{2069}\u{16B55}\u{102F1}\u{FE62}\u{09F2}\u{060C}\u{1B6E}\u{0085}\u{000B}\u{2002}\u{1F721}\u{2066}\u{2067}\u{2068}\u{2069}\u{000A}\u{001F}\u{1680}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{001F}\u{200A}\u{2066}\u{2067}\u{2068}\u{2069}\u{0085}\u{001F}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{001F}\u{2007}\u{2066}\u{2067}\u{2068}\u{2069}\u{0085}\u{000B}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{16A33}\u{2084}\u{208A}\u{20AC}\u{2044}\u{0F7B}\u{001D}\u{001F}\u{2006}\u{1F0B8}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{0009}\u{2007}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{0009}\u{2009}\u{2066}\u{2067}\u{2068}\u{2069}\u{000D}\u{000B}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{0009}\u{205F}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{000B}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{106D4}\u{1D7DC}\u{FF0D}\u{A839}\u{002E}\u{AA35}\u{000D}\u{0009}\u{2004}\u{2A3C}\u{2066}\u{2067}\u{2068}\u{2069}\u{2C13}\u{1D7F8}\u{FE62}\u{09F3}\u{FE55}\u{033D}\u{0085}\u{001F}\u{2008}\u{2B59}\u{2066}\u{2067}\u{2068}\u{2069}\u{2029}\u{0009}\u{205F}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{000B}\u{2001}\u{2066}\u{2067}\u{2068}\u{2069}\u{2029}\u{000B}\u{2005}\u{2066}\u{2067}\u{2068}\u{2069}\u{000A}\u{0009}\u{200A}\u{2066}\u{2067}\u{2068}\u{2069}\u{16F73}\u{FF12}\u{208A}\u{00B1}\u{2044}\u{1DA42}\u{2029}\u{000B}\u{2000}\u{FE47}\u{2066}\u{2067}\u{2068}\u{2069}\u{16954}\u{2076}\u{207A}\u{20B5}\u{FE52}\u{17CB}\u{000D}\u{0009}\u{2004}\u{2108}\u{2066}\u{2067}\u{2068}\u{2069}", Some(2)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, L, EN, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI], paragraphs: vec![ParagraphInfo { range: 0..308, level: 0 } ], });
// assert_eq!(process_text("\u{001E}\u{001F}\u{2000}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{0009}\u{2006}\u{2066}\u{2067}\u{2068}\u{2069}\u{000A}\u{000B}\u{3000}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{001F}\u{2028}\u{2066}\u{2067}\u{2068}\u{2069}\u{10B51}\u{FEDD}\u{002D}\u{FFE0}\u{003A}\u{E0157}\u{000A}\u{000B}\u{2005}\u{1F734}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{0009}\u{3000}\u{2066}\u{2067}\u{2068}\u{2069}\u{000D}\u{000B}\u{2028}\u{2066}\u{2067}\u{2068}\u{2069}\u{000D}\u{000B}\u{1680}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{000B}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{001F}\u{205F}\u{2066}\u{2067}\u{2068}\u{2069}\u{10AD9}\u{FC0A}\u{2212}\u{060A}\u{002E}\u{E0142}\u{000D}\u{001F}\u{1680}\u{10B3B}\u{2066}\u{2067}\u{2068}\u{2069}\u{2029}\u{001F}\u{2009}\u{2066}\u{2067}\u{2068}\u{2069}\u{2029}\u{0009}\u{2001}\u{2066}\u{2067}\u{2068}\u{2069}\u{000A}\u{001F}\u{0020}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{001F}\u{2005}\u{2066}\u{2067}\u{2068}\u{2069}\u{001D}\u{000B}\u{2004}\u{2066}\u{2067}\u{2068}\u{2069}\u{1E82C}\u{0780}\u{FE63}\u{20B0}\u{FE50}\u{0F95}\u{000A}\u{001F}\u{2008}\u{2AF2}\u{2066}\u{2067}\u{2068}\u{2069}\u{07C2}\u{FBBE}\u{FB29}\u{FFE1}\u{002E}\u{1DA10}\u{001D}\u{000B}\u{0020}\u{2776}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{001F}\u{2008}\u{2066}\u{2067}\u{2068}\u{2069}\u{0085}\u{000B}\u{2005}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{0009}\u{205F}\u{2066}\u{2067}\u{2068}\u{2069}\u{001E}\u{0009}\u{1680}\u{2066}\u{2067}\u{2068}\u{2069}\u{10C0F}\u{FED6}\u{FE63}\u{2034}\u{202F}\u{0AC8}\u{0085}\u{000B}\u{0020}\u{296E}\u{2066}\u{2067}\u{2068}\u{2069}\u{10004}\u{10A31}\u{06EF}\u{1D7E3}\u{207A}\u{066A}\u{10E7C}\u{003A}\u{1A18}\u{000A}\u{000B}\u{2028}\u{25F9}\u{2066}\u{2067}\u{2068}\u{2069}\u{000A}\u{000B}\u{2002}\u{2066}\u{2067}\u{2068}\u{2069}\u{001C}\u{0009}\u{0020}\u{2066}\u{2067}\u{2068}\u{2069}\u{10B7C}\u{FCCE}\u{208A}\u{20B2}\u{002C}\u{11230}\u{001D}\u{000B}\u{2001}\u{23BD}\u{2066}\u{2067}\u{2068}\u{2069}\u{1D03}\u{109A1}\u{FCFA}\u{2089}\u{FF0D}\u{00B0}\u{10E6C}\u{FE52}\u{0FB8}\u{2029}\u{001F}\u{3000}\u{1F74B}\u{2066}\u{2067}\u{2068}\u{2069}\u{10ADF}\u{06B7}\u{FF0B}\u{2034}\u{002F}\u{036D}\u{0085}\u{001F}\u{2002}\u{1F53F}\u{2066}\u{2067}\u{2068}\u{2069}\u{0834}\u{FD26}\u{208A}\u{20B1}\u{002F}\u{AAF6}\u{0085}\u{001F}\u{200A}\u{2054}\u{2066}\u{2067}\u{2068}\u{2069}\u{000D}\u{001F}\u{2005}\u{2066}\u{2067}\u{2068}\u{2069}\u{0085}\u{000B}\u{2028}\u{2066}\u{2067}\u{2068}\u{2069}\u{07C1}\u{FBAB}\u{207A}\u{20A8}\u{060C}\u{1DD9}\u{001D}\u{001F}\u{3000}\u{1F65D}\u{2066}\u{2067}\u{2068}\u{2069}\u{1FA2}\u{109C6}\u{1EE93}\u{1D7EB}\u{FF0D}\u{20BB}\u{0601}\u{FF0C}\u{FB1E}\u{001E}\u{001F}\u{200A}\u{1FCD}\u{2066}\u{2067}\u{2068}\u{2069}\u{109DC}\u{FCBC}\u{207A}\u{20BB}\u{FF0E}\u{1DA4E}\u{001D}\u{000B}\u{2001}\u{2284}\u{2066}\u{2067}\u{2068}\u{2069}\u{10B4B}\u{FCF3}\u{FF0B}\u{2033}\u{FF1A}\u{0FA4}\u{001E}\u{0009}\u{1680}\u{2F48}\u{2066}\u{2067}\u{2068}\u{2069}", Some(2)), BidiInfo { levels: vec![ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  3,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  3,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  2,  2,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  3,  2,  2,  3,  3,  3,  3,  3,  3,  4,  4,  4,  3,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  7,  7,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, L, R, AL, EN, ES, ET, AN, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, L, R, AL, EN, ES, ET, AN, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, B, S, WS, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, L, R, AL, EN, ES, ET, AN, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI, R, AL, ES, ET, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI], paragraphs: vec![ParagraphInfo { range: 0..359, level: 1 } ], });
// assert_eq!(process_text("\u{AA33}\u{1F101}\u{10E69}\u{11082}\u{FF17}\u{002D}\u{00A4}\u{202F}\u{0ECD}\u{1F600}\u{1036F}\u{1D7DA}\u{10E70}\u{330E}\u{10B9C}\u{0623}\u{102F4}\u{FE63}\u{0024}\u{10E7C}\u{060C}\u{1DA28}\u{31CD}\u{17DC}\u{FF17}\u{002B}\u{17DB}\u{002C}\u{1DA75}\u{1F05D}\u{132E8}\u{1098D}\u{FDA4}\u{248A}\u{002D}\u{20B9}\u{066C}\u{003A}\u{1DA28}\u{2116}\u{101F5}\u{2488}\u{10E78}\u{116A2}\u{1D7CF}\u{10E71}\u{3243}\u{2497}\u{002B}\u{20B1}\u{FE52}\u{E016E}\u{1F744}\u{10388}\u{109C4}\u{FDA3}\u{1D7ED}\u{FB29}\u{A838}\u{10E74}\u{FF1A}\u{AAB7}\u{1F4AD}\u{A5C9}\u{FF14}\u{0662}\u{115A5}\u{102EE}\u{FF0B}\u{2213}\u{FF0C}\u{1A5A}\u{1F70A}\u{17A1}\u{109CB}\u{068F}\u{06F1}\u{207A}\u{FFE6}\u{10E7D}\u{002F}\u{1C2F}\u{1F6CD}\u{2DB0}\u{2085}\u{10E7E}\u{1D090}\u{102F4}\u{0663}\u{F931}\u{102E9}\u{10E6A}\u{10023}\u{FF19}\u{207B}\u{060A}\u{002F}\u{0FAA}\u{26AA}\u{2F8CF}\u{083B}\u{077E}\u{2490}\u{FB29}\u{20AE}\u{10E79}\u{FE55}\u{E0196}\u{1F0EA}\u{2F8B3}\u{102EA}\u{10E71}\u{1229E}\u{2082}\u{066C}\u{0BEA}\u{0035}\u{10E6D}", Some(2)), BidiInfo { levels: vec![ 4,  4,  7,  7,  7,  7,  7,  7,  7,  7,  4,  4,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  7,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  4,  4,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  3,  3,  3,  4,  4,  7,  4,  4,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  3,  3,  3,  4,  4,  7,  4,  4,  7], classes: vec![L, EN, AN, L, EN, ES, ET, CS, NSM, ON, L, EN, AN, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, L, EN, AN, L, EN, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, L, EN, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, L, EN, AN, L, EN, AN, L, EN, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, L, EN, AN, L, EN, AN], paragraphs: vec![ParagraphInfo { range: 0..118, level: 2 } ], });
// assert_eq!(process_text("\u{10983}\u{075C}\u{002B}\u{20B7}\u{FF0E}\u{10A0D}\u{2ACB}\u{2C6E}\u{1E878}\u{FEA8}\u{102EB}\u{FE62}\u{20A4}\u{10E6A}\u{FF0C}\u{1DC1}\u{1F38A}\u{10BAE}\u{FCC4}\u{10CDD}\u{FBED}\u{FB29}\u{20A4}\u{FE55}\u{E0125}\u{2597}\u{A5D6}\u{FB32}\u{0707}\u{102ED}\u{002D}\u{20B4}\u{10E68}\u{003A}\u{E01CA}\u{2F8B}\u{10908}\u{FC21}\u{10894}\u{FE93}\u{FF0B}\u{FE69}\u{002C}\u{0610}\u{1F6BE}\u{A556}\u{0844}\u{06D0}\u{2084}\u{207B}\u{FE6A}\u{0600}\u{FF1A}\u{1060}\u{1F68F}\u{FB2F}\u{069D}\u{207A}\u{FF03}\u{FE52}\u{1D168}\u{2B14}\u{084E}\u{FB83}\u{10B24}\u{FC83}\u{207B}\u{20B4}\u{FF0E}\u{1DAA8}\u{1D222}\u{1972}\u{10863}\u{1EE4E}\u{1D7E0}\u{002B}\u{066A}\u{10E66}\u{002F}\u{E0121}\u{0BF5}\u{13372}\u{10ACD}\u{0693}\u{248D}\u{FE62}\u{20A9}\u{10E65}\u{2044}\u{1A62}\u{2268}\u{109E0}\u{FEBC}\u{10929}\u{06A3}\u{FF0D}\u{060A}\u{FF0E}\u{1073}\u{1F73D}\u{102C7}\u{10B84}\u{FD55}\u{1D7EB}\u{FB29}\u{2033}\u{0602}\u{FF0E}\u{E01E8}\u{31DB}\u{10A54}\u{06A7}\u{10AC1}\u{FCF9}\u{FE62}\u{00A5}\u{060C}\u{E01ED}\u{1F316}\u{122FE}\u{10C99}\u{0688}\u{1F104}\u{FF0D}\u{20B2}\u{0667}\u{FF1A}\u{1B81}\u{29BF}", Some(2)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  5,  5,  4,  4,  4,  4,  4,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  7,  7,  5,  5,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  7,  7,  5,  5,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, R, AL, ES, ET, CS, NSM, ON, R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON], paragraphs: vec![ParagraphInfo { range: 0..129, level: 3 } ], });
// assert_eq!(process_text("\u{24A0}\u{102F9}\u{002D}\u{20B6}\u{002C}\u{1DA23}\u{2314}\u{0C5A}\u{05E8}\u{FCD4}\u{1D7D8}\u{002D}\u{FFE0}\u{10E6E}\u{FF0C}\u{A9BC}\u{26E0}\u{132C3}\u{06F7}\u{10E68}\u{10E74}\u{A1C4}\u{1D7FD}\u{FE62}\u{20A9}\u{FE52}\u{031E}\u{2AAD}\u{1116}\u{109C1}\u{1EE86}\u{1F108}\u{FF0D}\u{FE6A}\u{10E76}\u{00A0}\u{1A6C}\u{1F33B}\u{1649}\u{102E3}\u{0601}\u{A46A}\u{1D7F3}\u{FF0D}\u{20A3}\u{0665}\u{202F}\u{06DC}\u{24F0}\u{146D}\u{1E80E}\u{FE76}\u{1D7F8}\u{2212}\u{20B4}\u{10E6E}\u{FF1A}\u{20DA}\u{2223}\u{A079}\u{1D7D0}\u{10E61}\u{1D8BF}\u{2080}\u{10E78}\u{09D7}\u{102E4}\u{207B}\u{2032}\u{10E66}\u{00A0}\u{11134}\u{260B}\u{0E19}\u{07CE}\u{071D}\u{102E9}\u{207B}\u{20B9}\u{10E70}\u{FE50}\u{1B6D}\u{22A9}\u{3354}\u{1D7EE}\u{0605}\u{0669}\u{2F9E3}\u{2080}\u{10E71}\u{0600}\u{1C4D}\u{102FB}\u{0666}", Some(2)), BidiInfo { levels: vec![ 7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  4,  4,  4,  4,  3,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  3,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  3,  3,  3,  7,  4,  4,  4,  7,  4,  4,  4], classes: vec![L, EN, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, AN, L, EN, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, L, EN, ES, ET, AN, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, L, EN, AN, L, EN, ES, ET, AN, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, L, EN, AN, AN, L, EN, AN, AN, L, EN, AN], paragraphs: vec![ParagraphInfo { range: 0..94, level: 4 } ], });
// assert_eq!(process_text("\u{10A56}\u{0719}\u{0858}\u{FE92}\u{10CE2}\u{FC1A}\u{10C43}\u{FC45}\u{FF0D}\u{20B3}\u{FE50}\u{07AF}\u{2923}\u{12428}\u{10CC4}\u{1EE81}\u{0030}\u{FF0B}\u{20A2}\u{10E64}\u{002F}\u{1DC7}\u{2601}\u{10A56}\u{075C}\u{1E842}\u{FBBE}\u{207A}\u{20B5}\u{002C}\u{111B6}\u{1F5B0}\u{2DC5}\u{10AF0}\u{FDC3}\u{1D7FF}\u{FE62}\u{0024}\u{0661}\u{FF0E}\u{0745}\u{2A11}", Some(2)), BidiInfo { levels: vec![ 7,  7,  7,  7,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  4,  4,  4,  4,  4,  4,  5,  5,  4,  4,  4,  4,  4,  4,  4], classes: vec![R, AL, R, AL, R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON, R, AL, R, AL, ES, ET, CS, NSM, ON, L, R, AL, EN, ES, ET, AN, CS, NSM, ON], paragraphs: vec![ParagraphInfo { range: 0..42, level: 5 } ], });
// assert_eq!(process_text("\u{10E7C}\u{06DD}\u{066B}\u{13179}\u{1F100}\u{10E73}\u{0604}\u{A10E}\u{FF12}\u{0666}", Some(2)), BidiInfo { levels: vec![ 7,  7,  4,  4,  4,  4,  4,  4,  4,  4], classes: vec![AN, AN, AN, L, EN, AN, AN, L, EN, AN], paragraphs: vec![ParagraphInfo { range: 0..10, level: 6 } ], });
// assert_eq!(process_text("\u{A603}\u{1E809}\u{2C03}\u{1088F}\u{0EA7}\u{10A8B}\u{1D99D}\u{1E84C}\u{13203}\u{05D3}\u{13C4}\u{10A9A}\u{1D610}\u{FC5E}\u{FF8F}\u{FD3A}\u{14504}\u{FCB5}\u{1113C}\u{0798}\u{1F1E7}\u{FD6C}\u{1F11}\u{FE88}\u{1D7DD}\u{07E7}\u{102F7}\u{10A7D}\u{249A}\u{FB26}\u{1D7DF}\u{10A6B}\u{1F108}\u{10C26}\u{102ED}\u{1E81E}\u{1D7DD}\u{FD00}\u{1D7FE}\u{1EEBA}\u{1D7CE}\u{FC6B}\u{1D7FA}\u{1EE0D}\u{1F108}\u{1EE5B}\u{1D7E1}\u{0785}\u{002B}\u{1E8AA}\u{208A}\u{10B5D}\u{2212}\u{10896}\u{207A}\u{10B0C}\u{FF0D}\u{1E8BB}\u{FF0D}\u{1E878}\u{FF0D}\u{0723}\u{002B}\u{FB94}\u{002D}\u{1EE5F}\u{FF0D}\u{FCA2}\u{FF0B}\u{FC0B}\u{208B}\u{077D}\u{20A3}\u{109E8}\u{0023}\u{109BE}\u{20B5}\u{0809}\u{2213}\u{1E898}\u{0609}\u{083A}\u{0024}\u{1081A}\u{212E}\u{FB79}\u{0E3F}\u{FEF2}\u{2033}\u{1EE12}\u{20AA}\u{FC5E}\u{A839}\u{070C}\u{060A}\u{0631}\u{FF1A}\u{109D4}\u{FF0C}\u{0835}\u{FF1A}\u{108ED}\u{002F}\u{10C45}\u{002E}\u{10902}\u{002E}\u{1E839}\u{060C}\u{FCEB}\u{202F}\u{FB86}\u{002E}\u{FBD4}\u{002C}\u{067F}\u{FE55}\u{0688}\u{FF0E}\u{FBF6}\u{102F}\u{10AC6}\u{08FC}\u{10B1A}\u{08EE}\u{10874}\u{033A}\u{10C28}\u{1DA59}\u{1E8BD}\u{2DEF}\u{1E8AA}\u{116B5}\u{FD26}\u{E01D1}\u{FD65}\u{E01A4}\u{1EE00}\u{E017A}\u{076E}\u{1DFE}\u{1EE77}\u{2DE1}\u{FB5C}\u{0009}\u{1E87C}\u{000B}\u{109C2}\u{0009}\u{10A23}\u{000B}\u{0839}\u{001F}\u{1082C}\u{000B}\u{10C33}\u{0009}\u{FCC3}\u{001F}\u{0776}\u{0009}\u{FC7F}\u{001F}\u{076D}\u{0009}\u{0788}\u{000B}\u{FC0B}\u{2000}\u{10913}\u{000C}\u{07C6}\u{2003}\u{108FF}\u{1680}\u{10AEF}\u{2004}\u{1E821}\u{0020}\u{05DF}\u{200A}\u{FE97}\u{200A}\u{0716}\u{2006}\u{FBB2}\u{2001}\u{FCEC}\u{200A}\u{FEE5}\u{2003}\u{FE9D}\u{1F668}\u{10A65}\u{1F3F9}\u{10A50}\u{27C0}\u{1E848}\u{A4C3}\u{10890}\u{1F583}\u{10A6B}\u{2E1A}\u{FB27}\u{1F057}\u{062A}\u{1F402}\u{FC49}\u{2396}\u{063E}\u{22B3}\u{FD14}\u{1F619}\u{FD04}\u{2977}\u{063E}\u{2067}\u{10898}\u{2067}\u{10909}\u{2067}\u{1099D}\u{2067}\u{10C24}\u{2067}\u{10A9F}\u{2067}\u{FB36}\u{2067}\u{FDB1}\u{2067}\u{1EEB7}\u{2067}\u{FC53}\u{2067}\u{FDF7}\u{2067}\u{06C9}\u{2067}\u{FB9E}\u{2067}\u{2212}\u{2067}\u{FF0D}\u{2067}\u{208A}\u{2067}\u{207B}\u{2067}\u{002D}\u{2067}\u{002B}\u{2067}\u{FF03}\u{2067}\u{060A}\u{2067}\u{FFE5}\u{2067}\u{20A2}\u{2067}\u{20AA}\u{2067}\u{20B0}\u{2067}\u{202F}\u{2067}\u{202F}\u{2067}\u{2044}\u{2067}\u{003A}\u{2067}\u{060C}\u{2067}\u{FE52}\u{2067}\u{1DAAB}\u{2067}\u{1DA4B}\u{2067}\u{0E4D}\u{2067}\u{0E4E}\u{2067}\u{A677}\u{2067}\u{1112D}\u{2067}\u{2FCB}\u{2067}\u{1F918}\u{2067}\u{0C7A}\u{2067}\u{1F07A}\u{2067}\u{1D216}\u{2067}\u{1F4BB}\u{2068}\u{10CB1}\u{2068}\u{109DB}\u{2068}\u{10C26}\u{2068}\u{05F0}\u{2068}\u{10B5D}\u{2068}\u{1089D}\u{2068}\u{FC0A}\u{2068}\u{FC7B}\u{2068}\u{FBF2}\u{2068}\u{FBE1}\u{2068}\u{FCFE}\u{2068}\u{06C8}\u{2069}\u{05E8}\u{2069}\u{10A57}\u{2069}\u{10B23}\u{2069}\u{1E8A9}\u{2069}\u{10884}\u{2069}\u{FB2A}\u{2069}\u{FB6A}\u{2069}\u{FD6F}\u{2069}\u{FD8D}\u{2069}\u{FEE4}\u{2069}\u{077E}\u{2069}\u{1EE09}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2], classes: vec![L, R, L, R, L, R, L, R, L, R, L, R, L, AL, L, AL, L, AL, L, AL, L, AL, L, AL, EN, R, EN, R, EN, R, EN, R, EN, R, EN, R, EN, AL, EN, AL, EN, AL, EN, AL, EN, AL, EN, AL, ES, R, ES, R, ES, R, ES, R, ES, R, ES, R, ES, AL, ES, AL, ES, AL, ES, AL, ES, AL, ES, AL, ET, R, ET, R, ET, R, ET, R, ET, R, ET, R, ET, AL, ET, AL, ET, AL, ET, AL, ET, AL, ET, AL, CS, R, CS, R, CS, R, CS, R, CS, R, CS, R, CS, AL, CS, AL, CS, AL, CS, AL, CS, AL, CS, AL, NSM, R, NSM, R, NSM, R, NSM, R, NSM, R, NSM, R, NSM, AL, NSM, AL, NSM, AL, NSM, AL, NSM, AL, NSM, AL, S, R, S, R, S, R, S, R, S, R, S, R, S, AL, S, AL, S, AL, S, AL, S, AL, S, AL, WS, R, WS, R, WS, R, WS, R, WS, R, WS, R, WS, AL, WS, AL, WS, AL, WS, AL, WS, AL, WS, AL, ON, R, ON, R, ON, R, ON, R, ON, R, ON, R, ON, AL, ON, AL, ON, AL, ON, AL, ON, AL, ON, AL, RLI, R, RLI, R, RLI, R, RLI, R, RLI, R, RLI, R, RLI, AL, RLI, AL, RLI, AL, RLI, AL, RLI, AL, RLI, AL, RLI, ES, RLI, ES, RLI, ES, RLI, ES, RLI, ES, RLI, ES, RLI, ET, RLI, ET, RLI, ET, RLI, ET, RLI, ET, RLI, ET, RLI, CS, RLI, CS, RLI, CS, RLI, CS, RLI, CS, RLI, CS, RLI, NSM, RLI, NSM, RLI, NSM, RLI, NSM, RLI, NSM, RLI, NSM, RLI, ON, RLI, ON, RLI, ON, RLI, ON, RLI, ON, RLI, ON, FSI, R, FSI, R, FSI, R, FSI, R, FSI, R, FSI, R, FSI, AL, FSI, AL, FSI, AL, FSI, AL, FSI, AL, FSI, AL, PDI, R, PDI, R, PDI, R, PDI, R, PDI, R, PDI, R, PDI, AL, PDI, AL, PDI, AL, PDI, AL, PDI, AL, PDI, AL], paragraphs: vec![ParagraphInfo { range: 0..174, level: 0 } ], });
// assert_eq!(process_text("\u{092C}\u{10E65}\u{3216}\u{10E70}\u{124AE}\u{0662}\u{18F4}\u{10E77}\u{A6E2}\u{0665}\u{118F}\u{10E61}\u{1D7D7}\u{0600}\u{2078}\u{10E76}\u{1D7FC}\u{10E6E}\u{1D7E0}\u{10E6F}\u{06F5}\u{10E7A}\u{102FB}\u{0605}\u{207B}\u{10E6A}\u{002B}\u{0604}\u{002D}\u{10E7D}\u{FB29}\u{10E78}\u{207B}\u{10E7A}\u{FF0B}\u{0664}\u{20A0}\u{10E74}\u{20A3}\u{10E7D}\u{20B7}\u{10E7E}\u{060A}\u{0664}\u{0609}\u{0662}\u{2033}\u{10E7A}\u{FE50}\u{0663}\u{060C}\u{10E72}\u{060C}\u{0666}\u{FE50}\u{10E78}\u{003A}\u{10E7D}\u{FF0C}\u{10E62}\u{114B3}\u{10E7E}\u{0F7E}\u{0662}\u{11368}\u{0663}\u{0A4C}\u{10E72}\u{1921}\u{10E7E}\u{1CD0}\u{10E66}\u{000B}\u{10E70}\u{0009}\u{10E7B}\u{0009}\u{0605}\u{001F}\u{0667}\u{000B}\u{0600}\u{0009}\u{10E6F}\u{2008}\u{0661}\u{3000}\u{10E7A}\u{2002}\u{066B}\u{2005}\u{0661}\u{2008}\u{10E6C}\u{2028}\u{0663}\u{1F481}\u{10E6D}\u{1F559}\u{0666}\u{1392}\u{10E68}\u{1F6D0}\u{10E63}\u{2B70}\u{10E69}\u{31DC}\u{0665}\u{2066}\u{3319}\u{2066}\u{1D4D5}\u{2066}\u{FA14}\u{2066}\u{1F15F}\u{2066}\u{1D569}\u{2066}\u{050A}\u{2066}\u{1F108}\u{2066}\u{1D7E0}\u{2066}\u{102FA}\u{2066}\u{FF14}\u{2066}\u{102F7}\u{2066}\u{102FB}\u{2066}\u{207A}\u{2066}\u{002D}\u{2066}\u{FB29}\u{2066}\u{208A}\u{2066}\u{208A}\u{2066}\u{208A}\u{2066}\u{FE69}\u{2066}\u{FF04}\u{2066}\u{20BD}\u{2066}\u{FFE0}\u{2066}\u{20A9}\u{2066}\u{2213}\u{2066}\u{FF0F}\u{2066}\u{00A0}\u{2066}\u{FF0C}\u{2066}\u{FF1A}\u{2066}\u{202F}\u{2066}\u{002F}\u{2066}\u{0981}\u{2066}\u{E010A}\u{2066}\u{0652}\u{2066}\u{20D3}\u{2066}\u{17BD}\u{2066}\u{E0173}\u{2066}\u{1F3FF}\u{2066}\u{1F3B2}\u{2066}\u{2AA6}\u{2066}\u{1D229}\u{2066}\u{1F4A0}\u{2066}\u{22F5}\u{2067}\u{1D411}\u{2067}\u{10139}\u{2067}\u{1316C}\u{2067}\u{1131D}\u{2067}\u{10356}\u{2067}\u{1339B}\u{2067}\u{2080}\u{2067}\u{2497}\u{2067}\u{1D7D1}\u{2067}\u{1F106}\u{2067}\u{102F6}\u{2067}\u{102E6}\u{2067}\u{10E6A}\u{2067}\u{0666}\u{2067}\u{0600}\u{2067}\u{10E73}\u{2067}\u{0603}\u{2067}\u{10E74}\u{2068}\u{1061C}\u{2068}\u{F9B3}\u{2068}\u{1891}\u{2068}\u{1BE1}\u{2068}\u{1D9B5}\u{2068}\u{1924}\u{2068}\u{0032}\u{2068}\u{1D7DF}\u{2068}\u{1D7FF}\u{2068}\u{1F105}\u{2068}\u{1D7F8}\u{2068}\u{1F104}\u{2068}\u{207B}\u{2068}\u{FE62}\u{2068}\u{FB29}\u{2068}\u{FF0D}\u{2068}\u{207B}\u{2068}\u{207A}\u{2068}\u{20B1}\u{2068}\u{20B0}\u{2068}\u{FF04}\u{2068}\u{09F3}\u{2068}\u{20B9}\u{2068}\u{FF05}\u{2068}\u{060C}\u{2068}\u{FE52}\u{2068}\u{FF1A}\u{2068}\u{FF0C}\u{2068}\u{FF0E}\u{2068}\u{002E}\u{2068}\u{E0196}\u{2068}\u{0595}\u{2068}\u{A9B9}\u{2068}\u{036F}\u{2068}\u{E0115}\u{2068}\u{0C46}\u{2068}\u{1F3C2}\u{2068}\u{2B15}\u{2068}\u{2616}\u{2068}\u{1F36C}\u{2068}\u{1F45F}\u{2068}\u{1F0EF}\u{2069}\u{0601}\u{2069}\u{10E6C}\u{2069}\u{10E75}\u{2069}\u{10E6F}\u{2069}\u{0601}\u{2069}\u{10E7C}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, AN, L, AN, L, AN, L, AN, L, AN, L, AN, EN, AN, EN, AN, EN, AN, EN, AN, EN, AN, EN, AN, ES, AN, ES, AN, ES, AN, ES, AN, ES, AN, ES, AN, ET, AN, ET, AN, ET, AN, ET, AN, ET, AN, ET, AN, CS, AN, CS, AN, CS, AN, CS, AN, CS, AN, CS, AN, NSM, AN, NSM, AN, NSM, AN, NSM, AN, NSM, AN, NSM, AN, S, AN, S, AN, S, AN, S, AN, S, AN, S, AN, WS, AN, WS, AN, WS, AN, WS, AN, WS, AN, WS, AN, ON, AN, ON, AN, ON, AN, ON, AN, ON, AN, ON, AN, LRI, L, LRI, L, LRI, L, LRI, L, LRI, L, LRI, L, LRI, EN, LRI, EN, LRI, EN, LRI, EN, LRI, EN, LRI, EN, LRI, ES, LRI, ES, LRI, ES, LRI, ES, LRI, ES, LRI, ES, LRI, ET, LRI, ET, LRI, ET, LRI, ET, LRI, ET, LRI, ET, LRI, CS, LRI, CS, LRI, CS, LRI, CS, LRI, CS, LRI, CS, LRI, NSM, LRI, NSM, LRI, NSM, LRI, NSM, LRI, NSM, LRI, NSM, LRI, ON, LRI, ON, LRI, ON, LRI, ON, LRI, ON, LRI, ON, RLI, L, RLI, L, RLI, L, RLI, L, RLI, L, RLI, L, RLI, EN, RLI, EN, RLI, EN, RLI, EN, RLI, EN, RLI, EN, RLI, AN, RLI, AN, RLI, AN, RLI, AN, RLI, AN, RLI, AN, FSI, L, FSI, L, FSI, L, FSI, L, FSI, L, FSI, L, FSI, EN, FSI, EN, FSI, EN, FSI, EN, FSI, EN, FSI, EN, FSI, ES, FSI, ES, FSI, ES, FSI, ES, FSI, ES, FSI, ES, FSI, ET, FSI, ET, FSI, ET, FSI, ET, FSI, ET, FSI, ET, FSI, CS, FSI, CS, FSI, CS, FSI, CS, FSI, CS, FSI, CS, FSI, NSM, FSI, NSM, FSI, NSM, FSI, NSM, FSI, NSM, FSI, NSM, FSI, ON, FSI, ON, FSI, ON, FSI, ON, FSI, ON, FSI, ON, PDI, AN, PDI, AN, PDI, AN, PDI, AN, PDI, AN, PDI, AN], paragraphs: vec![ParagraphInfo { range: 0..162, level: 0 } ], });
// assert_eq!(process_text("\u{2066}\u{10926}\u{2066}\u{10CE3}\u{2066}\u{10CED}\u{2066}\u{1E870}\u{2066}\u{10A71}\u{2066}\u{10C0C}\u{2066}\u{FBAB}\u{2066}\u{FEDD}\u{2066}\u{06CE}\u{2066}\u{FE97}\u{2066}\u{1EE51}\u{2066}\u{FC40}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![LRI, R, LRI, R, LRI, R, LRI, R, LRI, R, LRI, R, LRI, AL, LRI, AL, LRI, AL, LRI, AL, LRI, AL, LRI, AL], paragraphs: vec![ParagraphInfo { range: 0..12, level: 0 } ], });
// assert_eq!(process_text("\u{2066}\u{0603}\u{2066}\u{0660}\u{2066}\u{0660}\u{2066}\u{10E6A}\u{2066}\u{10E6C}\u{2066}\u{10E74}\u{2068}\u{10E65}\u{2068}\u{10E7A}\u{2068}\u{10E66}\u{2068}\u{10E7C}\u{2068}\u{0661}\u{2068}\u{10E6A}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![LRI, AN, LRI, AN, LRI, AN, LRI, AN, LRI, AN, LRI, AN, FSI, AN, FSI, AN, FSI, AN, FSI, AN, FSI, AN, FSI, AN], paragraphs: vec![ParagraphInfo { range: 0..12, level: 0 } ], });
// assert_eq!(process_text("\u{A3A8}\u{10A8E}\u{A642}\u{FCCF}\u{0D2E}\u{2212}\u{11328}\u{FF04}\u{1330D}\u{003A}\u{10501}\u{16F90}\u{10A2}\u{21ED}\u{A6D4}\u{16F2F}\u{102BE}\u{10932}\u{1688A}\u{0726}\u{AB9A}\u{102F3}\u{1F55}\u{FE62}\u{03EB}\u{2032}\u{ABF6}\u{0601}\u{31B9}\u{003A}\u{1A82}\u{302B}\u{3334}\u{238F}\u{33F9}\u{10C8A}\u{1B30}\u{FE86}\u{10689}\u{10B69}\u{006C}\u{1EE6C}\u{1D7D1}\u{10A61}\u{102EC}\u{0630}\u{248B}\u{FE62}\u{06F8}\u{00A4}\u{1D7F9}\u{FF1A}\u{FF19}\u{1074}\u{0033}\u{2F57}\u{1D7EF}\u{1F14E}\u{1D7FE}\u{10B12}\u{1F101}\u{FEC0}\u{2494}\u{1F10A}\u{06F3}\u{FF0B}\u{102E1}\u{20AB}\u{102F5}\u{066C}\u{1F107}\u{00A0}\u{102FB}\u{1DA2C}\u{102FB}\u{19E4}\u{102F6}\u{10855}\u{0038}\u{FD6D}\u{2499}\u{1E815}\u{1D7D3}\u{FD52}\u{FF0D}\u{1092D}\u{208B}\u{FBD5}\u{FB29}\u{FB29}\u{208A}\u{FF04}\u{208B}\u{2044}\u{FE62}\u{1D181}\u{FB29}\u{204F}\u{207B}\u{168BE}\u{2212}\u{1088F}\u{2212}\u{0786}\u{FB29}\u{102F7}\u{207B}\u{FE63}\u{208B}\u{00A4}\u{002D}\u{0669}\u{208A}\u{060C}\u{FB29}\u{0C4B}\u{FB29}\u{1F4C6}\u{002B}\u{1E8B1}\u{208A}\u{FBFF}\u{FE62}\u{1E862}\u{207B}\u{0682}\u{20AE}\u{1E8BD}\u{20A9}\u{FD8A}\u{2032}\u{002D}\u{0BF9}\u{FFE0}\u{FFE1}\u{FF0C}\u{20BA}\u{1CE3}\u{20A7}\u{1F817}\u{20A6}\u{191E}\u{A838}\u{1E858}\u{FF04}\u{FC2A}\u{FFE5}\u{1D7FE}\u{20BE}\u{2212}\u{17DB}\u{20B9}\u{2034}\u{0605}\u{0E3F}\u{003A}\u{2033}\u{E0111}\u{20B8}\u{FFFA}\u{FE5F}\u{10CEB}\u{20AB}\u{FCA0}\u{20A4}\u{10CEC}\u{20BA}\u{FC85}\u{FF1A}\u{10CC2}\u{FE52}\u{FC80}\u{FE50}\u{FB29}\u{002E}\u{066A}\u{FF0E}\u{003A}\u{FE52}\u{A679}\u{FF1A}\u{1D21C}\u{002C}\u{A11C}\u{FF0F}\u{05DE}\u{FE52}\u{1EE42}\u{FE50}\u{2494}\u{FE52}\u{002D}\u{060C}\u{0024}\u{FF0F}\u{10E73}\u{FF1A}\u{FE52}\u{FF0C}\u{1103B}\u{003A}\u{1F770}\u{00A0}\u{10C47}\u{FF1A}\u{FD5B}\u{FE55}\u{10842}\u{FF1A}\u{FED7}\u{09C1}\u{10A7B}\u{0F73}\u{0752}\u{032D}\u{FE62}\u{E015D}\u{20BD}\u{1DA07}\u{060C}\u{FE2A}\u{08EE}\u{1C32}\u{27E2}\u{E013C}\u{106F5}\u{0487}\u{10834}\u{09C4}\u{1EE3B}\u{2DEC}\u{06F8}\u{1D17C}\u{208B}\u{06E2}\u{0609}\u{20ED}\u{0661}\u{10A03}\u{002C}\u{09BC}\u{E012E}\u{1171F}\u{2231}\u{1DE1}\u{109C7}\u{08E7}\u{FB87}\u{08FA}\u{109C8}\u{112E3}\u{FD2C}\u{000B}\u{10C11}\u{0009}\u{FEF1}\u{0009}\u{208A}\u{0009}\u{A839}\u{001F}\u{FF0C}\u{000B}\u{1AB4}\u{000B}\u{02C6}\u{0009}\u{1BC73}\u{000B}\u{109FF}\u{001F}\u{1EEA3}\u{0009}\u{FF11}\u{001F}\u{208B}\u{001F}\u{20BC}\u{000B}\u{0662}\u{001F}\u{202F}\u{0009}\u{0FA3}\u{001F}\u{26F7}\u{000B}\u{1080B}\u{0009}\u{FBDF}\u{0009}\u{1085F}\u{001F}\u{FEBA}\u{2005}\u{1E832}\u{2004}\u{06BD}\u{2004}\u{207A}\u{2009}\u{20A2}\u{000C}\u{002C}\u{2006}\u{0FA9}\u{2007}\u{2210}\u{2000}\u{018C}\u{3000}\u{10989}\u{2003}\u{FCE0}\u{3000}\u{FF16}\u{2001}\u{FF0D}\u{2006}\u{A838}\u{2005}\u{066B}\u{205F}\u{003A}\u{2005}\u{0BCD}\u{2001}\u{231A}\u{2008}\u{10C11}\u{3000}\u{FBA5}\u{2006}\u{10B27}\u{3000}\u{FC6B}\u{FE18}\u{10C8E}\u{256E}\u{0629}\u{FE56}\u{FF0B}\u{2AF4}\u{20BD}\u{2B12}\u{2044}\u{1F455}\u{1DED}\u{1F0DE}\u{23B1}\u{FE48}\u{16896}\u{2100}\u{10B8E}\u{00AC}\u{FC58}\u{3010}\u{102F7}\u{1F050}\u{002B}\u{1F3C1}\u{20A1}\u{2B36}\u{10E79}\u{10155}\u{002F}\u{1F706}\u{07A6}\u{2A2F}\u{27F9}\u{23D4}\u{10A76}\u{2418}\u{FC5E}\u{2220}\u{10A1C}\u{1F729}\u{FE86}\u{2067}\u{1E8BE}\u{2067}\u{FDB3}\u{2067}\u{FF0D}\u{2067}\u{20B7}\u{2067}\u{FE55}\u{2067}\u{1BED}\u{2067}\u{23DC}\u{2067}\u{10B8B}\u{2067}\u{FDB2}\u{2067}\u{208B}\u{2067}\u{FFE6}\u{2067}\u{202F}\u{2067}\u{115DD}\u{2067}\u{2055}\u{2068}\u{1E88D}\u{2068}\u{0707}\u{2068}\u{10CEE}\u{2068}\u{072C}\u{2069}\u{10AD6}\u{2069}\u{FD99}\u{2069}\u{FF0D}\u{2069}\u{20A4}\u{2069}\u{00A0}\u{2069}\u{1D169}\u{2069}\u{2FBA}\u{2069}\u{A945}\u{2069}\u{1E897}\u{2069}\u{06C7}\u{2069}\u{06F5}\u{2069}\u{002D}\u{2069}\u{A838}\u{2069}\u{10E63}\u{2069}\u{FF0C}\u{2069}\u{0302}\u{2069}\u{10160}\u{2069}\u{10C91}\u{2069}\u{06C7}\u{2069}\u{10A6E}\u{2069}\u{0772}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2], classes: vec![L, R, L, AL, L, ES, L, ET, L, CS, L, NSM, L, ON, L, L, L, R, L, AL, L, EN, L, ES, L, ET, L, AN, L, CS, L, NSM, L, ON, L, R, L, AL, L, R, L, AL, EN, R, EN, AL, EN, ES, EN, ET, EN, CS, EN, NSM, EN, ON, EN, L, EN, R, EN, AL, EN, EN, EN, ES, EN, ET, EN, AN, EN, CS, EN, NSM, EN, ON, EN, R, EN, AL, EN, R, EN, AL, ES, R, ES, AL, ES, ES, ES, ET, ES, CS, ES, NSM, ES, ON, ES, L, ES, R, ES, AL, ES, EN, ES, ES, ES, ET, ES, AN, ES, CS, ES, NSM, ES, ON, ES, R, ES, AL, ES, R, ES, AL, ET, R, ET, AL, ET, ES, ET, ET, ET, CS, ET, NSM, ET, ON, ET, L, ET, R, ET, AL, ET, EN, ET, ES, ET, ET, ET, AN, ET, CS, ET, NSM, ET, ON, ET, R, ET, AL, ET, R, ET, AL, CS, R, CS, AL, CS, ES, CS, ET, CS, CS, CS, NSM, CS, ON, CS, L, CS, R, CS, AL, CS, EN, CS, ES, CS, ET, CS, AN, CS, CS, CS, NSM, CS, ON, CS, R, CS, AL, CS, R, CS, AL, NSM, R, NSM, AL, NSM, ES, NSM, ET, NSM, CS, NSM, NSM, NSM, ON, NSM, L, NSM, R, NSM, AL, NSM, EN, NSM, ES, NSM, ET, NSM, AN, NSM, CS, NSM, NSM, NSM, ON, NSM, R, NSM, AL, NSM, R, NSM, AL, S, R, S, AL, S, ES, S, ET, S, CS, S, NSM, S, ON, S, L, S, R, S, AL, S, EN, S, ES, S, ET, S, AN, S, CS, S, NSM, S, ON, S, R, S, AL, S, R, S, AL, WS, R, WS, AL, WS, ES, WS, ET, WS, CS, WS, NSM, WS, ON, WS, L, WS, R, WS, AL, WS, EN, WS, ES, WS, ET, WS, AN, WS, CS, WS, NSM, WS, ON, WS, R, WS, AL, WS, R, WS, AL, ON, R, ON, AL, ON, ES, ON, ET, ON, CS, ON, NSM, ON, ON, ON, L, ON, R, ON, AL, ON, EN, ON, ES, ON, ET, ON, AN, ON, CS, ON, NSM, ON, ON, ON, R, ON, AL, ON, R, ON, AL, RLI, R, RLI, AL, RLI, ES, RLI, ET, RLI, CS, RLI, NSM, RLI, ON, RLI, R, RLI, AL, RLI, ES, RLI, ET, RLI, CS, RLI, NSM, RLI, ON, FSI, R, FSI, AL, FSI, R, FSI, AL, PDI, R, PDI, AL, PDI, ES, PDI, ET, PDI, CS, PDI, NSM, PDI, ON, PDI, L, PDI, R, PDI, AL, PDI, EN, PDI, ES, PDI, ET, PDI, AN, PDI, CS, PDI, NSM, PDI, ON, PDI, R, PDI, AL, PDI, R, PDI, AL], paragraphs: vec![ParagraphInfo { range: 0..228, level: 0 } ], });
// assert_eq!(process_text("\u{14D8}\u{133DD}\u{1A24}\u{1D7EB}\u{1E85}\u{FE62}\u{120DD}\u{20A8}\u{1D701}\u{002C}\u{1D57B}\u{1035}\u{A80C}\u{2EC2}\u{111AC}\u{2F80D}\u{14552}\u{10C3B}\u{11119}\u{0685}\u{AB35}\u{1D7DE}\u{12310}\u{002D}\u{1200C}\u{20A7}\u{A5CE}\u{10E66}\u{A1F1}\u{202F}\u{2870}\u{E01AA}\u{1D112}\u{1F809}\u{004A}\u{16922}\u{13408}\u{102E6}\u{13126}\u{10E6E}\u{10A2}\u{10E69}\u{124A4}\u{10E7D}\u{1D7F6}\u{1863}\u{2080}\u{1D7E9}\u{1F105}\u{FF0D}\u{102FB}\u{066A}\u{1D7F7}\u{FF0E}\u{1D7FA}\u{073F}\u{102F3}\u{21F6}\u{1F102}\u{1D563}\u{2084}\u{109AA}\u{102E7}\u{FDC4}\u{102E7}\u{1F10A}\u{1D7CF}\u{002D}\u{1D7D2}\u{00A3}\u{2489}\u{10E7E}\u{06F0}\u{FE50}\u{1F102}\u{0A51}\u{102F1}\u{22F9}\u{FF11}\u{1A33}\u{2088}\u{2076}\u{102E4}\u{10E6C}\u{FF11}\u{10E6B}\u{1D7F1}\u{10E73}\u{FB29}\u{3317}\u{208A}\u{1D7E5}\u{002D}\u{002B}\u{2212}\u{20BC}\u{FB29}\u{003A}\u{FF0B}\u{E01B0}\u{208A}\u{1F05C}\u{2212}\u{14598}\u{208A}\u{0832}\u{002D}\u{FE7C}\u{207B}\u{1D7D3}\u{FE63}\u{207B}\u{2212}\u{20B3}\u{FF0D}\u{10E6A}\u{208B}\u{003A}\u{002B}\u{081E}\u{2212}\u{22BE}\u{FB29}\u{A4EE}\u{FE63}\u{06F4}\u{FF0B}\u{10E7D}\u{002D}\u{0664}\u{2212}\u{10E69}\u{20AA}\u{1145}\u{00A2}\u{06F0}\u{20AF}\u{208A}\u{20B0}\u{20B0}\u{20A4}\u{060C}\u{20BA}\u{A9B7}\u{0AF1}\u{2921}\u{09F3}\u{12472}\u{FFE5}\u{10AC4}\u{0024}\u{1EE12}\u{20A8}\u{2495}\u{20B3}\u{208B}\u{20A6}\u{FFE5}\u{2034}\u{10E6A}\u{00A4}\u{2044}\u{20B6}\u{0F83}\u{09FB}\u{32B8}\u{2213}\u{106D1}\u{17DB}\u{102F8}\u{FE6A}\u{10E6E}\u{20BC}\u{0664}\u{20B5}\u{0600}\u{060C}\u{28B8}\u{FF0C}\u{2499}\u{00A0}\u{208A}\u{FF0C}\u{20BB}\u{2044}\u{FE50}\u{FE55}\u{17CF}\u{2044}\u{2387}\u{00A0}\u{A364}\u{FF1A}\u{10B26}\u{FF1A}\u{FDAA}\u{002E}\u{2089}\u{002F}\u{207A}\u{002F}\u{00B1}\u{003A}\u{10E76}\u{FE50}\u{FF0E}\u{002C}\u{059B}\u{002E}\u{241E}\u{00A0}\u{31A0}\u{2044}\u{1D7F5}\u{002C}\u{10E75}\u{FE50}\u{0664}\u{2044}\u{10E72}\u{A950}\u{AB78}\u{E019A}\u{1F100}\u{E01AF}\u{208B}\u{111B9}\u{20B3}\u{0F9D}\u{00A0}\u{0C3F}\u{08F1}\u{11129}\u{2E8F}\u{0E4D}\u{144C2}\u{1CE0}\u{10CC6}\u{E0199}\u{FCC3}\u{E0187}\u{2086}\u{074A}\u{002B}\u{10A06}\u{FE69}\u{0654}\u{0665}\u{180D}\u{FF0E}\u{1DC6}\u{1A5C}\u{E01D4}\u{1F46C}\u{1DCE}\u{120D7}\u{0AC8}\u{0035}\u{07F3}\u{10E7A}\u{E0185}\u{10E77}\u{1772}\u{10E62}\u{000B}\u{0249}\u{0009}\u{1D7D5}\u{000B}\u{FE63}\u{0009}\u{00B1}\u{001F}\u{2044}\u{000B}\u{E01B8}\u{0009}\u{1F918}\u{000B}\u{100ED}\u{0009}\u{10A13}\u{000B}\u{FE98}\u{000B}\u{1D7FD}\u{000B}\u{FE63}\u{001F}\u{20B3}\u{001F}\u{10E61}\u{0009}\u{002C}\u{001F}\u{A679}\u{001F}\u{2B2F}\u{000B}\u{12A3}\u{001F}\u{FF17}\u{0009}\u{10E66}\u{001F}\u{10E77}\u{000B}\u{10E64}\u{2002}\u{0125}\u{2004}\u{1D7FF}\u{2006}\u{207A}\u{205F}\u{20A6}\u{2007}\u{FF0E}\u{2006}\u{FE29}\u{2001}\u{25B5}\u{2005}\u{19C2}\u{2004}\u{10C2B}\u{3000}\u{FCC4}\u{2001}\u{2498}\u{2004}\u{FB29}\u{205F}\u{20BB}\u{2004}\u{0604}\u{1680}\u{FE55}\u{2003}\u{0826}\u{2007}\u{1F488}\u{3000}\u{131DC}\u{2008}\u{1D7FA}\u{2028}\u{10E69}\u{205F}\u{0660}\u{2003}\u{0660}\u{A71A}\u{0403}\u{1F752}\u{FF10}\u{2F15}\u{002D}\u{2A45}\u{20B3}\u{29F8}\u{202F}\u{3004}\u{E0122}\u{1F7AE}\u{19E1}\u{1F7CC}\u{1D59B}\u{2939}\u{1E878}\u{2722}\u{FEE4}\u{1F686}\u{2087}\u{1807}\u{002B}\u{25F2}\u{2032}\u{2A5A}\u{10E6B}\u{1F898}\u{FF0E}\u{23C1}\u{A802}\u{2B23}\u{1F5D4}\u{1F7D2}\u{10517}\u{1F55D}\u{1D7E7}\u{270A}\u{10E72}\u{2B6A}\u{10E69}\u{2ED7}\u{0660}\u{2066}\u{18A1}\u{2066}\u{1D7ED}\u{2066}\u{FE62}\u{2066}\u{20BA}\u{2066}\u{FE52}\u{2066}\u{1171D}\u{2066}\u{23AA}\u{2066}\u{1233B}\u{2066}\u{2070}\u{2066}\u{208B}\u{2066}\u{09F3}\u{2066}\u{FE52}\u{2066}\u{0AC4}\u{2066}\u{FF5C}\u{2067}\u{12525}\u{2067}\u{1F108}\u{2067}\u{002D}\u{2067}\u{20BA}\u{2067}\u{002E}\u{2067}\u{0F97}\u{2067}\u{1D33F}\u{2067}\u{1F9B}\u{2067}\u{1098D}\u{2067}\u{FC13}\u{2067}\u{1D7EE}\u{2067}\u{FE63}\u{2067}\u{20AC}\u{2067}\u{10E72}\u{2067}\u{003A}\u{2067}\u{2DE9}\u{2067}\u{27E3}\u{2067}\u{014C}\u{2067}\u{1D7F4}\u{2067}\u{10E66}\u{2067}\u{336F}\u{2067}\u{2074}\u{2067}\u{0602}\u{2068}\u{1082F}\u{2068}\u{FEB8}\u{2068}\u{1628}\u{2068}\u{2083}\u{2068}\u{FF0B}\u{2068}\u{20BA}\u{2068}\u{FF0C}\u{2068}\u{0591}\u{2068}\u{1014A}\u{2068}\u{0D7D}\u{2068}\u{FF14}\u{2068}\u{002D}\u{2068}\u{09FB}\u{2068}\u{FF0C}\u{2068}\u{0FAD}\u{2068}\u{2253}\u{2069}\u{3181}\u{2069}\u{2494}\u{2069}\u{208A}\u{2069}\u{2032}\u{2069}\u{060C}\u{2069}\u{0B41}\u{2069}\u{2964}\u{2069}\u{1D653}\u{2069}\u{108E9}\u{2069}\u{FDC1}\u{2069}\u{1D7E0}\u{2069}\u{208A}\u{2069}\u{FE69}\u{2069}\u{10E60}\u{2069}\u{002E}\u{2069}\u{E01C8}\u{2069}\u{2390}\u{2069}\u{A313}\u{2069}\u{0030}\u{2069}\u{10E7B}\u{2069}\u{0604}\u{2069}\u{10E67}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, L, L, EN, L, ES, L, ET, L, CS, L, NSM, L, ON, L, L, L, R, L, AL, L, EN, L, ES, L, ET, L, AN, L, CS, L, NSM, L, ON, L, L, L, EN, L, AN, L, AN, L, AN, EN, L, EN, EN, EN, ES, EN, ET, EN, CS, EN, NSM, EN, ON, EN, L, EN, R, EN, AL, EN, EN, EN, ES, EN, ET, EN, AN, EN, CS, EN, NSM, EN, ON, EN, L, EN, EN, EN, AN, EN, AN, EN, AN, ES, L, ES, EN, ES, ES, ES, ET, ES, CS, ES, NSM, ES, ON, ES, L, ES, R, ES, AL, ES, EN, ES, ES, ES, ET, ES, AN, ES, CS, ES, NSM, ES, ON, ES, L, ES, EN, ES, AN, ES, AN, ES, AN, ET, L, ET, EN, ET, ES, ET, ET, ET, CS, ET, NSM, ET, ON, ET, L, ET, R, ET, AL, ET, EN, ET, ES, ET, ET, ET, AN, ET, CS, ET, NSM, ET, ON, ET, L, ET, EN, ET, AN, ET, AN, ET, AN, CS, L, CS, EN, CS, ES, CS, ET, CS, CS, CS, NSM, CS, ON, CS, L, CS, R, CS, AL, CS, EN, CS, ES, CS, ET, CS, AN, CS, CS, CS, NSM, CS, ON, CS, L, CS, EN, CS, AN, CS, AN, CS, AN, NSM, L, NSM, EN, NSM, ES, NSM, ET, NSM, CS, NSM, NSM, NSM, ON, NSM, L, NSM, R, NSM, AL, NSM, EN, NSM, ES, NSM, ET, NSM, AN, NSM, CS, NSM, NSM, NSM, ON, NSM, L, NSM, EN, NSM, AN, NSM, AN, NSM, AN, S, L, S, EN, S, ES, S, ET, S, CS, S, NSM, S, ON, S, L, S, R, S, AL, S, EN, S, ES, S, ET, S, AN, S, CS, S, NSM, S, ON, S, L, S, EN, S, AN, S, AN, S, AN, WS, L, WS, EN, WS, ES, WS, ET, WS, CS, WS, NSM, WS, ON, WS, L, WS, R, WS, AL, WS, EN, WS, ES, WS, ET, WS, AN, WS, CS, WS, NSM, WS, ON, WS, L, WS, EN, WS, AN, WS, AN, WS, AN, ON, L, ON, EN, ON, ES, ON, ET, ON, CS, ON, NSM, ON, ON, ON, L, ON, R, ON, AL, ON, EN, ON, ES, ON, ET, ON, AN, ON, CS, ON, NSM, ON, ON, ON, L, ON, EN, ON, AN, ON, AN, ON, AN, LRI, L, LRI, EN, LRI, ES, LRI, ET, LRI, CS, LRI, NSM, LRI, ON, LRI, L, LRI, EN, LRI, ES, LRI, ET, LRI, CS, LRI, NSM, LRI, ON, RLI, L, RLI, EN, RLI, ES, RLI, ET, RLI, CS, RLI, NSM, RLI, ON, RLI, L, RLI, R, RLI, AL, RLI, EN, RLI, ES, RLI, ET, RLI, AN, RLI, CS, RLI, NSM, RLI, ON, RLI, L, RLI, EN, RLI, AN, RLI, L, RLI, EN, RLI, AN, FSI, R, FSI, AL, FSI, L, FSI, EN, FSI, ES, FSI, ET, FSI, CS, FSI, NSM, FSI, ON, FSI, L, FSI, EN, FSI, ES, FSI, ET, FSI, CS, FSI, NSM, FSI, ON, PDI, L, PDI, EN, PDI, ES, PDI, ET, PDI, CS, PDI, NSM, PDI, ON, PDI, L, PDI, R, PDI, AL, PDI, EN, PDI, ES, PDI, ET, PDI, AN, PDI, CS, PDI, NSM, PDI, ON, PDI, L, PDI, EN, PDI, AN, PDI, AN, PDI, AN], paragraphs: vec![ParagraphInfo { range: 0..273, level: 0 } ], });
// assert_eq!(process_text("\u{0BC2}\u{10B85}\u{1B7B}\u{0699}\u{00B3}\u{1E86C}\u{1F101}\u{0630}\u{FF0D}\u{1E885}\u{002D}\u{1EEA8}\u{09FB}\u{1E8C2}\u{00A5}\u{1EEAF}\u{002F}\u{1088B}\u{FF0C}\u{076B}\u{11132}\u{10A2C}\u{2DE7}\u{FEEC}\u{001F}\u{10A12}\u{001F}\u{FCD0}\u{200A}\u{10924}\u{2009}\u{FCAF}\u{1390}\u{10831}\u{FE3A}\u{0727}\u{2066}\u{10A84}\u{2066}\u{0792}\u{2066}\u{002D}\u{2066}\u{20B5}\u{2066}\u{FF0F}\u{2066}\u{1DDB}\u{2066}\u{1F707}\u{2066}\u{12524}\u{2066}\u{10A96}\u{2066}\u{1EE27}\u{2066}\u{2489}\u{2066}\u{FE62}\u{2066}\u{20A0}\u{2066}\u{10E64}\u{2066}\u{202F}\u{2066}\u{0311}\u{2066}\u{26DE}\u{2066}\u{080C}\u{2066}\u{FECC}\u{2066}\u{1E849}\u{2066}\u{0764}\u{2067}\u{10863}\u{2067}\u{1EE7A}\u{2067}\u{1E86E}\u{2067}\u{1EE4B}\u{2067}\u{207A}\u{2067}\u{2031}\u{2067}\u{FE55}\u{2067}\u{115B5}\u{2067}\u{A49A}\u{2067}\u{AA56}\u{2067}\u{080C}\u{2067}\u{FC8A}\u{2067}\u{102F2}\u{2067}\u{208A}\u{2067}\u{20BA}\u{2067}\u{10E76}\u{2067}\u{FF0C}\u{2067}\u{11637}\u{2067}\u{1F394}\u{2068}\u{10C45}\u{2068}\u{FBEC}\u{2068}\u{10B01}\u{2068}\u{FE82}\u{2068}\u{207A}\u{2068}\u{2031}\u{2068}\u{002E}\u{2068}\u{0946}\u{2068}\u{19F6}\u{2068}\u{A449}\u{2068}\u{10A21}\u{2068}\u{1EE89}\u{2068}\u{2086}\u{2068}\u{FF0D}\u{2068}\u{060A}\u{2068}\u{10E78}\u{2068}\u{002E}\u{2068}\u{10A01}\u{2068}\u{2B09}\u{2069}\u{084A}\u{2069}\u{FBB8}", Some(0)), BidiInfo { levels: vec![ 3,  3,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2], classes: vec![L, R, L, AL, EN, R, EN, AL, ES, R, ES, AL, ET, R, ET, AL, CS, R, CS, AL, NSM, R, NSM, AL, S, R, S, AL, WS, R, WS, AL, ON, R, ON, AL, LRI, R, LRI, AL, LRI, ES, LRI, ET, LRI, CS, LRI, NSM, LRI, ON, LRI, L, LRI, R, LRI, AL, LRI, EN, LRI, ES, LRI, ET, LRI, AN, LRI, CS, LRI, NSM, LRI, ON, LRI, R, LRI, AL, LRI, R, LRI, AL, RLI, R, RLI, AL, RLI, R, RLI, AL, RLI, ES, RLI, ET, RLI, CS, RLI, NSM, RLI, ON, RLI, L, RLI, R, RLI, AL, RLI, EN, RLI, ES, RLI, ET, RLI, AN, RLI, CS, RLI, NSM, RLI, ON, FSI, R, FSI, AL, FSI, R, FSI, AL, FSI, ES, FSI, ET, FSI, CS, FSI, NSM, FSI, ON, FSI, L, FSI, R, FSI, AL, FSI, EN, FSI, ES, FSI, ET, FSI, AN, FSI, CS, FSI, NSM, FSI, ON, PDI, R, PDI, AL], paragraphs: vec![ParagraphInfo { range: 0..79, level: 0 } ], });
// assert_eq!(process_text("\u{1E12}\u{0603}\u{1D7E5}\u{10E66}\u{FE62}\u{0668}\u{20A0}\u{10E63}\u{00A0}\u{10E7D}\u{0A42}\u{10E68}\u{001F}\u{10E7A}\u{2003}\u{066C}\u{1F3F6}\u{0668}\u{2066}\u{1D860}\u{2066}\u{2083}\u{2066}\u{FE63}\u{2066}\u{20A7}\u{2066}\u{FE52}\u{2066}\u{1CD9}\u{2066}\u{295B}\u{2066}\u{101D}\u{2066}\u{1E82D}\u{2066}\u{0718}\u{2066}\u{2078}\u{2066}\u{FE62}\u{2066}\u{20AC}\u{2066}\u{0603}\u{2066}\u{FF0F}\u{2066}\u{17B7}\u{2066}\u{295B}\u{2066}\u{1167}\u{2066}\u{2079}\u{2066}\u{10E60}\u{2066}\u{10E77}\u{2066}\u{10E64}\u{2067}\u{10E6E}\u{2067}\u{1DA38}\u{2067}\u{102E2}\u{2067}\u{0666}\u{2068}\u{13113}\u{2068}\u{1F109}\u{2068}\u{FE63}\u{2068}\u{20A9}\u{2068}\u{FE52}\u{2068}\u{AAEC}\u{2068}\u{1F091}\u{2068}\u{A59A}\u{2068}\u{2089}\u{2068}\u{FF0B}\u{2068}\u{09F2}\u{2068}\u{0660}\u{2068}\u{FF0C}\u{2068}\u{1B3A}\u{2068}\u{1F73F}\u{2068}\u{121CD}\u{2068}\u{1D7F3}\u{2068}\u{10E6B}\u{2068}\u{10E6D}\u{2068}\u{10E62}\u{2069}\u{10E6C}", Some(0)), BidiInfo { levels: vec![ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3], classes: vec![L, AN, EN, AN, ES, AN, ET, AN, CS, AN, NSM, AN, S, AN, WS, AN, ON, AN, LRI, L, LRI, EN, LRI, ES, LRI, ET, LRI, CS, LRI, NSM, LRI, ON, LRI, L, LRI, R, LRI, AL, LRI, EN, LRI, ES, LRI, ET, LRI, AN, LRI, CS, LRI, NSM, LRI, ON, LRI, L, LRI, EN, LRI, AN, LRI, AN, LRI, AN, RLI, AN, RLI, L, RLI, EN, RLI, AN, FSI, L, FSI, EN, FSI, ES, FSI, ET, FSI, CS, FSI, NSM, FSI, ON, FSI, L, FSI, EN, FSI, ES, FSI, ET, FSI, AN, FSI, CS, FSI, NSM, FSI, ON, FSI, L, FSI, EN, FSI, AN, FSI, AN, FSI, AN, PDI, AN], paragraphs: vec![ParagraphInfo { range: 0..56, level: 0 } ], });
// assert_eq!(process_text("\u{2066}\u{10CEA}\u{2066}\u{FC29}", Some(0)), BidiInfo { levels: vec![ 3,  3], classes: vec![LRI, R, LRI, AL], paragraphs: vec![ParagraphInfo { range: 0..2, level: 0 } ], });
// assert_eq!(process_text("\u{2066}\u{10E74}\u{2068}\u{0605}", Some(0)), BidiInfo { levels: vec![ 3,  3], classes: vec![LRI, AN, FSI, AN], paragraphs: vec![ParagraphInfo { range: 0..2, level: 0 } ], });
// assert_eq!(process_text("\u{109C7}\u{2C57}\u{109F7}\u{12049}\u{10B28}\u{1475}\u{10C0E}\u{1723}\u{10885}\u{AA76}\u{109FC}\u{0254}\u{10C3E}\u{FF0D}\u{10857}\u{208B}\u{10CFE}\u{207B}\u{1E889}\u{FE63}\u{10B6C}\u{002D}\u{109EA}\u{002D}\u{0856}\u{A839}\u{10A40}\u{00A4}\u{10A24}\u{FE69}\u{05EA}\u{20A1}\u{10B33}\u{20BA}\u{109BC}\u{00A5}\u{10A61}\u{FF0E}\u{05E7}\u{FF0C}\u{1E82A}\u{00A0}\u{0845}\u{FF0E}\u{10903}\u{FE50}\u{1E8C9}\u{2044}\u{1E84F}\u{001F}\u{10890}\u{001F}\u{109F9}\u{000B}\u{1092D}\u{000B}\u{1086E}\u{000B}\u{10B4C}\u{0009}\u{1085C}\u{2007}\u{10A61}\u{205F}\u{0844}\u{2028}\u{109A4}\u{000C}\u{1E801}\u{2004}\u{1E829}\u{2008}\u{109A1}\u{2947}\u{10A42}\u{1F460}\u{FB3B}\u{2A39}\u{1E8B7}\u{2B73}\u{109D3}\u{1D327}\u{1E8C0}\u{10B3C}\u{10891}\u{2066}\u{1E853}\u{2066}\u{10A88}\u{2066}\u{10A66}\u{2066}\u{10815}\u{2066}\u{10859}\u{2066}\u{1E80C}\u{2067}\u{109B6}\u{2067}\u{109A5}\u{2067}\u{10C28}\u{2067}\u{1087B}\u{2067}\u{1E839}\u{2067}\u{1E81E}\u{2068}\u{10A45}\u{2068}\u{05D2}\u{2068}\u{1E8C7}\u{2068}\u{109C4}\u{2068}\u{1098E}\u{2068}\u{1E8B7}\u{2069}\u{1E801}\u{2069}\u{10871}\u{2069}\u{07D6}\u{2069}\u{10A8E}\u{2069}\u{0801}\u{2069}\u{FC43}\u{1148C}\u{FBA2}\u{1D922}\u{FBA7}\u{16C4}\u{FCB4}\u{3125}\u{FC88}\u{16F03}\u{FC4C}\u{121CF}\u{1EEA7}\u{FE63}\u{FD94}\u{FF0B}\u{FCBF}\u{207A}\u{FEAD}\u{FF0D}\u{FCAD}\u{208B}\u{0694}\u{208A}\u{063E}\u{2033}\u{FCB5}\u{FFE1}\u{FE90}\u{20A5}\u{FEB9}\u{FFE6}\u{0778}\u{09FB}\u{FD82}\u{20A6}\u{FD31}\u{FF1A}\u{06FB}\u{FF1A}\u{FB60}\u{FF1A}\u{0684}\u{060C}\u{072B}\u{060C}\u{FBC1}\u{FE55}\u{FBD3}\u{000B}\u{FC18}\u{0009}\u{06AA}\u{0009}\u{FD71}\u{000B}\u{FD95}\u{000B}\u{06E5}\u{001F}\u{FC6E}\u{2004}\u{FB6D}\u{2003}\u{06BC}\u{205F}\u{FB68}\u{3000}\u{FEC5}\u{2009}\u{FD02}\u{200A}\u{FD50}\u{22DA}\u{1EE9B}\u{22B1}\u{FD95}\u{1F83C}\u{06A1}\u{10157}\u{FCBD}\u{269C}\u{FD10}\u{2398}\u{FD80}\u{2066}\u{FBE1}\u{2066}\u{FB95}\u{2066}\u{068D}\u{2066}\u{FDB4}\u{2066}\u{FD3B}\u{2066}\u{FBD8}\u{2067}\u{FE77}\u{2067}\u{FEE2}\u{2067}\u{078D}\u{2067}\u{1EEB7}\u{2067}\u{079D}\u{2067}\u{FCA7}\u{2068}\u{FCA3}\u{2068}\u{FC6A}\u{2068}\u{FE70}\u{2068}\u{FBD6}\u{2068}\u{1EE8C}\u{2068}\u{0721}\u{2069}\u{078F}\u{2069}\u{FBAD}\u{2069}\u{1EE10}\u{2069}\u{FBB2}\u{2069}\u{FB64}\u{2069}", Some(0)), BidiInfo { levels: vec![ 2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2], classes: vec![R, L, R, L, R, L, R, L, R, L, R, L, R, ES, R, ES, R, ES, R, ES, R, ES, R, ES, R, ET, R, ET, R, ET, R, ET, R, ET, R, ET, R, CS, R, CS, R, CS, R, CS, R, CS, R, CS, R, S, R, S, R, S, R, S, R, S, R, S, R, WS, R, WS, R, WS, R, WS, R, WS, R, WS, R, ON, R, ON, R, ON, R, ON, R, ON, R, ON, R, LRI, R, LRI, R, LRI, R, LRI, R, LRI, R, LRI, R, RLI, R, RLI, R, RLI, R, RLI, R, RLI, R, RLI, R, FSI, R, FSI, R, FSI, R, FSI, R, FSI, R, FSI, R, PDI, R, PDI, R, PDI, R, PDI, R, PDI, R, PDI, AL, L, AL, L, AL, L, AL, L, AL, L, AL, L, AL, ES, AL, ES, AL, ES, AL, ES, AL, ES, AL, ES, AL, ET, AL, ET, AL, ET, AL, ET, AL, ET, AL, ET, AL, CS, AL, CS, AL, CS, AL, CS, AL, CS, AL, CS, AL, S, AL, S, AL, S, AL, S, AL, S, AL, S, AL, WS, AL, WS, AL, WS, AL, WS, AL, WS, AL, WS, AL, ON, AL, ON, AL, ON, AL, ON, AL, ON, AL, ON, AL, LRI, AL, LRI, AL, LRI, AL, LRI, AL, LRI, AL, LRI, AL, RLI, AL, RLI, AL, RLI, AL, RLI, AL, RLI, AL, RLI, AL, FSI, AL, FSI, AL, FSI, AL, FSI, AL, FSI, AL, FSI, AL, PDI, AL, PDI, AL, PDI, AL, PDI, AL, PDI, AL, PDI], paragraphs: vec![ParagraphInfo { range: 0..132, level: 1 } ], })
    }

    #[test]
    fn test_reorder_line() {
        use super::{process_text, reorder_line};
        use std::borrow::Cow;
        fn reorder(s: &str) -> Cow<str> {
            let info = process_text(s, None);
            let para = &info.paragraphs[0];
            reorder_line(s, para.range.clone(), &info.levels)
        }

        fn reorder_with_para_level(s: &str, level: Option<u8>) -> Cow<str> {
            let info = process_text(s, level);
            let para = &info.paragraphs[0];
            reorder_line(s, para.range.clone(), &info.levels)
        }

        assert_eq!(reorder("abc123"), "abc123");
        assert_eq!(reorder("1.-2"), "1.-2");
        assert_eq!(reorder("1-.2"), "1-.2");
        assert_eq!(reorder("abc\u{2067}.-\u{2069}ghi"),
                           "abc\u{2067}-.\u{2069}ghi");
        assert_eq!(reorder("abc אבג"), "abc גבא");
        assert_eq!(reorder("אבג abc"), "abc גבא");
        //Numbers being weak LTR characters, cannot reorder strong RTL
        assert_eq!(reorder("123 אבג"), "גבא 123");
        //Testing for RLE Character
        //assert_eq!(reorder("\u{202B}abc אבג\u{202C}"), "\u{202B}\u{202C}גבא abc");
        //Testing neutral characters 
        assert_eq!(reorder("אבג? אבג"), "גבא ?גבא");
        assert_eq!(reorder("A אבג?"), "A גבא?");
        //Testing neutral characters with Implicit RTL Marker ---->
        //The given test highlights a possible non-conformance issue that will perhaps be fixed in the subsequent steps.
        //assert_eq!(reorder("A אבג?\u{202f}"), "A \u{202f}?גבא");
        assert_eq!(reorder("אבג abc"), "abc גבא");
        assert_eq!(reorder("abc\u{2067}.-\u{2069}ghi"),
                          "abc\u{2067}-.\u{2069}ghi");
        //assert_eq!(reorder("Hello, \u{2068}\u{202E}world\u{202C}\u{2069}!"),
                           //"Hello, \u{2068}\u{202E}\u{202C}dlrow\u{2069}!");
// * 
// * The comments below help locate where to push Automated Test Cases. Do not remove or change indentation.
// * 
//BeginInsertedTestCases: Test cases from BidiCharacterTest.txt go here
//EndInsertedTestCases: Test cases from BidiCharacterTest.txt go here

// * Test case built using default para level
assert_eq!(reorder("\u{061C}"),"\u{061C}");//BidiCharacterTest.txt Line Number:56
assert_eq!(reorder("\u{2680}\u{0028}\u{0061}\u{0029}\u{0062}\u{05D0}\u{0028}\u{0029}"),"\u{2680}\u{0028}\u{0061}\u{0029}\u{0062}\u{05D0}\u{0028}\u{0029}");//BidiCharacterTest.txt Line Number:3315
assert_eq!(reorder("\u{0061}\u{2680}\u{0028}\u{0028}\u{0029}\u{0029}"),"\u{0061}\u{2680}\u{0028}\u{0028}\u{0029}\u{0029}");//BidiCharacterTest.txt Line Number:7503
assert_eq!(reorder("\u{2680}\u{0028}\u{0061}\u{05D0}\u{0062}\u{0028}\u{005B}\u{0029}\u{005D}"),"\u{2680}\u{0028}\u{0061}\u{05D0}\u{0062}\u{0028}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:88169
assert_eq!(reorder("\u{2680}\u{0028}\u{0061}\u{2681}\u{0062}\u{005B}\u{0029}\u{005D}"),"\u{2680}\u{0028}\u{0061}\u{2681}\u{0062}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:27857
assert_eq!(reorder("\u{0061}\u{2680}\u{0028}\u{0062}\u{2681}\u{005B}\u{0029}\u{005D}"),"\u{0061}\u{2680}\u{0028}\u{0062}\u{2681}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:29109
assert_eq!(reorder("\u{0061}\u{0028}\u{0062}\u{2680}\u{0063}\u{005B}\u{0029}\u{005D}"),"\u{0061}\u{0028}\u{0062}\u{2680}\u{0063}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:29677
assert_eq!(reorder("\u{0028}\u{0028}\u{0029}\u{0028}\u{0029}"),"\u{0028}\u{0028}\u{0029}\u{0028}\u{0029}");//BidiCharacterTest.txt Line Number:59325
assert_eq!(reorder("\u{2680}\u{0028}\u{05D0}\u{0028}\u{0029}\u{0061}\u{05D1}\u{0028}\u{0029}"),"\u{0029}\u{0028}\u{05D1}\u{0061}\u{0029}\u{0028}\u{05D0}\u{0028}\u{2680}");//BidiCharacterTest.txt Line Number:64556

// * Test Case built using explicit para level
assert_eq!(reorder_with_para_level("\u{061C}", Some(0)),"\u{061C}");//BidiCharacterTest.txt Line Number:56
assert_eq!(reorder_with_para_level("\u{2680}\u{0028}\u{0061}\u{0029}\u{0062}\u{05D0}\u{0028}\u{0029}", Some(0)),"\u{2680}\u{0028}\u{0061}\u{0029}\u{0062}\u{05D0}\u{0028}\u{0029}");//BidiCharacterTest.txt Line Number:3315
assert_eq!(reorder_with_para_level("\u{0061}\u{2680}\u{0028}\u{0028}\u{0029}\u{0029}", Some(0)),"\u{0061}\u{2680}\u{0028}\u{0028}\u{0029}\u{0029}");//BidiCharacterTest.txt Line Number:7503
assert_eq!(reorder_with_para_level("\u{2680}\u{0028}\u{0061}\u{05D0}\u{0062}\u{0028}\u{005B}\u{0029}\u{005D}", Some(0)),"\u{2680}\u{0028}\u{0061}\u{05D0}\u{0062}\u{0028}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:88169
assert_eq!(reorder_with_para_level("\u{2680}\u{0028}\u{0061}\u{2681}\u{0062}\u{005B}\u{0029}\u{005D}", Some(0)),"\u{2680}\u{0028}\u{0061}\u{2681}\u{0062}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:27857
assert_eq!(reorder_with_para_level("\u{0061}\u{2680}\u{0028}\u{0062}\u{2681}\u{005B}\u{0029}\u{005D}", Some(0)),"\u{0061}\u{2680}\u{0028}\u{0062}\u{2681}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:29109
assert_eq!(reorder_with_para_level("\u{0061}\u{0028}\u{0062}\u{2680}\u{0063}\u{005B}\u{0029}\u{005D}", Some(0)),"\u{0061}\u{0028}\u{0062}\u{2680}\u{0063}\u{005B}\u{0029}\u{005D}");//BidiCharacterTest.txt Line Number:29677
assert_eq!(reorder_with_para_level("\u{0028}\u{0028}\u{0029}\u{0028}\u{0029}", Some(0)),"\u{0028}\u{0028}\u{0029}\u{0028}\u{0029}");//BidiCharacterTest.txt Line Number:59325
assert_eq!(reorder_with_para_level("\u{2680}\u{0028}\u{05D0}\u{0028}\u{0029}\u{0061}\u{05D1}\u{0028}\u{0029}", Some(1)),"\u{0029}\u{0028}\u{05D1}\u{0061}\u{0029}\u{0028}\u{05D0}\u{0028}\u{2680}");//BidiCharacterTest.txt Line Number:64556
    }

    #[test]
    fn test_is_ltr() {
        use super::is_ltr;
        assert_eq!(is_ltr(10), true);
        assert_eq!(is_ltr(11), false);
        assert_eq!(is_ltr(20), true);
    }

    #[test]
    fn test_is_rtl() {
        use super::is_rtl;
        assert_eq!(is_rtl(13), true);
        assert_eq!(is_rtl(11), true);
        assert_eq!(is_rtl(20), false);
    }

    #[test]
    fn test_removed_by_x9() {
        use prepare::removed_by_x9;
        let rem_classes = &[RLE, LRE, RLO, LRO, PDF, BN];
        let not_classes = &[L, RLI, AL, LRI, PDI];
        for x in rem_classes {
            assert_eq!(removed_by_x9(*x), true);
        }
        for x in not_classes {
            assert_eq!(removed_by_x9(*x), false);
        }
    }

    #[test]
    fn test_not_removed_by_x9() {
        use prepare::not_removed_by_x9;
        let non_x9_classes = &[L, R, AL, EN, ES, ET, AN, CS, NSM, B, S, WS, ON, LRI, RLI, FSI, PDI];
        for x in non_x9_classes {
            assert_eq!(not_removed_by_x9(&x), true);
        }
    }
}