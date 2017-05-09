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

mod char_data;
mod explicit;
mod prepare;
mod implicit;

pub use char_data::{BidiClass, bidi_class, UNICODE_VERSION};
use BidiClass::*;

#[deprecated(since="0.2.6", note="please use `char_data` module instead")]
pub use char_data::tables;

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
            implicit::resolve_neutral(sequence, levels, classes);
        }
        implicit::resolve_levels(classes, levels);
        assign_levels_to_removed_chars(para.level, &initial_classes, levels);
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

#[inline]
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

/// Assign levels to characters removed by rule X9.
///
/// The levels assigned to these characters are not specified by the algorithm.  This function
/// assigns each one the level of the previous character, to avoid breaking level runs.
fn assign_levels_to_removed_chars(para_level: u8, classes: &[BidiClass], levels: &mut [u8]) {
    for i in 0..levels.len() {
        if prepare::removed_by_x9(classes[i]) {
            levels[i] = if i > 0 { levels[i-1] } else { para_level };
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_initial_scan() {
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
    fn test_process_text() {
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
    }

    #[test]
    fn test_reorder_line() {
        use std::borrow::Cow;
        fn reorder(s: &str) -> Cow<str> {
            let info = process_text(s, None);
            let para = &info.paragraphs[0];
            reorder_line(s, para.range.clone(), &info.levels)
        }
        assert_eq!(reorder("abc123"), "abc123");
        assert_eq!(reorder("1.-2"), "1.-2");
        assert_eq!(reorder("1-.2"), "1-.2");
        assert_eq!(reorder("abc אבג"), "abc גבא");
        //Numbers being weak LTR characters, cannot reorder strong RTL
        assert_eq!(reorder("123 אבג"), "גבא 123");
        //Testing for RLE Character
        assert_eq!(reorder("\u{202B}abc אבג\u{202C}"), "\u{202B}\u{202C}גבא abc");
        //Testing neutral characters
        assert_eq!(reorder("אבג? אבג"), "גבא ?גבא");
        //Testing neutral characters with special case
        assert_eq!(reorder("A אבג?"), "A גבא?");
        //Testing neutral characters with Implicit RTL Marker
        //The given test highlights a possible non-conformance issue that will perhaps be fixed in the subsequent steps.
        //assert_eq!(reorder("A אבג?\u{202f}"), "A \u{202f}?גבא");
        assert_eq!(reorder("אבג abc"), "abc גבא");
        assert_eq!(reorder("abc\u{2067}.-\u{2069}ghi"),
                           "abc\u{2067}-.\u{2069}ghi");
        assert_eq!(reorder("Hello, \u{2068}\u{202E}world\u{202C}\u{2069}!"),
                           "Hello, \u{2068}\u{202E}\u{202C}dlrow\u{2069}!");
    }

    #[test]
    fn test_is_ltr() {
        assert_eq!(is_ltr(10), true);
        assert_eq!(is_ltr(11), false);
        assert_eq!(is_ltr(20), true);
    }

    #[test]
    fn test_is_rtl() {
        assert_eq!(is_rtl(13), true);
        assert_eq!(is_rtl(11), true);
        assert_eq!(is_rtl(20), false);
    }

}
