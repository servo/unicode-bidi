// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! 3.3.3 Preparations for Implicit Processing
//!
//! http://www.unicode.org/reports/tr9/#Preparations_for_Implicit_Processing

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
