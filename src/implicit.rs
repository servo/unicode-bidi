// Copyright 2015 The Servo Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! 3.3.4 - 3.3.6. Resolve implicit levels and types.

use alloc::vec::Vec;
use core::cmp::max;
use core::ops::Range;

use super::char_data::BidiClass::{self, *};
use super::level::Level;
use super::prepare::{not_removed_by_x9, removed_by_x9, IsolatingRunSequence, LevelRun};
use super::BidiDataSource;

/// 3.3.4 Resolving Weak Types
///
/// <http://www.unicode.org/reports/tr9/#Resolving_Weak_Types>
#[cfg_attr(feature = "flame_it", flamer::flame)]
pub fn resolve_weak(sequence: &IsolatingRunSequence, processing_classes: &mut [BidiClass]) {
    // FIXME (#8): This function applies steps W1-W6 in a single pass.  This can produce
    // incorrect results in cases where a "later" rule changes the value of `prev_class` seen
    // by an "earlier" rule.  We should either split this into separate passes, or preserve
    // extra state so each rule can see the correct previous class.

    // FIXME: Also, this could be the cause of increased failure for using longer-UTF-8 chars in
    // conformance tests, like BidiTest:69635 (AL ET EN)

    let mut prev_class = sequence.sos;
    let mut last_strong_is_al = false;
    let mut et_run_indices = Vec::new(); // for W5

    // Like sequence.runs.iter().flat_map(Clone::clone), but make indices itself clonable.
    fn id(x: LevelRun) -> LevelRun {
        x
    }
    let mut indices = sequence
        .runs
        .iter()
        .cloned()
        .flat_map(id as fn(LevelRun) -> LevelRun);

    while let Some(i) = indices.next() {
        match processing_classes[i] {
            // <http://www.unicode.org/reports/tr9/#W1>
            NSM => {
                processing_classes[i] = match prev_class {
                    RLI | LRI | FSI | PDI => ON,
                    _ => prev_class,
                };
            }
            EN => {
                if last_strong_is_al {
                    // W2. If previous strong char was AL, change EN to AN.
                    processing_classes[i] = AN;
                } else {
                    // W5. If a run of ETs is adjacent to an EN, change the ETs to EN.
                    for j in &et_run_indices {
                        processing_classes[*j] = EN;
                    }
                    et_run_indices.clear();
                }
            }
            // <http://www.unicode.org/reports/tr9/#W3>
            AL => processing_classes[i] = R,

            // <http://www.unicode.org/reports/tr9/#W4>
            ES | CS => {
                let next_class = indices
                    .clone()
                    .map(|j| processing_classes[j])
                    .find(not_removed_by_x9)
                    .unwrap_or(sequence.eos);
                processing_classes[i] = match (prev_class, processing_classes[i], next_class) {
                    (EN, ES, EN) | (EN, CS, EN) => EN,
                    (AN, CS, AN) => AN,
                    (_, _, _) => ON,
                }
            }
            // <http://www.unicode.org/reports/tr9/#W5>
            ET => {
                match prev_class {
                    EN => processing_classes[i] = EN,
                    _ => et_run_indices.push(i), // In case this is followed by an EN.
                }
            }
            class => {
                if removed_by_x9(class) {
                    continue;
                }
            }
        }

        prev_class = processing_classes[i];
        match prev_class {
            L | R => {
                last_strong_is_al = false;
            }
            AL => {
                last_strong_is_al = true;
            }
            _ => {}
        }
        if prev_class != ET {
            // W6. If we didn't find an adjacent EN, turn any ETs into ON instead.
            for j in &et_run_indices {
                processing_classes[*j] = ON;
            }
            et_run_indices.clear();
        }
    }

    // W7. If the previous strong char was L, change EN to L.
    let mut last_strong_is_l = sequence.sos == L;
    for run in &sequence.runs {
        for i in run.clone() {
            match processing_classes[i] {
                EN if last_strong_is_l => {
                    processing_classes[i] = L;
                }
                L => {
                    last_strong_is_l = true;
                }
                R | AL => {
                    last_strong_is_l = false;
                }
                _ => {}
            }
        }
    }
}

/// 3.3.5 Resolving Neutral Types
///
/// <http://www.unicode.org/reports/tr9/#Resolving_Neutral_Types>
#[cfg_attr(feature = "flame_it", flamer::flame)]
pub fn resolve_neutral<D: BidiDataSource>(
    text: &str,
    data_source: &D,
    sequence: &IsolatingRunSequence,
    levels: &[Level],
    original_classes: &[BidiClass],
    processing_classes: &mut [BidiClass],
) {
    // e = embedding direction
    let e: BidiClass = levels[sequence.runs[0].start].bidi_class();
    let not_e = if e == BidiClass::L {
        BidiClass::R
    } else {
        BidiClass::L
    };
    // N0. Process bracket pairs.

    // > Identify the bracket pairs in the current isolating run sequence according to BD16.
    let bracket_pairs = identify_bracket_pairs(text, data_source, sequence, original_classes);

    // > For each bracket-pair element in the list of pairs of text positions
    //
    // Note: Rust ranges are interpreted as [start..end), be careful using `pair` directly
    // for indexing as it will include the opening bracket pair but not the closing one
    for pair in bracket_pairs {
        #[cfg(feature = "std")]
        debug_assert!(
            pair.start < processing_classes.len(),
            "identify_bracket_pairs returned a range that is out of bounds!"
        );
        #[cfg(feature = "std")]
        debug_assert!(
            pair.end < processing_classes.len(),
            "identify_bracket_pairs returned a range that is out of bounds!"
        );
        let mut found_e = false;
        let mut found_not_e = false;
        let mut class_to_set = None;
        // > Inspect the bidirectional types of the characters enclosed within the bracket pair.
        //
        // Note: the algorithm wants us to inspect the types of the *enclosed* characters,
        // not the brackets themselves, however since the brackets will never be L or R, we can
        // just scan them as well and not worry about trying to skip them in the array (they may take
        // up multiple indices in processing_classes if they're multibyte!).
        //
        // `pair` is [start, end) so we will end up processing the opening character but not the closing one.
        //
        // Note: Given that processing_classes has been modified in the previous runs, and resolve_weak
        // modifies processing_classes inconsistently at non-character-boundaries,
        // this and the later iteration will end up iterating over some obsolete classes.
        // This is fine since all we care about is looking for strong
        // classes, and strong_classes do not change in resolve_weak. The alternative is calling `.char_indices()`
        // on the text (or checking `text.get(idx).is_some()`), which would be a way to avoid hitting these
        // processing_classes of bytes not on character boundaries. This is both cleaner and likely to be faster
        // (this is worth benchmarking, though!) so we'll stick with the current approach of iterating over processing_classes.
        for &class in &processing_classes[pair.clone()] {
            if class == e {
                found_e = true;
            } else if class == not_e {
                found_not_e = true;
            }

            // if we have found a character with the class of the embedding direction
            // we can bail early
            if found_e {
                break;
            }
        }
        // > If any strong type (either L or R) matching the embedding direction is found
        if found_e {
            // > .. set the type for both brackets in the pair to match the embedding direction
            class_to_set = Some(e);
        // > Otherwise, if there is a strong type it must be opposite the embedding direction
        } else if found_not_e {
            // Therefore, test for an established context with a preceding strong type by
            // checking backwards before the opening paired bracket
            // until the first strong type (L, R, or sos) is found.
            // (see note above about processing_classes and character boundaries)
            let previous_strong = processing_classes[..pair.start]
                .iter()
                .copied()
                .rev()
                .find(|class| *class == BidiClass::L || *class == BidiClass::R)
                .unwrap_or(sequence.sos);

            // > If the preceding strong type is also opposite the embedding direction,
            // > context is established,
            // > so set the type for both brackets in the pair to that direction.
            // AND
            // > Otherwise set the type for both brackets in the pair to the embedding direction.
            // > Either way it gets set to previous_strong
            //
            // XXXManishearth perhaps the reason the spec writes these as two separate lines is
            // because sos is supposed to be handled differently?
            class_to_set = Some(previous_strong);
        }

        if let Some(class_to_set) = class_to_set {
            // update all processing classes corresponding to the start and end elements, as requested.
            // We should include all bytes of the character, not the first one.
            let start_len_utf8 = text[pair.start..].chars().next().unwrap().len_utf8();
            let end_len_utf8 = text[pair.start..].chars().next().unwrap().len_utf8();
            for class in &mut processing_classes[pair.start..pair.start + start_len_utf8] {
                *class = class_to_set;
            }
            for class in &mut processing_classes[pair.end..pair.end + end_len_utf8] {
                *class = class_to_set;
            }
            // > Any number of characters that had original bidirectional character type NSM prior to the application of
            // > W1 that immediately follow a paired bracket which changed to L or R under N0 should change to match the type of their preceding bracket.

            // This rule deals with sequences of NSMs, so we can just update them all at once, we don't need to worry
            // about character boundaries. We do need to be careful to skip the full set of bytes for the parentheses characters.
            let nsm_start = pair.start + start_len_utf8;
            for (idx, class) in original_classes[nsm_start..].iter().enumerate() {
                if *class == BidiClass::NSM {
                    processing_classes[nsm_start + idx] = class_to_set;
                } else {
                    break;
                }
            }
            let nsm_end = pair.end + end_len_utf8;
            for (idx, class) in original_classes[nsm_end..].iter().enumerate() {
                if *class == BidiClass::NSM {
                    processing_classes[nsm_end + idx] = class_to_set;
                } else {
                    break;
                }
            }
        }
        // > Otherwise, there are no strong types within the bracket pair
        // > Therefore, do not set the type for that bracket pair
    }

    // N1 and N2
    // indices of every byte in this isolating run sequence
    // XXXManishearth Note for later: is it okay to iterate over every index here, since
    // that includes char boundaries?
    let mut indices = sequence.runs.iter().flat_map(Clone::clone);
    let mut prev_class = sequence.sos;
    while let Some(mut i) = indices.next() {
        // Process sequences of NI characters.
        let mut ni_run = Vec::new();
        if is_NI(processing_classes[i]) {
            // Consume a run of consecutive NI characters.
            ni_run.push(i);
            let mut next_class;
            loop {
                match indices.next() {
                    Some(j) => {
                        i = j;
                        if removed_by_x9(processing_classes[i]) {
                            continue;
                        }
                        next_class = processing_classes[j];
                        if is_NI(next_class) {
                            ni_run.push(i);
                        } else {
                            break;
                        }
                    }
                    None => {
                        next_class = sequence.eos;
                        break;
                    }
                };
            }

            // N1-N2.
            //
            // <http://www.unicode.org/reports/tr9/#N1>
            // <http://www.unicode.org/reports/tr9/#N2>
            let new_class = match (prev_class, next_class) {
                (L, L) => L,
                (R, R)
                | (R, AN)
                | (R, EN)
                | (AN, R)
                | (AN, AN)
                | (AN, EN)
                | (EN, R)
                | (EN, AN)
                | (EN, EN) => R,
                (_, _) => e,
            };
            for j in &ni_run {
                processing_classes[*j] = new_class;
            }
            ni_run.clear();
        }
        prev_class = processing_classes[i];
    }
}

/// 3.1.3 Identifying Bracket Pairs
///
/// Returns all paired brackets in the source
///
/// <https://www.unicode.org/reports/tr9/#BD16>
fn identify_bracket_pairs<D: BidiDataSource>(
    text: &str,
    data_source: &D,
    run_sequence: &IsolatingRunSequence,
    original_classes: &[BidiClass],
) -> Vec<Range<usize>> {
    let mut ret = vec![];
    let mut stack = vec![];

    let index_range = run_sequence.text_range();
    let slice = if let Some(slice) = text.get(index_range.clone()) {
        slice
    } else {
        #[cfg(feature = "std")]
        std::debug_assert!(
            false,
            "Found broken indices in isolating run sequence: found indices {}..{} for string {:?}",
            index_range.start,
            index_range.end,
            text
        );
        return ret;
    };

    // XXXManishearth perhaps try and coalesce this into one of the earlier
    // full-string iterator runs, perhaps explicit::compute()
    for (i, ch) in slice.char_indices() {
        // all paren characters are ON
        // From BidiBrackets.txt:
        // > The Unicode property value stability policy guarantees that characters
        // > which have bpt=o or bpt=c also have bc=ON and Bidi_M=Y
        if original_classes[i] != BidiClass::ON {
            continue;
        }

        if let Some((matched, is_open)) = data_source.bidi_matched_bracket(ch) {
            if is_open {
                // If an opening paired bracket is found ...

                // ... and there is no room in the stack,
                // stop processing BD16 for the remainder of the isolating run sequence.
                if stack.len() >= 63 {
                    break;
                }
                // ... push its Bidi_Paired_Bracket property value and its text position onto the stack
                stack.push((matched, i))
            } else {
                // If a closing paired bracket is found, do the following

                // Declare a variable that holds a reference to the current stack element
                // and initialize it with the top element of the stack.
                // AND
                // Else, if the current stack element is not at the bottom of the stack
                for (stack_index, element) in stack.iter().enumerate().rev() {
                    // Compare the closing paired bracket being inspected or its canonical
                    // equivalent to the bracket in the current stack element.
                    if element.0 == ch {
                        // If the values match, meaning the two characters form a bracket pair, then

                        // Append the text position in the current stack element together with the
                        // text position of the closing paired bracket to the list.
                        ret.push(element.1..i);

                        // Pop the stack through the current stack element inclusively.
                        stack.truncate(stack_index);
                        break;
                    }
                }
            }
        }
    }
    // Sort the list of pairs of text positions in ascending order based on
    // the text position of the opening paired bracket.
    ret.sort_by_key(|r| r.start);
    ret
}

/// 3.3.6 Resolving Implicit Levels
///
/// Returns the maximum embedding level in the paragraph.
///
/// <http://www.unicode.org/reports/tr9/#Resolving_Implicit_Levels>
#[cfg_attr(feature = "flame_it", flamer::flame)]
pub fn resolve_levels(original_classes: &[BidiClass], levels: &mut [Level]) -> Level {
    let mut max_level = Level::ltr();

    assert_eq!(original_classes.len(), levels.len());
    for i in 0..levels.len() {
        match (levels[i].is_rtl(), original_classes[i]) {
            (false, AN) | (false, EN) => levels[i].raise(2).expect("Level number error"),
            (false, R) | (true, L) | (true, EN) | (true, AN) => {
                levels[i].raise(1).expect("Level number error")
            }
            (_, _) => {}
        }
        max_level = max(max_level, levels[i]);
    }

    max_level
}

/// Neutral or Isolate formatting character (B, S, WS, ON, FSI, LRI, RLI, PDI)
///
/// <http://www.unicode.org/reports/tr9/#NI>
#[allow(non_snake_case)]
fn is_NI(class: BidiClass) -> bool {
    match class {
        B | S | WS | ON | FSI | LRI | RLI | PDI => true,
        _ => false,
    }
}
