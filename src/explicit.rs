// Copyright 2015 The Servo Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! 3.3.2 Explicit Levels and Directions
//!
//! http://www.unicode.org/reports/tr9/#Explicit_Levels_and_Directions

use super::char_data::BidiClass;
use super::level::Level;

use BidiClass::*;

/// Compute explicit embedding levels for one paragraph of text (X1-X8).
///
/// `classes[i]` must contain the BidiClass of the char at byte index `i`,
/// for each char in `text`.
pub fn compute(
    text: &str,
    para_level: Level,
    initial_classes: &[BidiClass],
    levels: &mut [Level],
    classes: &mut [BidiClass],
) {
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
                let char_is_rtl = match initial_classes[i] {
                    RLE | RLO | RLI => true,
                    _ => false,
                };

                let last_level = stack.last().level;
                let new_level = if char_is_rtl {
                    last_level.new_explicit_next_rtl()
                } else {
                    last_level.new_explicit_next_ltr()
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

                if new_level.is_ok() && overflow_isolate_count == 0 &&
                   overflow_embedding_count == 0 {
                    let new_level = new_level.unwrap();
                    stack.push(
                        new_level,
                        match initial_classes[i] {
                            RLO => OverrideStatus::RTL,
                            LRO => OverrideStatus::LTR,
                            RLI | LRI | FSI => OverrideStatus::Isolate,
                            _ => OverrideStatus::Neutral,
                        },
                    );
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
                            _ => continue,
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
                    continue;
                }
                if overflow_embedding_count > 0 {
                    overflow_embedding_count -= 1;
                    continue;
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
            levels[i + j] = levels[i];
            classes[i + j] = classes[i];
        }
    }
}

/// Entries in the directional status stack:
struct Status {
    level: Level,
    status: OverrideStatus,
}

#[derive(PartialEq)]
enum OverrideStatus {
    Neutral,
    RTL,
    LTR,
    Isolate,
}

struct DirectionalStatusStack {
    vec: Vec<Status>,
}

impl DirectionalStatusStack {
    fn new() -> Self {
        DirectionalStatusStack { vec: Vec::with_capacity(Level::max_explicit_depth() as usize + 2) }
    }
    fn push(&mut self, level: Level, status: OverrideStatus) {
        self.vec
            .push(
                Status {
                    level: level,
                    status: status,
                },
            );
    }
    fn last(&self) -> &Status {
        self.vec.last().unwrap()
    }
}
