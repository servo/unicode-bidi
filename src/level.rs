// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Embedding Levels and Related Types
//!
//! http://www.unicode.org/reports/tr9/#BD2

#![forbid(unsafe_code)]

use char_data::BidiClass;

/// Maximum depth of the directional status stack.
pub const MAX_DEPTH: u8 = 125;

/// Embedding Level
///
/// Embedding Levels are numbers, where even values denote a left-to-right (LTR) direction and odd
/// values a right-to-left (RTL).
///
/// http://www.unicode.org/reports/tr9/#BD2
#[derive(Copy, Clone, Debug, Ord, Eq, PartialEq, PartialOrd)]
pub struct Level(pub u8);

impl Level {
    // == Inquiries ==

    /// Levels from 0 through MAX_DEPTH are valid.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0 <= MAX_DEPTH
    }

    /// If this level is left-to-right.
    #[inline]
    pub fn is_ltr(&self) -> bool {
        self.0 % 2 == 0
    }

    /// If this level is right-to-left.
    #[inline]
    pub fn is_rtl(&self) -> bool {
        self.0 % 2 == 1
    }

    // == Mutators ==

    /// Raise level by `amount`.
    #[inline]
    pub fn raise(&mut self, amount: u8) {
        debug_assert!(self.0 as i16 + amount as i16 <= MAX_DEPTH as i16);
        self.0 += amount;
    }

    /// Lower level by `amount`.
    #[inline]
    pub fn lower(&mut self, amount: u8) {
        debug_assert!(self.0 as i16 - amount as i16 >= 0);
        self.0 -= amount;
    }

    // == Helpers ==

    /// The next LTR (even) level greater than this.
    #[inline]
    pub fn get_next_ltr_level(&self) -> Level {
        Level((self.0 + 2) & !1)
    }

    /// The next RTL (odd) level greater than this.
    #[inline]
    pub fn get_next_rtl_level(&self) -> Level {
        Level((self.0 + 1) | 1)
    }

    /// The lowest RTL (odd) level greater than or equal to this.
    #[inline]
    pub fn get_lowest_rtl_level_ge(&self) -> Level {
        Level(self.0 | 1)
    }

    /// Generate a character type based on a level (as specified in steps X10 and N2).
    #[inline]
    pub fn gen_bidi_class(&self) -> BidiClass {
        if self.is_rtl() {
            BidiClass::R
        } else {
            BidiClass::L
        }
    }

    #[cfg(test)]
    pub fn gen_vec(v: &[u8]) -> Vec<Level> {
        v.iter().map(|&x| Level(x)).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_is_valid() {
        assert_eq!(Level(0).is_valid(), true);
        assert_eq!(Level(1).is_valid(), true);
        assert_eq!(Level(10).is_valid(), true);
        assert_eq!(Level(11).is_valid(), true);
        assert_eq!(Level(124).is_valid(), true);
        assert_eq!(Level(125).is_valid(), true);
        assert_eq!(Level(126).is_valid(), false);
        assert_eq!(Level(255).is_valid(), false);
    }

    #[test]
    fn test_is_ltr() {
        assert_eq!(Level(0).is_ltr(), true);
        assert_eq!(Level(1).is_ltr(), false);
        assert_eq!(Level(10).is_ltr(), true);
        assert_eq!(Level(11).is_ltr(), false);
        assert_eq!(Level(124).is_ltr(), true);
        assert_eq!(Level(125).is_ltr(), false);
    }

    #[test]
    fn test_is_rtl() {
        assert_eq!(Level(0).is_rtl(), false);
        assert_eq!(Level(1).is_rtl(), true);
        assert_eq!(Level(10).is_rtl(), false);
        assert_eq!(Level(11).is_rtl(), true);
        assert_eq!(Level(124).is_rtl(), false);
        assert_eq!(Level(125).is_rtl(), true);
    }

    #[test]
    fn test_gen_vec() {
        assert_eq!(
            Level::gen_vec(&[0, 1, 125]),
            vec![Level(0), Level(1), Level(125)]
        );
    }
}
