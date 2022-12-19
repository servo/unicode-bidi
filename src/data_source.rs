// Copyright 2015 The Servo Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::BidiClass;

/// This trait abstracts over a data source that is able to produce the Unicode Bidi class for a given
/// character
pub trait BidiDataSource {
    fn bidi_class(&self, c: char) -> BidiClass;
    /// If this character is a bracket according to BidiBrackets.txt,
    /// return its corresponding matched bracket, and whether or not it is an
    /// opening bracket
    ///
    /// The default implementation will pull in a small amount of hardcoded data,
    /// regardless of the `hardcoded-data` feature. This is in part for convenience
    /// (since this data is small and changes less often), and in part so that this method can be
    /// added without needing a breaking version bump.
    /// Override this method in your custom data source to prevent the use of hardcoded data.
    fn bidi_matched_bracket(&self, c: char) -> Option<(char, bool)> {
        crate::char_data::bidi_matched_bracket(c)
    }
}
