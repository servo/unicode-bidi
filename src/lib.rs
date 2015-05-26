// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod tables;
pub use tables::UNICODE_VERSION;
pub use tables::bidi::{BidiClass, bidi_class};

#[cfg(test)]
mod test {
    use super::bidi_class;
    use super::BidiClass::*;

    #[test]
    fn test_bidi_class() {
        assert_eq!(bidi_class('c'), L);
        assert_eq!(bidi_class('\u{05D1}'), R);
        assert_eq!(bidi_class('\u{0627}'), AL);
    }
}
