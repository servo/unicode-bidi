// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Functions for looking up Bidi Paired Bracket properties.
//!
//! http://www.unicode.org/reports/tr9/#Paired_Brackets

/// The Bidi_Paired_Bracket_Type property of a Unicode code point
#[derive(Debug, Eq, PartialEq)]
pub enum BracketType {
    Open,
    Close,
    None
}

/// Returns the BracketType for the given code point
pub fn bracket_type(c: char) -> BracketType {
    match BRACKET_CHARS.binary_search(&c) {
        Ok(i) => if i % 2 == 0 {
            BracketType::Open
        } else {
            BracketType::Close
        },
        Err(_) => BracketType::None
    }
}

/// Returns a unique identifier for the bracket pair of the given code point
///
/// If `c` is a paired bracket character, then this returns an identifier that is unique to `c` and
/// its partner:  An opening paired bracket and a closing paired bracket share the same number if
/// and only if they are part of the same bracket pair.
///
/// Otherwise, returns `None`.
pub fn pair_id(c: char) -> Option<u8> {
    debug_assert!((BRACKET_CHARS.len() / 2) <= u8::max_value() as usize);

    match BRACKET_CHARS.binary_search(&c) {
        Ok(i) => Some((i / 2) as u8),
        Err(_) => None
    }
}

/// Ordered list of paired brackets.
///
/// Based on http://www.unicode.org/Public/8.0.0/ucd/BidiBrackets.txt
///
/// This list has the following properties: The code points in the list are sorted in ascending
/// order.  Bracket pairs are adjacent, with each opening paired bracket immediately preceding its
/// closing paired bracket.  (Therefore, all opening brackets are at even indices, while all
/// closing brackets are at odd indices.)
static BRACKET_CHARS: &'static [char] = &[
    '\u{0028}', // LEFT PARENTHESIS
    '\u{0029}', // RIGHT PARENTHESIS
    '\u{005B}', // LEFT SQUARE BRACKET
    '\u{005D}', // RIGHT SQUARE BRACKET
    '\u{007B}', // LEFT CURLY BRACKET
    '\u{007D}', // RIGHT CURLY BRACKET
    '\u{0F3A}', // TIBETAN MARK GUG RTAGS GYON
    '\u{0F3B}', // TIBETAN MARK GUG RTAGS GYAS
    '\u{0F3C}', // TIBETAN MARK ANG KHANG GYON
    '\u{0F3D}', // TIBETAN MARK ANG KHANG GYAS
    '\u{169B}', // OGHAM FEATHER MARK
    '\u{169C}', // OGHAM REVERSED FEATHER MARK
    '\u{2045}', // LEFT SQUARE BRACKET WITH QUILL
    '\u{2046}', // RIGHT SQUARE BRACKET WITH QUILL
    '\u{207D}', // SUPERSCRIPT LEFT PARENTHESIS
    '\u{207E}', // SUPERSCRIPT RIGHT PARENTHESIS
    '\u{208D}', // SUBSCRIPT LEFT PARENTHESIS
    '\u{208E}', // SUBSCRIPT RIGHT PARENTHESIS
    '\u{2308}', // LEFT CEILING
    '\u{2309}', // RIGHT CEILING
    '\u{230A}', // LEFT FLOOR
    '\u{230B}', // RIGHT FLOOR
    '\u{2329}', // LEFT-POINTING ANGLE BRACKET
    '\u{232A}', // RIGHT-POINTING ANGLE BRACKET
    '\u{2768}', // MEDIUM LEFT PARENTHESIS ORNAMENT
    '\u{2769}', // MEDIUM RIGHT PARENTHESIS ORNAMENT
    '\u{276A}', // MEDIUM FLATTENED LEFT PARENTHESIS ORNAMENT
    '\u{276B}', // MEDIUM FLATTENED RIGHT PARENTHESIS ORNAMENT
    '\u{276C}', // MEDIUM LEFT-POINTING ANGLE BRACKET ORNAMENT
    '\u{276D}', // MEDIUM RIGHT-POINTING ANGLE BRACKET ORNAMENT
    '\u{276E}', // HEAVY LEFT-POINTING ANGLE QUOTATION MARK ORNAMENT
    '\u{276F}', // HEAVY RIGHT-POINTING ANGLE QUOTATION MARK ORNAMENT
    '\u{2770}', // HEAVY LEFT-POINTING ANGLE BRACKET ORNAMENT
    '\u{2771}', // HEAVY RIGHT-POINTING ANGLE BRACKET ORNAMENT
    '\u{2772}', // LIGHT LEFT TORTOISE SHELL BRACKET ORNAMENT
    '\u{2773}', // LIGHT RIGHT TORTOISE SHELL BRACKET ORNAMENT
    '\u{2774}', // MEDIUM LEFT CURLY BRACKET ORNAMENT
    '\u{2775}', // MEDIUM RIGHT CURLY BRACKET ORNAMENT
    '\u{27C5}', // LEFT S-SHAPED BAG DELIMITER
    '\u{27C6}', // RIGHT S-SHAPED BAG DELIMITER
    '\u{27E6}', // MATHEMATICAL LEFT WHITE SQUARE BRACKET
    '\u{27E7}', // MATHEMATICAL RIGHT WHITE SQUARE BRACKET
    '\u{27E8}', // MATHEMATICAL LEFT ANGLE BRACKET
    '\u{27E9}', // MATHEMATICAL RIGHT ANGLE BRACKET
    '\u{27EA}', // MATHEMATICAL LEFT DOUBLE ANGLE BRACKET
    '\u{27EB}', // MATHEMATICAL RIGHT DOUBLE ANGLE BRACKET
    '\u{27EC}', // MATHEMATICAL LEFT WHITE TORTOISE SHELL BRACKET
    '\u{27ED}', // MATHEMATICAL RIGHT WHITE TORTOISE SHELL BRACKET
    '\u{27EE}', // MATHEMATICAL LEFT FLATTENED PARENTHESIS
    '\u{27EF}', // MATHEMATICAL RIGHT FLATTENED PARENTHESIS
    '\u{2983}', // LEFT WHITE CURLY BRACKET
    '\u{2984}', // RIGHT WHITE CURLY BRACKET
    '\u{2985}', // LEFT WHITE PARENTHESIS
    '\u{2986}', // RIGHT WHITE PARENTHESIS
    '\u{2987}', // Z NOTATION LEFT IMAGE BRACKET
    '\u{2988}', // Z NOTATION RIGHT IMAGE BRACKET
    '\u{2989}', // Z NOTATION LEFT BINDING BRACKET
    '\u{298A}', // Z NOTATION RIGHT BINDING BRACKET
    '\u{298B}', // LEFT SQUARE BRACKET WITH UNDERBAR
    '\u{298C}', // RIGHT SQUARE BRACKET WITH UNDERBAR
    '\u{298D}', // LEFT SQUARE BRACKET WITH TICK IN TOP CORNER
    '\u{298E}', // RIGHT SQUARE BRACKET WITH TICK IN BOTTOM CORNER
    '\u{298F}', // LEFT SQUARE BRACKET WITH TICK IN BOTTOM CORNER
    '\u{2990}', // RIGHT SQUARE BRACKET WITH TICK IN TOP CORNER
    '\u{2991}', // LEFT ANGLE BRACKET WITH DOT
    '\u{2992}', // RIGHT ANGLE BRACKET WITH DOT
    '\u{2993}', // LEFT ARC LESS-THAN BRACKET
    '\u{2994}', // RIGHT ARC GREATER-THAN BRACKET
    '\u{2995}', // DOUBLE LEFT ARC GREATER-THAN BRACKET
    '\u{2996}', // DOUBLE RIGHT ARC LESS-THAN BRACKET
    '\u{2997}', // LEFT BLACK TORTOISE SHELL BRACKET
    '\u{2998}', // RIGHT BLACK TORTOISE SHELL BRACKET
    '\u{29D8}', // LEFT WIGGLY FENCE
    '\u{29D9}', // RIGHT WIGGLY FENCE
    '\u{29DA}', // LEFT DOUBLE WIGGLY FENCE
    '\u{29DB}', // RIGHT DOUBLE WIGGLY FENCE
    '\u{29FC}', // LEFT-POINTING CURVED ANGLE BRACKET
    '\u{29FD}', // RIGHT-POINTING CURVED ANGLE BRACKET
    '\u{2E22}', // TOP LEFT HALF BRACKET
    '\u{2E23}', // TOP RIGHT HALF BRACKET
    '\u{2E24}', // BOTTOM LEFT HALF BRACKET
    '\u{2E25}', // BOTTOM RIGHT HALF BRACKET
    '\u{2E26}', // LEFT SIDEWAYS U BRACKET
    '\u{2E27}', // RIGHT SIDEWAYS U BRACKET
    '\u{2E28}', // LEFT DOUBLE PARENTHESIS
    '\u{2E29}', // RIGHT DOUBLE PARENTHESIS
    '\u{3008}', // LEFT ANGLE BRACKET
    '\u{3009}', // RIGHT ANGLE BRACKET
    '\u{300A}', // LEFT DOUBLE ANGLE BRACKET
    '\u{300B}', // RIGHT DOUBLE ANGLE BRACKET
    '\u{300C}', // LEFT CORNER BRACKET
    '\u{300D}', // RIGHT CORNER BRACKET
    '\u{300E}', // LEFT WHITE CORNER BRACKET
    '\u{300F}', // RIGHT WHITE CORNER BRACKET
    '\u{3010}', // LEFT BLACK LENTICULAR BRACKET
    '\u{3011}', // RIGHT BLACK LENTICULAR BRACKET
    '\u{3014}', // LEFT TORTOISE SHELL BRACKET
    '\u{3015}', // RIGHT TORTOISE SHELL BRACKET
    '\u{3016}', // LEFT WHITE LENTICULAR BRACKET
    '\u{3017}', // RIGHT WHITE LENTICULAR BRACKET
    '\u{3018}', // LEFT WHITE TORTOISE SHELL BRACKET
    '\u{3019}', // RIGHT WHITE TORTOISE SHELL BRACKET
    '\u{301A}', // LEFT WHITE SQUARE BRACKET
    '\u{301B}', // RIGHT WHITE SQUARE BRACKET
    '\u{FE59}', // SMALL LEFT PARENTHESIS
    '\u{FE5A}', // SMALL RIGHT PARENTHESIS
    '\u{FE5B}', // SMALL LEFT CURLY BRACKET
    '\u{FE5C}', // SMALL RIGHT CURLY BRACKET
    '\u{FE5D}', // SMALL LEFT TORTOISE SHELL BRACKET
    '\u{FE5E}', // SMALL RIGHT TORTOISE SHELL BRACKET
    '\u{FF08}', // FULLWIDTH LEFT PARENTHESIS
    '\u{FF09}', // FULLWIDTH RIGHT PARENTHESIS
    '\u{FF3B}', // FULLWIDTH LEFT SQUARE BRACKET
    '\u{FF3D}', // FULLWIDTH RIGHT SQUARE BRACKET
    '\u{FF5B}', // FULLWIDTH LEFT CURLY BRACKET
    '\u{FF5D}', // FULLWIDTH RIGHT CURLY BRACKET
    '\u{FF5F}', // FULLWIDTH LEFT WHITE PARENTHESIS
    '\u{FF60}', // FULLWIDTH RIGHT WHITE PARENTHESIS
    '\u{FF62}', // HALFWIDTH LEFT CORNER BRACKET
    '\u{FF63}', // HALFWIDTH RIGHT CORNER BRACKET
];

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_bracket_type() {
        assert_eq!(bracket_type('a'), BracketType::None);
        assert_eq!(bracket_type('('), BracketType::Open);
        assert_eq!(bracket_type(')'), BracketType::Close);
        assert_eq!(bracket_type('{'), BracketType::Open);
        assert_eq!(bracket_type('}'), BracketType::Close);
    }

    #[test]
    fn test_pair_id() {
        assert_eq!(pair_id('a'), None);
        assert!(pair_id('(').is_some());
        assert!(pair_id('(') == pair_id(')'));
        assert!(pair_id('[') == pair_id(']'));
        assert!(pair_id('(') != pair_id(']'));
    }
}
