#!/usr/bin/env python
#
# Based on src/etc/unicode.py from Rust 1.2.0.
#
# Copyright 2011-2013 The Rust Project Developers.
# Copyright 2015 The Servo Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.


import fileinput, re, os, sys, operator
from bidiCharacterTestParser import parse_all_test_cases_from_BidiCharacterTest_txt
from bidiCharacterTestParser import delete_all_lines_between_markers
from bidiTest import fetch_BidiTest_txt_test_cases

preamble = '''// NOTE:
// The following code was generated by "tools/generate.py". do not edit directly

#![allow(missing_docs, non_upper_case_globals, non_snake_case)]
'''

# these are the surrogate codepoints, which are not valid rust characters
surrogate_codepoints = (0xd800, 0xdfff)

def fetch(f):
    if not os.path.exists(os.path.basename(f)):
        os.system("curl -O http://www.unicode.org/Public/UNIDATA/%s"
                  % f)

    if not os.path.exists(os.path.basename(f)):
        sys.stderr.write("cannot load %s" % f)
        exit(1)

def is_surrogate(n):
    return surrogate_codepoints[0] <= n <= surrogate_codepoints[1]

def load_unicode_data(f):
    fetch(f)
    udict = {};

    range_start = -1;
    for line in fileinput.input(f):
        data = line.split(';');
        if len(data) != 15:
            continue
        cp = int(data[0], 16);
        if is_surrogate(cp):
            continue
        if range_start >= 0:
            for i in xrange(range_start, cp):
                udict[i] = data;
            range_start = -1;
        if data[1].endswith(", First>"):
            range_start = cp;
            continue;
        udict[cp] = data;

    # Mapping of code point to Bidi_Class property:
    bidi_class = {}

    for code in udict:
        [code_org, name, gencat, combine, bidi,
         decomp, deci, digit, num, mirror,
         old, iso, upcase, lowcase, titlecase ] = udict[code];

        if bidi not in bidi_class:
            bidi_class[bidi] = []
        bidi_class[bidi].append(code)

    # Default Bidi_Class for unassigned codepoints.
    # http://www.unicode.org/Public/UNIDATA/extracted/DerivedBidiClass.txt
    default_ranges = [
            (0x0600, 0x07BF, "AL"), (0x08A0, 0x08FF, "AL"),
            (0xFB50, 0xFDCF, "AL"), (0xFDF0, 0xFDFF, "AL"),
            (0xFE70, 0xFEFF, "AL"), (0x1EE00, 0x0001EEFF, "AL"),

            (0x0590, 0x05FF, "R"), (0x07C0, 0x089F, "R"),
            (0xFB1D, 0xFB4F, "R"), (0x00010800, 0x00010FFF, "R"),
            (0x0001E800, 0x0001EDFF, "R"), (0x0001EF00, 0x0001EFFF, "R"),

            (0x20A0, 0x20CF, "ET")]

    for (start, end, default) in default_ranges:
        for code in range(start, end+1):
            if not code in udict:
                bidi_class[default].append(code)

    bidi_class = group_cats(bidi_class)
    return bidi_class

def group_cats(cats):
    cats_out = []
    for cat in cats:
        cats_out.extend([(x, y, cat) for (x, y) in group_cat(cats[cat])])
    cats_out.sort(key=lambda w: w[0])
    return (sorted(cats.keys()), cats_out)

def group_cat(cat):
    cat_out = []
    letters = sorted(set(cat))
    cur_start = letters.pop(0)
    cur_end = cur_start
    for letter in letters:
        assert letter > cur_end, \
            "cur_end: %s, letter: %s" % (hex(cur_end), hex(letter))
        if letter == cur_end + 1:
            cur_end = letter
        else:
            cat_out.append((cur_start, cur_end))
            cur_start = cur_end = letter
    cat_out.append((cur_start, cur_end))
    return cat_out

def format_table_content(f, content, indent):
    line = " "*indent
    first = True
    for chunk in content.split(","):
        if len(line) + len(chunk) < 98:
            if first:
                line += chunk
            else:
                line += ", " + chunk
            first = False
        else:
            f.write(line + ",\n")
            line = " "*indent + chunk
    f.write(line)

def escape_char(c):
    return "'\\u{%x}'" % c

def emit_table(f, name, t_data, t_type = "&'static [(char, char)]", is_pub=True,
        pfun=lambda x: "(%s,%s)" % (escape_char(x[0]), escape_char(x[1]))):
    pub_string = ""
    if is_pub:
        pub_string = "pub "
    f.write("    %sconst %s: %s = &[\n" % (pub_string, name, t_type))
    data = ""
    first = True
    for dat in t_data:
        if not first:
            data += ","
        first = False
        data += pfun(dat)
    format_table_content(f, data, 8)
    f.write("\n    ];\n\n")

def emit_bidi_module(f, bidi_class, cats):
    f.write("""pub use self::BidiClass::*;

    #[allow(non_camel_case_types)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    /// Represents the Unicode character property **Bidi_Class**, also known as
    /// the *bidirectional character type*.
    ///
    /// Use the `bidi_class` function to look up the BidiClass of a code point.
    ///
    /// http://www.unicode.org/reports/tr9/#Bidirectional_Character_Types
    pub enum BidiClass {
""")
    for cat in cats:
        f.write("        " + cat + ",\n")
    f.write("""    }

    fn bsearch_range_value_table(c: char, r: &'static [(char, char, BidiClass)]) -> BidiClass {
        use ::std::cmp::Ordering::{Equal, Less, Greater};
        match r.binary_search_by(|&(lo, hi, _)| {
            if lo <= c && c <= hi { Equal }
            else if hi < c { Less }
            else { Greater }
        }) {
            Ok(idx) => {
                let (_, _, cat) = r[idx];
                cat
            }
            // UCD/extracted/DerivedBidiClass.txt: "All code points not explicitly listed
            // for Bidi_Class have the value Left_To_Right (L)."
            Err(_) => L
        }
    }

    /// Find the BidiClass of a single char.
    pub fn bidi_class(c: char) -> BidiClass {
        bsearch_range_value_table(c, bidi_class_table)
    }

""")

    emit_table(f, "bidi_class_table", bidi_class, "&'static [(char, char, BidiClass)]",
        pfun=lambda x: "(%s,%s,%s)" % (escape_char(x[0]), escape_char(x[1]), x[2]),
        is_pub=False)

if __name__ == "__main__":
    os.chdir("../src/") # changing download path to /unicode-bidi/src/
    r = "tables.rs"
    #Delete Pre-Inserted Test Cases
    delete_all_lines_between_markers("lib.rs", "//BeginInsertedTestCases", "//EndInsertedTestCases")
    print("Deleted all previous test cases...")
    # *
    # * Un-comment the next commands on the next six lines (excluding 
    # * the descriptive) comments ("Download/Parse...") to automatically insert test cases 
    # * from BidiTest.txt and BidiCharacterTest.txt
    # * 
    # * Download BidiCharacterTest.txt
    #fetch("BidiCharacterTest.txt")
    # * Parse all test cases from BidiCharacterTest.txt and place them in lib.rs after 'marker'
    #parse_all_test_cases_from_BidiCharacterTest_txt()
    #print("Inserted Test Cases from BidiCharacterTest.txt...")
    # * Download BidiTest.txt
    fetch("BidiTest.txt")
    # * Parse all test cases from BidiTest.txt and place them in lib.rs after 'marker'
    fetch_BidiTest_txt_test_cases()
    print("Inserted Test Cases from BidiTest.txt...")

    if os.path.exists(r):
        os.remove(r)
    with open(r, "w") as rf:
        # write the file's preamble
        rf.write(preamble)

        # download and parse all the data
        fetch("ReadMe.txt")
        with open("ReadMe.txt") as readme:
            pattern = "for Version (\d+)\.(\d+)\.(\d+) of the Unicode"
            unicode_version = re.search(pattern, readme.read()).groups()
        rf.write("""
/// The version of [Unicode](http://www.unicode.org/)
/// that the `bidi_class` function is based on.
pub const UNICODE_VERSION: (u64, u64, u64) = (%s, %s, %s);
""" % unicode_version)
        (bidi_cats, bidi_class) = load_unicode_data("UnicodeData.txt")
        emit_bidi_module(rf, bidi_class, bidi_cats)
