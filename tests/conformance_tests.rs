// Copyright 2014 The html5ever Project Developers. See the
// COPYRIGHT file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg(test)]

extern crate unicode_bidi;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use unicode_bidi::{bidi_class, BidiInfo, format_chars, Level, process_text};

const TEST_DATA_DIR: &str = "tests/data";
const BASE_TEST_FILE_NAME: &str = "BidiTest.txt";
//const CHAR_TEST_FILE_NAME: &str = "BidiCharacterTest.txt";

fn open_test_file(filename: &str) -> File {
    let path = Path::new(TEST_DATA_DIR).join(filename);
    return File::open(&path).unwrap();
}

#[derive(Debug)]
struct Fail {
    pub levels: Vec<String>,
    pub ordering: Vec<String>,
    pub input: Vec<String>,
    pub input_chars: String,
    pub para_level: Option<Level>,
}

#[test]
#[should_panic(expected = "60494 test cases failed! (196253 passed)")]
fn base_tests() {
    let file = open_test_file(BASE_TEST_FILE_NAME);
    let read = BufReader::new(file);

    // State
    let mut passed_num: i32 = 0;
    let mut fails: Vec<Fail> = Vec::new();
    let mut set_levels: Vec<String> = Vec::new();
    let mut set_ordering: Vec<String> = Vec::new();
    //let mut line_num = 0;

    for line in read.lines().map(|x| x.unwrap()) {
        //line_num += 1;
        let line = line.trim();

        // Empty and comment lines
        if line.is_empty() || line.starts_with('#') {
            // Ignore
            continue;
        }

        // State setting lines
        if line.starts_with('@') {
            let tokens: Vec<String> = line.split_whitespace().map(|x| x.to_owned()).collect();
            let (setting, values) = (tokens[0].as_ref(), tokens[1..].to_vec());
            match setting {
                "@Levels:" => {
                    set_levels = values.to_owned();
                }
                "@Reorder:" => {
                    set_ordering = values.to_owned();
                }
                _ => {
                    // Ignore, to allow some degree of forward compatibility
                }
            }
            continue;
        }

        // Data lines
        {
            // Levels and ordering need to be set before any data line
            assert!(set_levels.len() > 0);
            assert!(set_ordering.len() <= set_levels.len());

            let pieces: Vec<&str> = line.split(';').collect();
            let input: Vec<&str> = pieces[0].split_whitespace().collect();
            let bitset: u8 = pieces[1].trim().parse().unwrap();
            assert!(input.len() > 0);
            assert!(bitset > 0);

            let input_chars = get_sample_string_from_bidi_classes(input.to_owned());

            for para_level in gen_para_levels(bitset) {
                let bidi_info = process_text(&input_chars, para_level);

                // Levels
                let exp_levels: Vec<&str> = set_levels.iter().map(|x| x.as_ref()).collect();
                let levels = gen_levels_list_from_bidi_info(&input_chars, &bidi_info);
                if levels != exp_levels {
                    fails.push(
                        Fail {
                            levels: set_levels.to_owned(),
                            ordering: set_ordering.to_owned(),
                            input: input.iter().map(|x| x.to_string()).collect(),
                            input_chars: input_chars.to_owned(),
                            para_level,
                        },
                    );
                } else {
                    passed_num += 1;
                }

                // Ordering
                // TODO: Add reorder map to API output and test the map here
            }
        }
    }

    if fails.len() > 0 {
        // TODO: Show a list of failed cases when the number is less than 1K
        panic!(
            "{} test cases failed! ({} passed) {{\n\
            \n\
            0: {:?}\n\
            \n\
            ...\n\
            \n\
            {}: {:?}\n\
            \n\
            }}",
            fails.len(),
            passed_num,
            fails[0],
            fails.len() - 1,
            fails[fails.len() - 1],
        );
    }
}

fn gen_para_levels(bitset: u8) -> Vec<Option<Level>> {
    /// Values: auto-LTR, LTR, RTL
    // TODO: Support auto-RTL
    // FIXME: Convert back to `const` when `const fn` becomes possible
    let para_level_values: &[Option<Level>] = &[None, Some(Level::ltr()), Some(Level::rtl())];
    assert!(bitset < (1 << para_level_values.len()));

    (0..3)
        .filter(|bit| bitset & (1u8 << bit) == 1)
        .map(|idx| para_level_values[idx])
        .collect()
}

/// We need to collaps levels to one-per-character from one-per-byte format.
///
/// TODO: Move to impl BidiInfo as pub api
fn gen_levels_list_from_bidi_info(input_chars: &str, bidi_info: &BidiInfo) -> Vec<Level> {
    input_chars
        .char_indices()
        .map(|(i, _)| bidi_info.levels[i])
        .collect()
}

fn get_sample_string_from_bidi_classes(class_names: Vec<&str>) -> String {
    class_names
        .iter()
        .map(|class_name| gen_char_from_bidi_class(class_name))
        .collect()
}

/// TODO: Auto-gen in tables.rs ?
fn gen_char_from_bidi_class(class_name: &str) -> char {
    match class_name {
        "AL" => '\u{060b}',
        "AN" => '\u{0605}',
        "B" => '\u{000a}',
        "BN" => '\u{0000}',
        "CS" => '\u{002c}',
        "EN" => '\u{0039}',
        "ES" => '\u{002b}',
        "ET" => '\u{0023}',
        "FSI" => format_chars::FSI,
        "L" => '\u{0041}',
        "LRE" => format_chars::LRE,
        "LRI" => format_chars::LRI,
        "LRO" => format_chars::LRO,
        "NSM" => '\u{0300}',
        "ON" => '\u{0021}',
        "PDF" => format_chars::PDF,
        "PDI" => format_chars::PDI,
        "R" => '\u{0590}',
        "RLE" => format_chars::RLE,
        "RLI" => format_chars::RLI,
        "RLO" => format_chars::RLO,
        "S" => '\u{0009}',
        "WS" => '\u{000c}',
        &_ => panic!("Invalid Bidi_Class name: {}", class_name),
    }
}

#[test]
fn test_gen_char_from_bidi_class() {
    use unicode_bidi::BidiClass::*;
    for class in vec![
        AL,
        AN,
        B,
        BN,
        CS,
        EN,
        ES,
        ET,
        FSI,
        L,
        LRE,
        LRI,
        LRO,
        NSM,
        ON,
        PDF,
        PDI,
        R,
        RLE,
        RLI,
        RLO,
        S,
        WS,
    ] {
        let class_name = format!("{:?}", class);
        let sample_char = gen_char_from_bidi_class(&class_name);
        assert_eq!(bidi_class(sample_char), class);
    }
}
