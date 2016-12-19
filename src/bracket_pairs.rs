//pub mod lib;
pub mod tables;

//pub mod lib;

mod bracket_pair_resolver{

	pub use std::vec::Vec;
	pub use tables::{BidiClass, bidi_class, UNICODE_VERSION};
	pub use brackets::{BracketType, bracket_type, pair_id};
	//pub use lib::prepare::IsolatingRunSequence;
	use super::prepare::IsolatingRunSequence;
	pub use tables::BidiClass::*;
	pub use brackets::BracketType::*;

	mod bracket_pair {
		use std::cmp::Ordering;
		#[derive(Debug, Copy, Clone)]
		pub struct BracketPair {
		    pub ich_opener: u8,
		    pub ich_closer: u8
		}
		impl Ord for BracketPair {
		    fn cmp(&self, other: &Self) -> Ordering {
		        self.ich_opener.cmp(&other.ich_opener)
		    }
		}
		impl PartialOrd for BracketPair {
		    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		        Some(self.cmp(other))
		    }
		}
		impl PartialEq for BracketPair {
		    fn eq(&self, other: &Self) -> bool {
		        (self.ich_opener, self.ich_closer) == (other.ich_opener, other.ich_closer)
		    }
		}
		impl Eq for BracketPair { }
	}

	fn locate_brackets(indexes: &[char]) -> Vec<bracket_pair::BracketPair> {
		let mut bracket_pair_list = Vec::<bracket_pair::BracketPair>::new();
		let mut stack_index = Vec::<u8>::new();
		let mut stack_char = Vec::<char>::new();
		//let mut next_position_on_stack:i32 = -1;
		//next_position_on_stack = next_position_on_stack - 1;

		//println!("{},{}, {}", next_position_on_stack, stack_char.len(), stack_index.len());
		'outer_loop: for index in 0..indexes.len() {
			let case = bracket_type(indexes[index as usize]);
			//print!("\nFor index:{} indexes[index]:, {}: pair_values[indexes[index]:{}",index, indexes[index], case);
			if case == Open {
				//println!("{}", next_position_on_stack);
				//next_position_on_stack = next_position_on_stack + 1;
	        	stack_index.push(index as u8);
	        	stack_char.push(indexes[index as usize] as char);
	        	//print!("-->Opening Bracket, {}", indexes[index]);
			  	}
		  	else if case == Close {
				  	//print!("-->Closing Bracket, {}", indexes[index]);
				  	if stack_index.len()==0 { // no opening bracket exists
				  		continue 'outer_loop;
				  	}//search the stack
				  	for rev_index in (0..stack_index.len()).rev(){ //iterate down the stack
				  		if pair_id(indexes[index as usize])==pair_id(stack_char[rev_index as usize]) { // match found
				  			let new_br_pair: bracket_pair::BracketPair = bracket_pair::BracketPair {ich_opener:stack_index[rev_index as usize], ich_closer:index as u8}; //make a new BracketPair
				  			stack_index.remove(rev_index);
				  			stack_char.remove(rev_index);
				  			bracket_pair_list.push(new_br_pair);
				  			println!("!!Added to list[{}, {}]", new_br_pair.ich_opener, new_br_pair.ich_closer);
				  		}
			  		}
				}
			else if case == None {
	  			//print!("-->Not a Bracket, {}", indexes[index]);
		 	}
		}
		bracket_pair_list.sort();
		bracket_pair_list//return
	}

	fn is_strong_type_by_n0(class: BidiClass) -> bool {
		class == R || class == L
	}

	fn return_strong_type_by_n0(index: u8, indexes: &[char]) -> BidiClass {
		let c=bidi_class(indexes[index as usize]);
		match c {
			EN => R,
			AN => R,
			AL => R,
			R  => R,
			L  => L,
			_  => ON
		}
	}

	fn classify_pair_content(indexes: &[char], curr_pair: 
		bracket_pair::BracketPair, dir_embed: BidiClass) -> BidiClass {
		let mut dir_opposite = ON;
		for pos in curr_pair.ich_opener+1..curr_pair.ich_closer{
			//println!("return_strong_type_by_n0({}) is ON? {}", indexes[pos as usize], return_strong_type_by_n0(pos, indexes)==ON)
			let dir_found = return_strong_type_by_n0(pos, indexes);
			if is_strong_type_by_n0(dir_found){
				if dir_found == dir_embed{
					return dir_embed;
				} 
				dir_opposite = dir_found;
			}
		}
		//Return Opposite direction, unless no opposite direction found
		dir_opposite
	}

	fn first_strong_class_before_pair(indexes: &[char], 
		curr_pair: bracket_pair::BracketPair) -> BidiClass {
		let mut dir_before = ON;
		'for_loop: for index in (0..curr_pair.ich_closer).rev() {
			let dir_found = return_strong_type_by_n0(index, indexes);
			if dir_found != ON{
				dir_before = dir_found;
				break 'for_loop;
			}
		}
		dir_before
	}

	fn resolve_bracket_pair(indexes: &[char], classes: &mut [BidiClass], dir_embed: BidiClass,
				 sos: &i8, curr_pair: bracket_pair::BracketPair) {
		println!("Trying to resolve {}--{}", curr_pair.ich_opener, curr_pair.ich_closer);
		let mut set_direction = true;
		let mut dir_of_pair = classify_pair_content(indexes, curr_pair, dir_embed);
		if  dir_of_pair == ON {
			set_direction = false;
		}
		if dir_of_pair != dir_embed {
			let dir_before = first_strong_class_before_pair(indexes, curr_pair);
			if dir_before == dir_embed || dir_before == ON {
				dir_of_pair = dir_embed
			}
		}
		if set_direction == true{
			set_dir_of_br_pair(classes, curr_pair, dir_of_pair);
		}
	}

	fn set_dir_of_br_pair( classes: &mut [BidiClass], br_pair: bracket_pair::BracketPair, 
					dir_to_be_set: BidiClass) {
		classes[br_pair.ich_opener as usize] = dir_to_be_set;
		classes[br_pair.ich_closer as usize] = dir_to_be_set;
	}

	pub fn resolve_all_paired_brackets(indexes: &[char], classes: &mut [BidiClass], 
					sos: &i8, level: &i8) {
		let bracket_pair_list= locate_brackets(indexes);
		let dir_embed:BidiClass = 
		if 1 == (level & 1) {
			R
		} else{
			L
		};
		for br_pair in bracket_pair_list {
			println!("resolving {}--{}",br_pair.ich_opener, br_pair.ich_closer);
			resolve_bracket_pair(indexes, classes, dir_embed, sos, br_pair);
		}
	}

	pub fn resolve_n0(sequence: &IsolatingRunSequence, classes: &mut [BidiClass]
			,level: &i8) {
		resolve_all_paired_brackets(&sequence.runs.iter(), &mut classes,
			&sequence.sos, &level);
	}
}

fn main() {
// 	//println!("x {}", x)
// 	//something(2);
// 	//					   [(, [, }, {, ], )];
// 	//let indexes: [ char; 6] = [0, 1, 2, 3, 4, 5];
// 	//let string = String::new("\u{0028}\u{0061}\u{005B}\u{005B}\u{005D}\u{05D0}\u{005D}\u{007B}\u{0028}\u{005B}\u{005D}\u{2680}\u{05D1}\u{007D}\u{0029}\u{0029}\u{05D2}");
// 	//let indexes = string.char_indices();
// 	let indexes = ['\u{0061}', '\u{0028}', '\u{05D0}', '\u{005B}', '\u{05D1}', '\u{005D}', '\u{0021}','\u{0029}','\u{0062}'];
// 	//indexes[-1] = 1;
// 	// indexes[1] = 2;
// 	// something(&indexes);
// 	let mut classes: [tables::BidiClass; 11] = [tables::BidiClass::R;11];// = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
// 	let sos: i8 = 0;
// 	let level: i8 = 0;
// 	bracket_pair_resolver::resolve_all_paired_brackets(&indexes, &mut classes, &sos, &level);
// 	//let c = bracket_type(indexes[0]);
// 	//println!("c:{}", c==Open);





}